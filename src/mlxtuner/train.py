"""Drive mlx-lm's LoRA trainer from a resolved RunConfig, capturing metrics as we go."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rich.console import Console

from . import __version__
from .config import RunConfig
from .data import TEMPLATED_KINDS, DataError, Prepared, prepare
from .hardware import Machine, detect, estimate_train_gb, fits

console = Console()

# Converted train/valid.jsonl live here inside the run directory. The leading dot keeps it
# clear of anything the user put there, and prepare() refuses to delete a directory it did
# not write, so `--output .` can no longer wipe a project's own ./data.
DATA_SUBDIR = ".mlxtuner-data"


class MetricsCallback:
    """Collects what mlx-lm reports so we can write it to mlxtuner.json."""

    def __init__(self) -> None:
        self.train: list[dict[str, Any]] = []
        self.val: list[dict[str, Any]] = []

    def on_train_loss_report(self, info: dict) -> None:
        self.train.append(dict(info))

    def on_val_loss_report(self, info: dict) -> None:
        self.val.append(dict(info))

    @property
    def peak_gb(self) -> float | None:
        return max((r.get("peak_memory", 0) for r in self.train), default=None)


def build_args(cfg: RunConfig, data: Prepared, n_train: int) -> SimpleNamespace:
    """Map our config onto the argparse namespace mlx_lm.lora expects."""
    from mlx_lm.lora import CONFIG_DEFAULTS

    t, lo = cfg.train, cfg.lora
    batch_size = int(t.batch_size)
    # mlx-lm's iterate_batches drops the short final batch, so an epoch is floor(), not ceil().
    steps_per_epoch = max(1, n_train // batch_size)
    iters = t.iters or max(1, int(steps_per_epoch * t.epochs))
    steps_per_eval = t.steps_per_eval or max(1, iters // 4)
    save_every = t.save_every or steps_per_eval

    args = SimpleNamespace(**CONFIG_DEFAULTS)
    args.model = cfg.model
    args.train = True
    args.test = False
    args.data = str(data.dir)
    args.adapter_path = t.output
    args.fine_tune_type = lo.type
    args.num_layers = int(lo.num_layers)
    args.lora_parameters = {"rank": lo.rank, "dropout": lo.dropout, "scale": lo.scale}
    args.batch_size = batch_size
    args.iters = iters
    args.learning_rate = t.lr
    args.optimizer = t.optimizer
    args.max_seq_length = int(t.max_seq_length)
    args.grad_checkpoint = bool(t.grad_checkpoint)
    args.grad_accumulation_steps = t.grad_accumulation_steps
    args.steps_per_report = t.steps_per_report
    args.steps_per_eval = steps_per_eval
    args.save_every = save_every
    args.val_batches = t.val_batches
    args.seed = t.seed
    args.mask_prompt = t.mask_prompt and data.kind != "text"
    args.resume_adapter_file = t.resume
    args.report_to = None
    if t.warmup_steps > 0:
        args.lr_schedule = {
            "name": "cosine_decay",
            "warmup": t.warmup_steps,
            "warmup_init": 0.0,
            "arguments": [t.lr, iters, t.lr * 0.1],
        }
    return args


def run(cfg: RunConfig, dry_run: bool = False, machine: Machine | None = None) -> Path:
    t0 = time.time()
    machine = machine or detect()
    cfg = cfg.resolved(machine)
    out = Path(cfg.train.output)
    out.mkdir(parents=True, exist_ok=True)

    console.rule("[bold]mlxtuner train")
    console.print(
        f"[dim]mlxtuner {__version__}[/]  {machine.chip}, {machine.ram_label} (tier {machine.tier})"
    )

    # -- data ----------------------------------------------------------------
    try:
        data = prepare(cfg.data, out / DATA_SUBDIR)
    except DataError as e:
        raise SystemExit(f"data error: {e}") from None
    console.print(
        f"data: {data.n_train:,} train / {data.n_valid:,} valid  "
        f"[dim]({data.source_format} -> {data.kind}, dropped {data.dropped})[/]"
    )
    _check_batch_size(cfg, data)
    for reason, n in sorted(data.drop_reasons.items(), key=lambda kv: -kv[1])[:3]:
        console.print(f"  [yellow]dropped {n}[/]: {reason}")

    # -- plan ----------------------------------------------------------------
    args = build_args(cfg, data, data.n_train)
    n_layers = _resolve_num_layers(cfg, args)
    shown = f"all({n_layers})" if args.num_layers < 0 and n_layers else args.num_layers
    console.print(
        f"model: {cfg.model}   {cfg.lora.type} rank={cfg.lora.rank} scale={cfg.lora.scale:g} "
        f"layers={shown}"
    )
    console.print(
        f"schedule: {args.iters} iters  batch={args.batch_size}  seq={args.max_seq_length}  "
        f"lr={args.learning_rate:g}  grad_checkpoint={'on' if args.grad_checkpoint else 'off'}  "
        f"mask_prompt={'on' if args.mask_prompt else 'off'}"
    )
    est = _estimate(cfg, args, n_layers)
    if est is None and args.num_layers < 0:
        console.print(
            "[dim]memory: no estimate — num_layers=-1 and this model's layer count is not known "
            "offline[/]"
        )
    if est is not None:
        verdict = fits(machine, est)
        colour = {"yes": "green", "tight": "yellow", "no": "red"}[verdict]
        console.print(
            f"memory: ~{est} GB estimated peak  [{colour}]{verdict}[/] for {machine.ram_label}"
        )
        if verdict == "no":
            console.print(
                "[red]This will probably swap or crash.[/] Lower train.max_seq_length, train.batch_size or "
                "lora.num_layers, or pick a smaller model (`mlxtuner models`)."
            )
    _saved_config(cfg).to_yaml(out / "mlxtuner.yaml")
    if dry_run:
        console.print("[green]dry run OK[/]  (data converted, nothing trained)")
        return out

    # -- train ---------------------------------------------------------------
    import mlx.core as mx
    from mlx_lm.lora import train_model
    from mlx_lm.tuner.datasets import load_dataset
    from mlx_lm.utils import load

    console.print(f"[dim]loading {cfg.model} ...[/]")
    model, tokenizer = load(cfg.model, tokenizer_config={"trust_remote_code": True})
    if data.kind in TEMPLATED_KINDS and getattr(tokenizer, "chat_template", None) is None:
        raise SystemExit(
            f"{cfg.model} has no chat template, which mlx-lm needs for {data.kind} data; "
            "use an -Instruct model, or convert your dataset to the `text` format"
        )
    train_set, valid_set, _ = load_dataset(args, tokenizer)
    if args.num_layers < 0 or args.num_layers > len(model.layers):
        args.num_layers = len(model.layers)

    mx.reset_peak_memory()
    metrics = MetricsCallback()
    train_model(args, model, train_set, valid_set, metrics)

    # -- save metadata -------------------------------------------------------
    last_train = metrics.train[-1] if metrics.train else {}
    last_val = metrics.val[-1] if metrics.val else {}
    meta = {
        "mlxtuner_version": __version__,
        "model": cfg.model,
        "machine": {"chip": machine.chip, "ram_gb": machine.ram_gb},
        "kind": data.kind,
        "train_examples": data.n_train,
        "valid_examples": data.n_valid,
        "iters": args.iters,
        "final_train_loss": last_train.get("train_loss"),
        "final_val_loss": last_val.get("val_loss"),
        "peak_memory_gb": round(metrics.peak_gb, 2) if metrics.peak_gb else None,
        "tokens_per_second": round(last_train.get("tokens_per_second", 0), 1),
        "wall_time_s": round(time.time() - t0, 1),
        "history": {"train": metrics.train, "val": metrics.val},
    }
    (out / "mlxtuner.json").write_text(json.dumps(meta, indent=2, default=str))
    _write_readme(out, cfg, meta)
    _prune_checkpoints(out, cfg.train.keep_checkpoints)

    tl, vl = meta["final_train_loss"], meta["final_val_loss"]
    console.print(
        f"[green]done[/] in {meta['wall_time_s']}s"
        + (f"   train_loss={tl:.3f}" if tl is not None else "")
        + (f"  val_loss={vl:.3f}" if vl is not None else "")
        + (f"  peak_mem={meta['peak_memory_gb']} GB" if meta["peak_memory_gb"] else "")
    )
    console.print(f"adapter saved to [bold]{out}[/]   try it:  mlxtuner chat {out}")
    return out


def _resolve_num_layers(cfg: RunConfig, args: SimpleNamespace) -> int | None:
    """The real layer count behind ``num_layers``, or None if -1 and the model is unknown.

    ``-1`` means "every layer" to mlx-lm. Feeding that to the estimator made it *subtract*
    memory, so a run that needed twice the RAM was reported as a comfortable fit.
    """
    if args.num_layers >= 0:
        return args.num_layers
    from .inference import model_num_layers

    return model_num_layers(cfg.model)


def _saved_config(cfg: RunConfig) -> RunConfig:
    """The config written into the run directory: local data paths made absolute.

    ``mlxtuner eval <run dir>`` re-reads this file, and a relative path only resolves from the
    directory training happened to start in. Hub dataset ids are left alone.
    """
    out = cfg.model_copy(deep=True)
    p = Path(os.path.expanduser(cfg.data.path))
    if p.exists():
        out.data.path = str(p.resolve())
    return out


def _check_batch_size(cfg: RunConfig, data: Prepared) -> None:
    """mlx-lm refuses a split smaller than the batch, but only once the model is loaded."""
    batch = int(cfg.train.batch_size)
    if data.n_train < batch:
        raise SystemExit(
            f"data error: {data.n_train} training example(s) but batch_size={batch}. "
            f"Use --set train.batch_size={data.n_train} or a larger dataset."
        )
    if 0 < data.n_valid < batch:
        console.print(
            f"[yellow]skipping validation:[/] the held-out split has {data.n_valid} example(s), "
            f"fewer than batch_size={batch}. Raise data.eval_fraction, or lower train.batch_size, "
            "to get a val loss."
        )
        (data.dir / "valid.jsonl").unlink()
        data.n_valid = 0


def _prune_checkpoints(out: Path, keep: int) -> None:
    """Drop the numbered per-save copies mlx-lm leaves behind (adapters.safetensors stays)."""
    if keep < 0:
        return
    ckpts = sorted(out.glob("[0-9]" * 7 + "_adapters.safetensors"))
    for c in ckpts[: len(ckpts) - keep] if keep else ckpts:
        c.unlink()


def _estimate(cfg: RunConfig, args: SimpleNamespace, num_layers: int | None) -> float | None:
    from .hardware import MODELS

    rec = next((m for m in MODELS if m.repo == cfg.model), None)
    if rec is None or num_layers is None:
        return None
    return estimate_train_gb(
        rec.weights_gb,
        args.batch_size,
        args.max_seq_length,
        num_layers,
        args.grad_checkpoint,
        rec.vocab_k,
    )


def _write_readme(out: Path, cfg: RunConfig, meta: dict[str, Any]) -> None:
    tl, vl = meta["final_train_loss"], meta["final_val_loss"]
    lines = [
        f"# LoRA adapter for `{cfg.model}`",
        "",
        f"Trained with [mlxtuner](https://github.com/Dolmaa24/mlxtuner) {meta['mlxtuner_version']} "
        f"on {meta['machine']['chip']} ({meta['machine']['ram_gb']:g} GB).",
        "",
        "| | |",
        "|---|---|",
        f"| Base model | `{cfg.model}` |",
        f"| Method | {cfg.lora.type} rank={cfg.lora.rank} scale={cfg.lora.scale:g} layers={cfg.lora.num_layers} |",
        f"| Train / valid examples | {meta['train_examples']:,} / {meta['valid_examples']:,} |",
        f"| Iters | {meta['iters']} (batch {cfg.train.batch_size}, seq {cfg.train.max_seq_length}) |",
        f"| Learning rate | {cfg.train.lr:g} |",
        *([f"| Final train loss | {tl:.3f} |"] if tl is not None else []),
        *([f"| Final val loss | {vl:.3f} |"] if vl is not None else []),
        *([f"| Peak memory | {meta['peak_memory_gb']} GB |"] if meta["peak_memory_gb"] else []),
        "",
        "## Use",
        "",
        "```bash",
        f"mlxtuner chat {cfg.train.output}",
        f"mlxtuner fuse {cfg.train.output} --output fused-model   # standalone model",
        "```",
        "",
        "Re-run with `mlxtuner train mlxtuner.yaml`.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")
