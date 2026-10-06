"""Evaluate an adapter: held-out loss / perplexity and sample generations, tuned vs base.

Uses mlx-lm's own ``evaluate`` and dataset classes, so the loss here is exactly the
validation loss printed during training (assistant/completion tokens only when
``mask_prompt`` is on).
"""

from __future__ import annotations

import json
import math
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rich.console import Console
from rich.table import Table

from .config import DataConfig
from .data import DataError, prepare
from .inference import base_model_of, is_adapter_dir, load_for_inference, stream_reply

console = Console()

# Below this, a loss comparison is noise rather than signal; say so instead of implying precision.
MIN_MEANINGFUL_EXAMPLES = 10


@dataclass
class EvalResult:
    path: str
    base_model: str
    kind: str
    examples: int
    mask_prompt: bool
    tuned_loss: float
    tuned_ppl: float
    base_loss: float | None = None
    base_ppl: float | None = None
    samples: list[dict[str, str]] = field(default_factory=list)
    wall_time_s: float = 0.0


def _rows_for_eval(
    data_cfg: DataConfig, work: Path, max_examples: int
) -> tuple[list[dict[str, Any]], str, str]:
    """Convert the dataset and return (rows, kind, which_split)."""
    p = prepare(data_cfg, work / "converted")
    src = p.dir / "valid.jsonl" if (p.dir / "valid.jsonl").exists() else p.dir / "train.jsonl"
    with open(src) as f:
        rows = [json.loads(line) for line in f][:max_examples]
    which = "valid split" if src.name == "valid.jsonl" else "train split (no valid split available)"
    return rows, p.kind, which


def _mlx_dataset(rows: list[dict[str, Any]], tokenizer: Any, mask_prompt: bool, work: Path) -> Any:
    """Build the same dataset object mlx-lm trains on, from already-converted rows."""
    from mlx_lm.tuner.datasets import CacheDataset, load_local_dataset

    d = work / "eval"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "valid.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    _, valid, _ = load_local_dataset(d, tokenizer, SimpleNamespace(mask_prompt=mask_prompt))
    return CacheDataset(valid)


def _loss(model: Any, dataset: Any, max_seq_length: int) -> float:
    from mlx_lm.tuner.trainer import evaluate

    return evaluate(model, dataset, batch_size=1, num_batches=-1, max_seq_length=max_seq_length)


def _prompt_messages(row: dict[str, Any]) -> list[dict[str, str]] | None:
    if "messages" in row:
        return row["messages"][:-1]
    if "prompt" in row:
        return [{"role": "user", "content": row["prompt"]}]
    return None


def _reference(row: dict[str, Any]) -> str:
    if "messages" in row:
        return row["messages"][-1]["content"]
    return row.get("completion", "")


def run_eval(
    path: str,
    data_cfg: DataConfig,
    max_seq_length: int = 2048,
    mask_prompt: bool = True,
    compare_base: bool = True,
    n_samples: int = 3,
    max_examples: int = 100,
    max_tokens: int = 128,
) -> EvalResult:
    import mlx.core as mx

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        rows, kind, which = _rows_for_eval(data_cfg, work, max_examples)
        if not rows:
            raise DataError("No examples to evaluate")
        mask = mask_prompt and kind != "text"
        console.print(
            f"scoring {len(rows)} examples from the {which}  [dim]({kind}, mask_prompt={'on' if mask else 'off'})[/]"
        )

        model, tok = load_for_inference(path)
        dataset = _mlx_dataset(rows, tok, mask, work)
        tuned_loss = _loss(model, dataset, max_seq_length)
        samples: list[dict[str, str]] = []
        for row in rows[:n_samples]:
            msgs = _prompt_messages(row)
            if msgs is None:
                break
            reply = "".join(stream_reply(model, tok, msgs, max_tokens=max_tokens, temperature=0))
            samples.append(
                {"prompt": msgs[-1]["content"], "reference": _reference(row), "tuned": reply}
            )

        base_name = base_model_of(path) if is_adapter_dir(path) else path
        result = EvalResult(
            path=path,
            base_model=base_name,
            kind=kind,
            examples=len(rows),
            mask_prompt=mask,
            tuned_loss=round(tuned_loss, 4),
            tuned_ppl=round(math.exp(tuned_loss), 3),
            samples=samples,
        )

        if compare_base and is_adapter_dir(path):
            del model
            mx.clear_cache()
            base, btok = load_for_inference(base_name)
            base_loss = _loss(base, _mlx_dataset(rows, btok, mask, work / "b"), max_seq_length)
            result.base_loss = round(base_loss, 4)
            result.base_ppl = round(math.exp(base_loss), 3)
            for s, row in zip(samples, rows[: len(samples)], strict=True):
                msgs = _prompt_messages(row)
                s["base"] = "".join(
                    stream_reply(base, btok, msgs, max_tokens=max_tokens, temperature=0)
                )

    result.wall_time_s = round(time.time() - t0, 1)
    return result


def print_result(r: EvalResult) -> None:
    table = Table(title=f"held-out loss on {r.examples} examples")
    table.add_column("model")
    table.add_column("loss", justify="right")
    table.add_column("perplexity", justify="right")
    if r.base_loss is not None:
        table.add_row(f"base: {r.base_model}", f"{r.base_loss:.4f}", f"{r.base_ppl:.3f}")
    table.add_row(f"tuned: {r.path}", f"{r.tuned_loss:.4f}", f"{r.tuned_ppl:.3f}")
    console.print(table)
    if r.base_loss is not None:
        delta = r.base_loss - r.tuned_loss
        verdict = "[green]lower[/]" if delta > 0 else "[red]higher[/]"
        console.print(f"tuned loss is {verdict} than base by {abs(delta):.4f} nats/token")
    if r.examples < MIN_MEANINGFUL_EXAMPLES:
        console.print(
            f"[yellow]caution:[/] only {r.examples} example(s) — too few to read much into these "
            "numbers. Raise data.eval_fraction, or point --data at a larger held-out file."
        )
    for i, s in enumerate(r.samples):
        console.rule(f"[dim]sample {i}")
        _labelled("bold cyan", "prompt", s["prompt"])
        _labelled("bold", "reference", s["reference"])
        if "base" in s:
            _labelled("bold yellow", "base", s["base"])
        _labelled("bold green", "tuned", s["tuned"])


def _labelled(style: str, label: str, text: str) -> None:
    console.print(f"[{style}]{label}:[/] ", end="")
    console.print(text.strip(), markup=False, highlight=False)


def save_result(r: EvalResult, out: Path) -> Path:
    out.write_text(json.dumps(asdict(r), indent=2))
    return out
