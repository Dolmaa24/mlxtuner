"""Run configuration. 'auto' values are resolved from the detected Mac at train time."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from .hardware import TIER_DEFAULTS, Machine

Auto = Literal["auto"]
DataFormat = Literal["auto", "messages", "alpaca", "sharegpt", "prompt_completion", "text"]


class DataConfig(BaseModel):
    path: str = Field(
        description="A .jsonl/.json/.csv file, a directory with train/valid.jsonl, or a Hub dataset id"
    )
    format: DataFormat = "auto"
    eval_fraction: float = Field(0.05, ge=0.0, lt=1.0, description="Held out as the validation set")
    max_samples: int | None = None
    system_prompt: str | None = Field(
        None, description="Added to conversations that lack a system turn"
    )
    shuffle_seed: int | None = 42
    text_field: str = "text"
    prompt_field: str = "prompt"
    completion_field: str = "completion"


class LoraConfig(BaseModel):
    type: Literal["lora", "dora", "full"] = "lora"
    rank: int = Field(8, ge=1)
    scale: float = Field(
        20.0, gt=0, description="MLX uses a direct scale (not alpha/r). 20 is mlx-lm's default."
    )
    dropout: float = Field(0.0, ge=0.0, le=1.0)
    num_layers: int | Auto = Field(
        "auto", description="How many transformer layers (from the top) get adapters. -1 = all"
    )


class TrainConfig(BaseModel):
    output: str = "adapters/run"
    epochs: float = Field(1.0, gt=0, description="Used to compute iters when iters is null")
    iters: int | None = Field(None, description="Total optimizer steps; overrides epochs")
    batch_size: int | Auto = "auto"
    max_seq_length: int | Auto = "auto"
    grad_checkpoint: bool | Auto = "auto"
    grad_accumulation_steps: int = Field(1, ge=1)
    lr: float = Field(
        1e-5, gt=0, description="mlx-lm's LoRA default; try 1e-4 for small datasets / bigger shifts"
    )
    optimizer: Literal["adam", "adamw", "sgd", "adafactor", "muon"] = "adam"
    warmup_steps: int = Field(0, ge=0, description="Linear warmup then cosine decay when > 0")
    mask_prompt: bool = Field(
        True, description="Loss only on assistant/completion tokens (ignored for text data)"
    )
    steps_per_report: int = 10
    steps_per_eval: int | None = Field(None, description="null = 4 evals per run")
    val_batches: int = Field(25, description="Validation batches per eval; -1 = whole set")
    save_every: int | None = Field(None, description="null = every eval")
    keep_checkpoints: int = Field(
        1, ge=-1, description="Numbered checkpoints to keep at the end; 0 = none, -1 = all"
    )
    seed: int = 0
    resume: str | None = Field(None, description="Path to adapters.safetensors to continue from")


class RunConfig(BaseModel):
    model: str = Field(
        description="mlx-community/* repo id, any HF model id (converted on load), or local path"
    )
    data: DataConfig
    lora: LoraConfig = Field(default_factory=LoraConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)

    @field_validator("data", mode="before")
    @classmethod
    def _data_str(cls, v: Any) -> Any:
        return {"path": v} if isinstance(v, str) else v

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: list[str] | None = None) -> RunConfig:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw, overrides)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], overrides: list[str] | None = None) -> RunConfig:
        if isinstance(raw.get("data"), str):
            # Expand the `data: train.jsonl` shorthand first, or `--set data.path=...` (and the
            # --data shortcut built on it) would hit a string where it needs a mapping.
            raw = {**raw, "data": {"path": raw["data"]}}
        if overrides:
            raw = apply_overrides(raw, overrides)
        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)

    # -- resolve 'auto' -----------------------------------------------------
    def resolved(self, machine: Machine) -> RunConfig:
        """Return a copy with every 'auto' replaced by the value for this Mac's RAM tier."""
        d = TIER_DEFAULTS[machine.tier]
        out = self.model_copy(deep=True)
        if out.train.batch_size == "auto":
            out.train.batch_size = d.batch_size
        if out.train.max_seq_length == "auto":
            out.train.max_seq_length = d.max_seq_length
        if out.train.grad_checkpoint == "auto":
            out.train.grad_checkpoint = d.grad_checkpoint
        if out.lora.num_layers == "auto":
            out.lora.num_layers = d.num_layers
        return out


_BOOLS = {"true": True, "false": False, "yes": True, "no": False, "on": True, "off": False}


def _parse_scalar(value: str) -> Any:
    """Parse a ``--set`` value without YAML 1.1's surprises (``0123`` is 123, not octal 83).

    Wrap a value in quotes to keep it a string: ``--set data.path="'0123'"``.
    """
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    low = v.lower()
    if low in {"", "null", "none", "~"}:
        return None
    if low in _BOOLS:
        return _BOOLS[low]
    try:
        return int(v, 10)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    if v[0] in "[{":  # inline lists / mappings, e.g. --set train.lr_schedule={...}
        try:
            return yaml.safe_load(v)
        except yaml.YAMLError:
            return v
    return v


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``["train.lr=1e-4", "lora.rank=16"]`` onto a nested dict (returns a copy)."""
    out = copy.deepcopy(raw)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like section.key=value, got: {item!r}")
        key, _, value = item.partition("=")
        parts = key.strip().split(".")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot set {key!r}: {p!r} is not a mapping")
        node[parts[-1]] = _parse_scalar(value.strip())
    return out


EXAMPLE_CONFIG = """\
# mlxtuner run config. 'auto' values are chosen from your Mac's RAM when training starts.
# Override anything from the CLI:  mlxtuner train config.yaml --set train.lr=1e-4

model: mlx-community/Qwen2.5-1.5B-Instruct-4bit   # see `mlxtuner models` for what fits your Mac

data:
  path: data/train.jsonl        # .jsonl/.json/.csv, a dir with train/valid.jsonl, or a Hub dataset id
  format: auto                  # auto | messages | alpaca | sharegpt | prompt_completion | text
  eval_fraction: 0.05
  # system_prompt: "You are a helpful assistant."

lora:
  type: lora                    # lora | dora | full
  rank: 8
  scale: 20.0
  num_layers: auto              # layers that get adapters (8 on 8 GB Macs, 16 above)

train:
  output: adapters/my-run
  epochs: 2
  batch_size: auto
  max_seq_length: auto
  grad_checkpoint: auto
  lr: 1.0e-5
  mask_prompt: true             # learn only from assistant turns
  keep_checkpoints: 1           # numbered checkpoints kept at the end (0 = none, -1 = all)
"""
