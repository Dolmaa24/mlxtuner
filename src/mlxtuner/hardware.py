"""Detect the Mac we're on and turn that into training defaults and model recommendations."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class Machine:
    chip: str
    ram_gb: float
    apple_silicon: bool

    @property
    def ram_label(self) -> str:
        """RAM the way the Mac is sold: 8 GiB of memory is "8 GB" on the spec sheet."""
        return f"{self.ram_gb * 1e9 / 1024**3:.0f} GB"

    @property
    def tier(self) -> str:
        """Coarse RAM bucket used for defaults: 8 | 16 | 32 | 64+.

        The cut-offs are in decimal GB, because that is what psutil reports and what
        ``mx.get_peak_memory() / 1e9`` measures, while Macs are sold in GiB: an "8 GB" Mac has
        8.59 GB, a "36 GB" one has 38.65. Thresholds written as if they were GiB put the 18 GB
        M3 Pro and the 36/48 GB M3 Max a tier too high — the 36 GB machine was handed the
        64 GB+ defaults (batch 4 x 2048), which need ~40 GB for the 14B it was then offered.
        """
        if self.ram_gb <= 13:  # 8 GiB = 8.59
            return "8"
        if self.ram_gb <= 21:  # 16 GiB = 17.18, 18 GiB = 19.33
            return "16"
        if self.ram_gb <= 42:  # 24 = 25.77, 32 = 34.36, 36 = 38.65
            return "32"
        return "64+"  # 48 GiB = 51.54 and up


def _sysctl(key: str) -> str | None:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def detect() -> Machine:
    import psutil

    ram_gb = psutil.virtual_memory().total / 1e9
    chip = _sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown"
    apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    return Machine(chip=chip, ram_gb=round(ram_gb, 1), apple_silicon=apple)


@dataclass(frozen=True)
class Defaults:
    batch_size: int
    max_seq_length: int
    num_layers: int
    grad_checkpoint: bool


# Conservative starting points per RAM tier, chosen so the tier's "sweet spot" model lands in
# the 'yes'/'tight' range of estimate_train_gb. Treat as a floor, not a limit: if a run reports
# a low peak memory, raise batch_size or max_seq_length.
TIER_DEFAULTS: dict[str, Defaults] = {
    "8": Defaults(batch_size=1, max_seq_length=1024, num_layers=8, grad_checkpoint=True),
    "16": Defaults(batch_size=1, max_seq_length=2048, num_layers=16, grad_checkpoint=True),
    "32": Defaults(batch_size=2, max_seq_length=2048, num_layers=16, grad_checkpoint=True),
    "64+": Defaults(batch_size=4, max_seq_length=2048, num_layers=16, grad_checkpoint=True),
}


@dataclass(frozen=True)
class ModelRec:
    repo: str
    params_b: float
    weights_gb: float  # safetensors size of the 4-bit weights ≈ resident memory for the base model
    vocab_k: int  # vocabulary size in thousands; drives the logits term of the memory estimate
    note: str = ""


# Curated 4-bit text models from mlx-community. weights_gb and vocab_k were read from each repo's
# safetensors sizes and config.json on 2026-09-21. Multimodal repos (gemma-3-4b+) are left out.
MODELS: list[ModelRec] = [
    ModelRec(
        "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        0.5,
        0.28,
        152,
        "fastest; good for pipeline tests",
    ),
    ModelRec("mlx-community/Qwen3-0.6B-4bit", 0.6, 0.34, 152, "thinking-mode model"),
    ModelRec("mlx-community/Llama-3.2-1B-Instruct-4bit", 1.2, 0.70, 128),
    ModelRec("mlx-community/gemma-3-1b-it-4bit", 1.0, 0.73, 262, "262k vocab: trains like a 3B"),
    ModelRec(
        "mlx-community/Qwen2.5-1.5B-Instruct-4bit", 1.5, 0.87, 152, "sweet spot for 8 GB Macs"
    ),
    ModelRec("mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit", 1.5, 0.87, 152, "code"),
    ModelRec("mlx-community/Qwen3-1.7B-4bit", 1.7, 0.97, 152),
    ModelRec("mlx-community/gemma-2-2b-it-4bit", 2.6, 1.47, 256),
    ModelRec("mlx-community/Llama-3.2-3B-Instruct-4bit", 3.2, 1.81, 128),
    ModelRec("mlx-community/Qwen2.5-3B-Instruct-4bit", 3.1, 1.74, 152),
    ModelRec(
        "mlx-community/Phi-3.5-mini-instruct-4bit",
        3.8,
        2.15,
        32,
        "32k vocab: low activation memory",
    ),
    ModelRec("mlx-community/Phi-4-mini-instruct-4bit", 3.8, 2.16, 200),
    ModelRec(
        "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        7.2,
        4.08,
        33,
        "32k vocab: cheapest 7B to train",
    ),
    ModelRec("mlx-community/Qwen2.5-7B-Instruct-4bit", 7.6, 4.28, 152, "sweet spot for 16 GB Macs"),
    ModelRec("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", 7.6, 4.28, 152, "code"),
    ModelRec("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", 8.0, 4.52, 128),
    ModelRec("mlx-community/Qwen3-8B-4bit", 8.2, 4.61, 152),
    ModelRec("mlx-community/Mistral-Nemo-Instruct-2407-4bit", 12.2, 6.89, 131),
    ModelRec("mlx-community/Qwen2.5-14B-Instruct-4bit", 14.8, 8.31, 152, "32 GB Macs"),
    ModelRec("mlx-community/Qwen2.5-32B-Instruct-4bit", 32.8, 18.43, 152, "64 GB Macs"),
    ModelRec("mlx-community/Llama-3.3-70B-Instruct-4bit", 70.6, 39.69, 128, "96 GB+ Macs"),
]

# Reference point for the estimator: Qwen2.5-0.5B-4bit (0.28 GB weights, hidden 896), measured
# on an M2 8 GB at ~1000 tokens/step:
#   batch 1, 8 layers,  grad ckpt on  -> 1.90 GB
#   batch 1, 8 layers,  grad ckpt off -> 2.05 GB
#   batch 2, 8 layers,  grad ckpt on  -> 3.38 GB
#   batch 1, 24 layers, grad ckpt on  -> 2.50 GB
# Cross-check on a bigger model, same machine: Qwen2.5-1.5B-4bit (0.87 GB), batch 1, 8 layers,
# grad ckpt on, longest example 645 tokens -> 2.12 GB measured, 2.3 GB estimated.
_REF_WEIGHTS_GB = 0.28
_OVERHEAD_GB = 0.4  # runtime + optimizer state at LoRA sizes
_LOGITS_GB_PER_KTOK_PER_KVOCAB = 1.2 / 152  # fp32 logits + their gradient
_LAYER_GB_PER_KTOK = {True: 0.03, False: 0.04}  # per trained layer, at the reference hidden size


def estimate_train_gb(
    weights_gb: float,
    batch_size: int,
    max_seq_length: int,
    num_layers: int,
    grad_checkpoint: bool,
    vocab_k: int = 152,
) -> float:
    """Peak-memory estimate for LoRA training, in GB. Within ~15 % of measured runs at the
    reference size; treat as ±30 % for larger models.

    Three terms: resident 4-bit weights; a logits term proportional to tokens × vocab (this
    dominates for small models with big vocabularies); and per-layer activations for the
    layers that receive gradients, scaled by hidden size (≈ sqrt of the weight size).

    ``num_layers`` must already be a real layer count. mlx-lm's ``-1`` ("every layer") would
    otherwise subtract memory and turn a run that cannot fit into a confident "yes".
    """
    if num_layers < 1:
        raise ValueError(
            f"num_layers must be a real layer count, got {num_layers}. "
            "Resolve -1 ('all layers') against the model before estimating."
        )
    ktok = batch_size * max_seq_length / 1000
    logits = ktok * vocab_k * _LOGITS_GB_PER_KTOK_PER_KVOCAB
    hidden_scale = (weights_gb / _REF_WEIGHTS_GB) ** 0.5
    layers = ktok * num_layers * _LAYER_GB_PER_KTOK[grad_checkpoint] * hidden_scale
    return round(weights_gb + _OVERHEAD_GB + logits + layers, 1)


def fits(machine: Machine, gb: float) -> str:
    """'yes' | 'tight' | 'no' for a given peak-memory need. macOS itself wants ~3 GB."""
    budget = machine.ram_gb - 3.0
    if gb <= budget * 0.75:
        return "yes"
    if gb <= budget:
        return "tight"
    return "no"
