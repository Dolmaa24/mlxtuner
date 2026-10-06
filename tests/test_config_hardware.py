from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mlxtuner.config import EXAMPLE_CONFIG, RunConfig, apply_overrides
from mlxtuner.hardware import MODELS, TIER_DEFAULTS, Machine, estimate_train_gb, fits


def test_example_config_valid():
    cfg = RunConfig.model_validate(yaml.safe_load(EXAMPLE_CONFIG))
    assert cfg.train.batch_size == "auto"


def test_data_as_string_shorthand():
    cfg = RunConfig.from_dict({"model": "m", "data": "d.jsonl"})
    assert cfg.data.path == "d.jsonl"


def test_overrides():
    cfg = RunConfig.from_dict(
        {"model": "m", "data": "d"}, ["train.lr=1e-4", "lora.rank=16", "train.batch_size=2"]
    )
    assert cfg.train.lr == 1e-4 and cfg.lora.rank == 16 and cfg.train.batch_size == 2
    with pytest.raises(ValueError, match="section.key=value"):
        apply_overrides({}, ["nope"])


def test_overrides_reach_a_data_shorthand_config():
    """`--data x` and `--set data.path=x` hit a string, not a mapping, before 0.2."""
    cfg = RunConfig.from_dict({"model": "m", "data": "old.jsonl"}, ["data.path=new.jsonl"])
    assert cfg.data.path == "new.jsonl"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("0123", 123),  # YAML 1.1 read this as octal 83
        ("yes", True),
        ("no", False),
        ("null", None),
        ("1e-4", 1e-4),
        ("-1", -1),
        ("runs/2024-05-05", "runs/2024-05-05"),
        ("a=b", "a=b"),
        ("'0123'", "0123"),  # quote to force a string
    ],
)
def test_override_scalars_are_predictable(value, expected):
    assert apply_overrides({}, [f"k={value}"])["k"] == expected


def test_estimator_rejects_the_all_layers_sentinel():
    """-1 meant 'every layer' to mlx-lm but *subtracted* memory here, turning 'no' into 'yes'."""
    with pytest.raises(ValueError, match="real layer count"):
        estimate_train_gb(4.28, 1, 2048, -1, True, 152)


def test_invalid_config_is_loud():
    with pytest.raises(ValidationError):
        RunConfig.from_dict({"model": "m", "data": "d", "lora": {"type": "qlora"}})


@pytest.mark.parametrize(
    "ram,tier",
    [(8.6, "8"), (16.0, "16"), (24.0, "32"), (36.0, "32"), (64.0, "64+"), (128.0, "64+")],
)
def test_tiers(ram, tier):
    assert Machine(chip="x", ram_gb=ram, apple_silicon=True).tier == tier


def test_resolved_fills_auto_from_tier():
    cfg = RunConfig.from_dict({"model": "m", "data": "d", "train": {"batch_size": 3}})
    r = cfg.resolved(Machine(chip="x", ram_gb=8.0, apple_silicon=True))
    d = TIER_DEFAULTS["8"]
    assert r.train.batch_size == 3  # explicit wins
    assert r.train.max_seq_length == d.max_seq_length
    assert r.lora.num_layers == d.num_layers
    assert r.train.grad_checkpoint == d.grad_checkpoint
    assert cfg.train.max_seq_length == "auto"  # original untouched


def test_estimator_tracks_measurements():
    # Measured on M2 8 GB, Qwen2.5-0.5B-4bit, ~1000 tokens per step.
    measured = [
        ((1, True, 8), 1.90),
        ((1, False, 8), 2.05),
        ((2, True, 8), 3.38),
        ((1, True, 24), 2.50),
    ]
    for (bs, gc, layers), peak in measured:
        est = estimate_train_gb(0.28, bs, 1000, layers, gc, vocab_k=152)
        assert peak <= est <= peak * 1.25, (bs, gc, layers, est, peak)
    # Qwen2.5-1.5B-4bit, batch 1, 8 layers, grad ckpt, longest example 645 tokens: 2.12 GB measured.
    est = estimate_train_gb(0.87, 1, 645, 8, True, vocab_k=152)
    assert 2.12 <= est <= 2.12 * 1.25, est


def test_estimator_monotonic():
    base = estimate_train_gb(4.3, 1, 1024, 16, True)
    assert estimate_train_gb(4.3, 2, 1024, 16, True) > base
    assert estimate_train_gb(4.3, 1, 2048, 16, True) > base
    assert estimate_train_gb(4.3, 1, 1024, 32, True) > base
    assert estimate_train_gb(4.3, 1, 1024, 16, False) > base


def test_tier_sweet_spots_fit():
    """Each tier's recommended model should be at least 'tight' with that tier's defaults."""
    by_repo = {m.repo: m for m in MODELS}
    for ram, repo in [
        (8, "mlx-community/Qwen2.5-1.5B-Instruct-4bit"),
        (16, "mlx-community/Qwen2.5-7B-Instruct-4bit"),
        (32, "mlx-community/Qwen2.5-14B-Instruct-4bit"),
    ]:
        m = Machine(chip="x", ram_gb=ram, apple_silicon=True)
        d = TIER_DEFAULTS[m.tier]
        rec = by_repo[repo]
        est = estimate_train_gb(
            rec.weights_gb,
            d.batch_size,
            d.max_seq_length,
            d.num_layers,
            d.grad_checkpoint,
            rec.vocab_k,
        )
        assert fits(m, est) in {"yes", "tight"}, (ram, repo, est)


def test_fits_thresholds():
    m = Machine(chip="x", ram_gb=16.0, apple_silicon=True)
    assert fits(m, 5.0) == "yes"
    assert fits(m, 12.0) == "tight"
    assert fits(m, 14.0) == "no"


def test_version_is_single_sourced():
    """pyproject must not carry its own version: hatch reads it from mlxtuner.__version__.

    Two literals drift — a release once shipped metadata saying 0.1.1 while the CLI said 0.1.0.
    """
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # 3.10 has no tomllib; dev extra provides tomli
        import tomli as tomllib

    raw = tomllib.loads((Path(__file__).resolve().parent.parent / "pyproject.toml").read_text())
    assert "version" in raw["project"].get("dynamic", []), (
        "pyproject should declare a dynamic version"
    )
    assert "version" not in raw["project"], (
        "pyproject has a static version; it would drift from __version__"
    )
    assert raw["tool"]["hatch"]["version"]["path"] == "src/mlxtuner/__init__.py"
