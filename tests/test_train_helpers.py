"""Regression tests for the 0.2 bug fixes, one test per bug."""

import pytest

from mlxtuner.config import DataConfig, RunConfig
from mlxtuner.data import Prepared, prepare
from mlxtuner.hardware import Machine
from mlxtuner.train import (
    DATA_SUBDIR,
    _check_batch_size,
    _prune_checkpoints,
    _saved_config,
    _write_readme,
    build_args,
)

M2 = Machine(chip="Apple M2", ram_gb=8.6, apple_silicon=True)


def _cfg(**train):
    return RunConfig.from_dict({"model": "m", "data": "d.jsonl", "train": train}).resolved(M2)


def test_saved_config_makes_local_data_paths_absolute(tmp_path, monkeypatch):
    """`mlxtuner eval <run dir>` re-reads this file; a relative path only worked from one cwd."""
    src = tmp_path / "d.jsonl"
    src.write_text('{"text": "a"}\n')
    monkeypatch.chdir(tmp_path)
    saved = _saved_config(RunConfig.from_dict({"model": "m", "data": "d.jsonl"}))
    assert saved.data.path == str(src.resolve())


def test_saved_config_leaves_hub_ids_alone():
    saved = _saved_config(RunConfig.from_dict({"model": "m", "data": "yahma/alpaca-cleaned"}))
    assert saved.data.path == "yahma/alpaca-cleaned"


def test_conversion_dir_never_deletes_files_it_did_not_write(tmp_path):
    """`--output .` used to rmtree the project's own ./data."""
    src = tmp_path / "d.jsonl"
    src.write_text('{"text": "a"}\n')
    work = tmp_path / "work"
    work.mkdir()
    (work / "precious.jsonl").write_text("keep me\n")
    with pytest.raises(Exception, match="Refusing to overwrite"):
        prepare(DataConfig(path=str(src)), work)
    assert (work / "precious.jsonl").exists()


def test_conversion_dir_reuse_is_fine(tmp_path):
    src = tmp_path / "d.jsonl"
    src.write_text("\n".join('{"text": "a"}' for _ in range(3)))
    for _ in range(2):
        p = prepare(DataConfig(path=str(src)), tmp_path / "work")
    assert p.n_train == 3


def _prepared(tmp_path, n_train, n_valid):
    d = tmp_path / DATA_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    (d / "valid.jsonl").write_text('{"text": "v"}\n' * n_valid)
    return Prepared(dir=d, kind="text", source_format="text", n_train=n_train, n_valid=n_valid)


def test_batch_larger_than_valid_split_skips_validation(tmp_path):
    """mlx-lm raised a bare ValueError here, but only after loading the model."""
    data = _prepared(tmp_path, n_train=19, n_valid=1)
    _check_batch_size(_cfg(batch_size=2), data)
    assert data.n_valid == 0
    assert not (data.dir / "valid.jsonl").exists()


def test_batch_larger_than_train_split_is_a_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="batch_size=4"):
        _check_batch_size(_cfg(batch_size=4), _prepared(tmp_path, n_train=2, n_valid=0))


def test_usable_valid_split_is_kept(tmp_path):
    data = _prepared(tmp_path, n_train=95, n_valid=5)
    _check_batch_size(_cfg(batch_size=2), data)
    assert data.n_valid == 5 and (data.dir / "valid.jsonl").exists()


def test_epoch_counts_the_batches_mlx_lm_actually_yields(tmp_path):
    """iterate_batches drops the short final batch, so an epoch is floor(n / batch)."""
    args = build_args(_cfg(batch_size=4, epochs=1), _prepared(tmp_path, 95, 0), 95)
    assert args.iters == 23  # not 24


def test_prune_checkpoints(tmp_path):
    for it in (3, 6, 9, 12):
        (tmp_path / f"{it:07d}_adapters.safetensors").write_text("x")
    (tmp_path / "adapters.safetensors").write_text("final")
    _prune_checkpoints(tmp_path, keep=1)
    left = sorted(p.name for p in tmp_path.iterdir())
    assert left == ["0000012_adapters.safetensors", "adapters.safetensors"]
    _prune_checkpoints(tmp_path, keep=0)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["adapters.safetensors"]


def test_prune_checkpoints_keep_all(tmp_path):
    (tmp_path / "0000003_adapters.safetensors").write_text("x")
    _prune_checkpoints(tmp_path, keep=-1)
    assert (tmp_path / "0000003_adapters.safetensors").exists()


def test_run_readme_table_survives_a_missing_loss(tmp_path):
    """A blank line inside a markdown table ends the table; conditional rows left one behind."""
    meta = {
        "mlxtuner_version": "0.2.0",
        "machine": {"chip": "Apple M2", "ram_gb": 8.6},
        "train_examples": 10,
        "valid_examples": 0,
        "iters": 10,
        "final_train_loss": None,
        "final_val_loss": 1.5,
        "peak_memory_gb": 2.1,
    }
    _write_readme(tmp_path, _cfg(), meta)
    table = (tmp_path / "README.md").read_text().split("## Use")[0].strip().splitlines()
    rows = table[table.index("|---|---|") :]
    assert all(r.startswith("|") for r in rows), rows
    assert "Final train loss" not in "\n".join(rows)
    assert "| Final val loss | 1.500 |" in rows
