# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).

## [0.2.0] - 2026-10-07

Bug-fix release. Everything below was reproduced on an M2 8 GB before and after the fix, and each
one has a regression test.

### Fixed

- **`mlxtuner eval <run dir>` only worked from the directory training was started in.** The run's
  `mlxtuner.yaml` recorded `data.path` exactly as typed, so a relative path stopped resolving as
  soon as you moved. Local paths are now stored absolute, and if the dataset has since moved or been
  deleted, `eval` falls back to the converted split the run kept.
- **`train --output <dir>` deleted `<dir>/data/`.** The converted dataset was written to `data/`
  inside the run directory and that directory was cleared first, so `--output .` in a project with a
  `data/` folder destroyed it. Converted data now lives in `.mlxtuner-data/`, and the conversion step
  refuses to delete a directory holding anything it did not write.
- **A held-out split smaller than `batch_size` crashed mid-run** with mlx-lm's bare
  `ValueError: Dataset must have at least batch_size=2 examples but only has 1`, after the model had
  already loaded. This hit 32/64 GB Macs (batch 2 and 4) on any dataset under ~40 rows. Validation is
  now skipped with an explanation, and a training set smaller than the batch is reported before
  anything is downloaded.
- **`lora.num_layers: -1` ("all layers") made the memory estimate *subtract* memory.** A Qwen2.5-7B
  run that needs ~13.9 GB was reported as ~6.9 GB and "fits". The estimator now rejects the sentinel,
  and `train` resolves it against the model's `config.json` (weights are not downloaded for this),
  printing `layers=all(24)`.
- **The generated run README's table broke whenever a loss was missing** — conditional rows collapsed
  to an empty string, and a blank line inside a markdown table ends the table.
- **`--data` / `--set data.path=…` failed on configs using the `data: train.jsonl` shorthand**
  ("Cannot set 'data.path': 'data' is not a mapping"). The shorthand is expanded before overrides.
- **`--set` parsed values as YAML 1.1**, so `0123` became 83 (octal) and `on`/`off` became booleans.
  Parsing is now explicit, with quoting (`--set data.path="'0123'"`) to force a string.
- **An epoch was over-counted.** `iters` used `ceil(n / batch_size)`, but mlx-lm's `iterate_batches`
  drops the short final batch; with 95 rows at batch 4 that trained 24 steps per "epoch" instead of 23.
- The end-of-run summary crashed on `None` instead of omitting a loss it never received.
- `fuse` / `export` echo their llama.cpp and `mlx_lm` commands unwrapped, so they can be copy-pasted.

### Added

- `train.keep_checkpoints` (default 1): mlx-lm writes a numbered copy of the adapter at every save,
  and nothing cleaned them up — a 12-iteration smoke run left 28 MB of near-identical 5.9 MB files.
  `0` keeps none, `-1` keeps all. `adapters.safetensors` is never touched.
- `eval` now says when a result rests on too few examples to mean anything, matching tunekit.

## [0.1.0] - 2026-09-30

First release, published as `mlxtuner`.

Named `mlxtune` during development; PyPI rejects that as too close to the unrelated, pre-existing
[`mlx-tune`](https://pypi.org/project/mlx-tune/), so the project was renamed before its first release.
The GitHub repository moved from `Dolmaa24/mlxtune` to `Dolmaa24/mlxtuner` (the old URL redirects).

Pre-release polish (30 Sep 2026): verified from a fresh clone — clean install, full test suite, every documented
command, the built wheel outside its source tree, and the whole quickstart end to end. Fixed along the way: a missing
local data file reported a Hub error instead of "no such file"; the README config reference was missing a few real keys.

- `mlxtuner check / models / init / validate / train / eval / chat / fuse / export / info`.
- RAM-tier defaults (`auto` batch size, sequence length, LoRA layers, gradient checkpointing) resolved at train time.
- Peak-memory estimator calibrated on measured runs (M2, 8 GB); `mlxtuner models` shows which curated 4-bit models fit.
- Dataset auto-detection and conversion to mlx-lm's `train/valid.jsonl` layout.
- `eval`: mlx-lm's own validation loss (assistant tokens only with `mask_prompt`), sample generations, tuned vs base.
- `export`: GGUF for any llama.cpp-supported architecture via its converter (after `fuse --dequantize`), plus an Ollama Modelfile; verified end-to-end through `ollama run` with a Qwen2.5 adapter.
- Workaround for `mlx_lm fuse` failing with a Hub repo id on recent huggingface_hub (`IncompleteSnapshotError`).
- `mlx-lm` pinned `<1.0` (mlxtuner uses `train_model`, `CONFIG_DEFAULTS`, `load_dataset` from its internals).
- Example configs, unit + integration tests, macOS CI (push, PR, weekly against latest deps), Dependabot, pre-commit.
