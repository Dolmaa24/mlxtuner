"""Load any common instruction-tuning format and write the train/valid.jsonl that mlx-lm expects.

mlx-lm reads a directory containing ``train.jsonl`` / ``valid.jsonl`` where each row is one of
``{"messages": [...]}``, ``{"prompt", "completion"}`` or ``{"text"}``. Everything else
(alpaca, ShareGPT, CSV, Hub datasets, list-of-parts content, odd column names) is converted here.
"""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import DataConfig

SUPPORTED_EXTENSIONS = {".jsonl", ".json", ".csv", ".txt"}


class DataError(ValueError):
    """User-facing dataset problem."""


_SHAREGPT_ROLES = {
    "human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
    "bot": "assistant", "model": "assistant", "system": "system", "tool": "tool", "function": "tool",
}  # fmt: skip
_VALID_ROLES = {"system", "user", "assistant", "tool"}
_INSTRUCTION_KEYS = ("instruction", "question", "prompt", "query")
_INPUT_KEYS = ("input", "context")
_OUTPUT_KEYS = ("output", "response", "answer", "completion")


# ---------------------------------------------------------------------------
# Loading rows
# ---------------------------------------------------------------------------


def _read_file(p: Path) -> list[dict[str, Any]]:
    ext = p.suffix.lower()
    if ext == ".jsonl":
        with open(p) as f:
            return [json.loads(line) for line in f if line.strip()]
    if ext == ".json":
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, dict):  # {"data": [...]} style wrappers
            for v in data.values():
                if isinstance(v, list):
                    return v
            raise DataError(f"{p}: JSON object has no list of rows")
        return data
    if ext == ".csv":
        with open(p, newline="") as f:
            return list(csv.DictReader(f))
    if ext == ".txt":
        with open(p) as f:
            return [{"text": line.rstrip("\n")} for line in f if line.strip()]
    raise DataError(f"Unsupported file type {ext!r}; use .jsonl, .json, .csv or .txt")


def load_rows(path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Return (train_rows, valid_rows_or_None) from a file, an mlx-lm style directory, or a Hub id."""
    p = Path(os.path.expanduser(path))
    if p.is_file():
        return _read_file(p), None
    if p.is_dir():
        train_f = p / "train.jsonl"
        if train_f.exists():
            valid_f = p / "valid.jsonl"
            return _read_file(train_f), (_read_file(valid_f) if valid_f.exists() else None)
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS)
        if not files:
            raise DataError(f"No data files in {p}")
        rows: list[dict[str, Any]] = []
        for f in files:
            rows.extend(_read_file(f))
        return rows, None
    if p.suffix.lower() in SUPPORTED_EXTENSIONS:
        # It names a data file, so it was meant to be local: don't send the user after a Hub extra.
        raise DataError(
            f"No such file: {p}\n"
            f"  (a data.path ending in {p.suffix} is treated as a local file, not a Hub dataset id)"
        )
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise DataError(
            f"{path!r} is not a local file or directory. To load a Hub dataset by id: "
            "pip install 'mlxtuner[hub]'"
        ) from e
    try:
        ds = load_dataset(path, split="train")
    except Exception as e:  # noqa: BLE001
        raise DataError(
            f"Could not load {path!r} as a file, directory or Hub dataset id: {e}"
        ) from e
    return [dict(r) for r in ds], None


# ---------------------------------------------------------------------------
# Detection + conversion (mirrors tunekit so datasets are portable between the two)
# ---------------------------------------------------------------------------


def detect_format(row: dict[str, Any], cfg: DataConfig | None = None) -> str:
    keys = set(row)
    if "messages" in keys and isinstance(row["messages"], list):
        return "messages"
    if "conversations" in keys and isinstance(row["conversations"], list):
        return "sharegpt"
    if {"chosen", "rejected"} <= keys:
        raise DataError(
            "This is a preference (DPO) dataset; mlxtuner does supervised fine-tuning only."
        )
    pf = cfg.prompt_field if cfg else "prompt"
    cf = cfg.completion_field if cfg else "completion"
    if pf in keys and cf in keys and isinstance(row[pf], str) and isinstance(row[cf], str):
        return "prompt_completion"
    if any(k in keys for k in _INSTRUCTION_KEYS) and any(k in keys for k in _OUTPUT_KEYS):
        return "alpaca"
    tf = cfg.text_field if cfg else "text"
    if tf in keys and isinstance(row[tf], str):
        return "text"
    raise DataError(
        f"Could not detect the dataset format from columns {sorted(keys)}.\nExpected one of:\n"
        "  messages:          {'messages': [{'role','content'}, ...]}\n"
        "  sharegpt:          {'conversations': [{'from','value'}, ...]}\n"
        "  alpaca:            {'instruction', 'input'?, 'output'}\n"
        "  prompt_completion: {'prompt', 'completion'}\n"
        "  text:              {'text'}\n"
        "Set data.format, or data.text_field / prompt_field / completion_field."
    )


def _first(d: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _flatten(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p if isinstance(p, str) else str(p.get("text", ""))
            for p in content
            if isinstance(p, str) or (isinstance(p, dict) and p.get("type", "text") == "text")
        )
    return "" if content is None else str(content)


def to_messages(row: dict[str, Any], fmt: str, system_prompt: str | None) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    if fmt == "messages":
        for m in row["messages"]:
            role = _SHAREGPT_ROLES.get(
                str(m.get("role", "")).lower(), str(m.get("role", "")).lower()
            )
            if role not in _VALID_ROLES:
                raise DataError(f"Invalid role {role!r}")
            msgs.append({"role": role, "content": _flatten(m.get("content"))})
    elif fmt == "sharegpt":
        system = _first(row, ("system", "system_prompt"))
        if system:
            msgs.append({"role": "system", "content": system})
        for t in row["conversations"]:
            raw = str(t.get("from", t.get("role", ""))).lower()
            role = _SHAREGPT_ROLES.get(raw)
            if role is None:
                raise DataError(f"Unknown sharegpt speaker {raw!r}")
            msgs.append({"role": role, "content": _flatten(t.get("value", t.get("content")))})
    elif fmt == "alpaca":
        instruction, output = _first(row, _INSTRUCTION_KEYS), _first(row, _OUTPUT_KEYS)
        if instruction is None or output is None:
            raise DataError("alpaca row is missing instruction/output")
        extra = _first(row, _INPUT_KEYS)
        system = _first(row, ("system", "system_prompt"))
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append(
            {"role": "user", "content": f"{instruction}\n\n{extra}" if extra else instruction}
        )
        msgs.append({"role": "assistant", "content": output})
    else:
        raise ValueError(fmt)

    if system_prompt and (not msgs or msgs[0]["role"] != "system"):
        msgs.insert(0, {"role": "system", "content": system_prompt})
    if not any(m["role"] == "assistant" for m in msgs):
        raise DataError("Conversation has no assistant turn")
    if msgs[-1]["role"] != "assistant":
        raise DataError("Conversation should end with an assistant turn")
    if any(m["role"] == "system" for m in msgs[1:]):
        raise DataError("System message must be the first turn")
    return msgs


def convert_row(row: dict[str, Any], fmt: str, cfg: DataConfig) -> dict[str, Any]:
    """One row in any supported format -> one mlx-lm row."""
    if fmt == "text":
        text = row.get(cfg.text_field)
        if not isinstance(text, str) or not text.strip():
            raise DataError("empty text")
        return {"text": text}
    if fmt == "prompt_completion":
        prompt, completion = row.get(cfg.prompt_field), row.get(cfg.completion_field)
        if not isinstance(completion, str) or not completion.strip():
            raise DataError("empty completion")
        if cfg.system_prompt:
            prompt = f"{cfg.system_prompt}\n\n{prompt}"
        return {"prompt": str(prompt), "completion": completion}
    return {"messages": to_messages(row, fmt, cfg.system_prompt)}


@dataclass
class Prepared:
    dir: Path  # directory containing train.jsonl / valid.jsonl for mlx-lm
    kind: str  # messages | prompt_completion | text
    source_format: str
    n_train: int
    n_valid: int
    dropped: int = 0
    drop_reasons: dict[str, int] = field(default_factory=dict)
    sample: dict[str, Any] | None = None


# The only files prepare() writes, and so the only ones it may delete on a re-run.
_OUR_FILES = {"train.jsonl", "valid.jsonl"}


def _clear_work_dir(work_dir: Path) -> None:
    """Empty the conversion directory, but never delete a directory we did not write."""
    if not work_dir.exists():
        return
    if not work_dir.is_dir():
        raise DataError(f"{work_dir} exists and is not a directory")
    stray = sorted(p.name for p in work_dir.iterdir() if p.name not in _OUR_FILES)
    if stray:
        raise DataError(
            f"Refusing to overwrite {work_dir}: it holds files mlxtuner did not write "
            f"({', '.join(stray[:4])}{', ...' if len(stray) > 4 else ''}). "
            "Point train.output at a directory of its own."
        )
    shutil.rmtree(work_dir)


def prepare(cfg: DataConfig, work_dir: Path) -> Prepared:
    """Load, convert, split and write mlx-lm's train/valid.jsonl into ``work_dir``."""
    train_rows, valid_rows = load_rows(cfg.path)
    if not train_rows:
        raise DataError("Dataset is empty")
    fmt = detect_format(train_rows[0], cfg) if cfg.format == "auto" else cfg.format
    kind = {"text": "text", "prompt_completion": "prompt_completion"}.get(fmt, "messages")

    reasons: Counter[str] = Counter()

    def convert_all(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for r in rows:
            try:
                out.append(convert_row(r, fmt, cfg))
            except DataError as e:
                reasons[str(e)] += 1
        return out

    if cfg.shuffle_seed is not None:
        random.Random(cfg.shuffle_seed).shuffle(train_rows)
    if cfg.max_samples:
        train_rows = train_rows[: cfg.max_samples]
    train = convert_all(train_rows)
    if not train:
        top = reasons.most_common(1)[0][0] if reasons else "unknown"
        raise DataError(f"Every row was rejected. Most common reason: {top}")

    if valid_rows is not None:
        valid = convert_all(valid_rows)
    elif cfg.eval_fraction > 0 and len(train) >= 10:
        n_valid = max(1, round(len(train) * cfg.eval_fraction))
        valid, train = train[:n_valid], train[n_valid:]
    else:
        valid = []

    _clear_work_dir(work_dir)
    work_dir.mkdir(parents=True)
    # mlx-lm treats a present-but-empty valid.jsonl as an error, so only write it when non-empty.
    for name, rows in (("train", train), ("valid", valid)):
        if not rows:
            continue
        with open(work_dir / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return Prepared(
        dir=work_dir,
        kind=kind,
        source_format=fmt,
        n_train=len(train),
        n_valid=len(valid),
        dropped=sum(reasons.values()),
        drop_reasons=dict(reasons),
        sample=train[0],
    )


def render(tokenizer: Any, row: dict[str, Any]) -> str:
    """The string the model actually trains on for one converted row."""
    if "messages" in row:
        return tokenizer.apply_chat_template(row["messages"], tokenize=False)
    if "prompt" in row:
        return row["prompt"] + row["completion"]
    return row["text"]


def token_stats(
    tokenizer: Any, rows: list[dict[str, Any]], max_seq_length: int, sample: int = 1000
) -> dict[str, Any]:
    rows = rows[:sample]
    lengths = sorted(len(tokenizer.encode(render(tokenizer, r))) for r in rows)
    n = len(lengths)

    def pct(p: float) -> int:
        return lengths[min(n - 1, int(p * n))]

    mean = sum(lengths) / n
    return {
        "sampled": n,
        "min": lengths[0],
        "p50": pct(0.5),
        "p90": pct(0.9),
        "max": lengths[-1],
        "mean": round(mean, 1),
        "over_max": sum(1 for x in lengths if x > max_seq_length),
    }
