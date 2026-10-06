"""Chat with an adapter or fused model, and fuse adapters into standalone models."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


def is_adapter_dir(path: str | Path) -> bool:
    return (Path(path) / "adapters.safetensors").exists()


def base_model_of(adapter_dir: str | Path) -> str:
    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    model = cfg.get("model")
    if not model:
        raise RuntimeError(f"{adapter_dir}/adapter_config.json has no 'model' entry")
    return model


_TOKENIZER_FILES = ["*.json", "*.txt", "*.jinja", "tokenizer.model", "*.tiktoken"]
_MODEL_FILES = [*_TOKENIZER_FILES, "model*.safetensors", "*.py", "*.jsonl", "tiktoken.model"]


def resolve_model_path(model: str, weights: bool = True) -> Path:
    """Local directory for a model id or path, downloading (a subset of) the repo if needed."""
    p = Path(model).expanduser()
    if p.exists():
        return p
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(model, allow_patterns=_MODEL_FILES if weights else _TOKENIZER_FILES)
    )


def model_num_layers(model: str) -> int | None:
    """Transformer layer count from a model's config.json, or None if it can't be read.

    Only config.json and the tokenizer are fetched, never the weights, so this is cheap enough
    to call while printing a training plan.
    """
    try:
        cfg = json.loads((resolve_model_path(model, weights=False) / "config.json").read_text())
    except Exception:  # noqa: BLE001 - offline, private repo, unusual layout: just skip the hint
        return None
    for key in ("num_hidden_layers", "n_layers", "num_layers", "n_layer"):
        v = cfg.get(key)
        if isinstance(v, int) and v > 0:
            return v
    return None


def load_tokenizer_only(model: str) -> Any:
    """Fetch just the tokenizer files for a model id / path (no weights)."""
    from mlx_lm.utils import load_tokenizer

    return load_tokenizer(
        resolve_model_path(model, weights=False), tokenizer_config_extra={"trust_remote_code": True}
    )


def load_for_inference(path: str, base: str | None = None) -> tuple[Any, Any]:
    from mlx_lm.utils import load

    if is_adapter_dir(path):
        base = base or base_model_of(path)
        return load(base, adapter_path=path, tokenizer_config={"trust_remote_code": True})
    return load(path, tokenizer_config={"trust_remote_code": True})


def stream_reply(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Iterator[str]:
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    sampler = make_sampler(temp=temperature, top_p=top_p)
    for resp in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler):
        yield resp.text


def chat_loop(path: str, system: str | None, max_tokens: int, temperature: float) -> None:
    console.print(f"[dim]loading {path} ...[/]")
    model, tok = load_for_inference(path)
    console.print("[green]ready[/]. /reset clears history, /quit exits.\n")
    history: list[dict[str, str]] = [{"role": "system", "content": system}] if system else []
    while True:
        try:
            user = console.input("[bold cyan]you>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not user:
            continue
        if user in {"/quit", "/exit", "/q"}:
            break
        if user == "/reset":
            history = history[:1] if system else []
            continue
        history.append({"role": "user", "content": user})
        console.print("[bold magenta]bot>[/] ", end="")
        reply = ""
        for piece in stream_reply(
            model, tok, history, max_tokens=max_tokens, temperature=temperature
        ):
            console.print(piece, end="", markup=False, highlight=False)
            reply += piece
        console.print()
        history.append({"role": "assistant", "content": reply})


# ---------------------------------------------------------------------------
# fuse / export
# ---------------------------------------------------------------------------


def fuse(adapter_dir: str, output: str, dequantize: bool = False, gguf: str | None = None) -> Path:
    """Merge the adapter into its base model with `mlx_lm fuse`. Returns the output dir."""
    if not is_adapter_dir(adapter_dir):
        raise RuntimeError(f"{adapter_dir} has no adapters.safetensors")
    # Pass a local path, not a repo id: mlx_lm's save() asks huggingface_hub for a *complete*
    # local snapshot of a repo id, which load() never creates (it downloads a subset of files),
    # so recent huggingface_hub versions raise IncompleteSnapshotError.
    base = str(resolve_model_path(base_model_of(adapter_dir)))
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", base, "--adapter-path", adapter_dir, "--save-path", output,
    ]  # fmt: skip
    if dequantize:
        cmd.append("--dequantize")
    if gguf:
        cmd += ["--export-gguf", "--gguf-path", gguf]
    console.print("[dim]$ " + " ".join(cmd) + "[/]", soft_wrap=True)
    subprocess.run(cmd, check=True)
    console.print(f"[green]fused model saved to {output}[/]")
    return Path(output)


def find_llama_cpp_converter(explicit: str | None = None) -> tuple[Path, str]:
    """Locate llama.cpp's convert_hf_to_gguf.py and the interpreter to run it with.

    Prefers a ``.venv`` inside the llama.cpp checkout if one exists (some people install the
    converter's pinned requirements there); otherwise the current interpreter, which works as
    long as ``sentencepiece`` and ``protobuf<5`` are installed (``pip install 'mlxtuner[gguf]'``).
    """
    import os

    candidates: list[Path] = []
    if explicit:
        e = Path(explicit).expanduser()
        candidates += [e, e / "convert_hf_to_gguf.py"]
    if os.environ.get("LLAMA_CPP_DIR"):
        candidates.append(Path(os.environ["LLAMA_CPP_DIR"]) / "convert_hf_to_gguf.py")
    candidates += [
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
        Path.cwd() / "llama.cpp" / "convert_hf_to_gguf.py",
    ]
    for c in candidates:
        if c.is_file():
            venv_py = c.parent / ".venv" / "bin" / "python"
            return c, (str(venv_py) if venv_py.exists() else sys.executable)
    raise FileNotFoundError(
        "Could not find llama.cpp's convert_hf_to_gguf.py. Either:\n"
        "  git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp\n"
        "  pip install 'mlxtuner[gguf]'\n"
        "or pass --llama-cpp /path/to/llama.cpp, or set LLAMA_CPP_DIR."
    )


def to_gguf(
    model_dir: str, output: str | None = None, quant: str = "q8_0", llama_cpp: str | None = None
) -> Path:
    """Convert a *dequantized* fused model directory to GGUF with llama.cpp's converter.

    Works for every architecture llama.cpp knows (Qwen, Phi, Gemma, Llama, ...). The directory
    must come from ``mlxtuner fuse --dequantize``: fused 4-bit weights are in MLX's own
    quantised layout, which the converter cannot read.
    """
    converter, python = find_llama_cpp_converter(llama_cpp)
    src = Path(model_dir)
    cfg = json.loads((src / "config.json").read_text())
    if "quantization" in cfg or "quantization_config" in cfg:
        raise RuntimeError(
            f"{src} holds MLX-quantised weights. Re-run:  mlxtuner fuse <adapter> --output {src} --dequantize"
        )
    out = Path(output) if output else src / f"{src.name}-{quant}.gguf"
    cmd = [python, str(converter), str(src), "--outfile", str(out), "--outtype", quant]
    console.print("[dim]$ " + " ".join(cmd) + "[/]", soft_wrap=True)
    subprocess.run(cmd, check=True)
    console.print(f"[green]GGUF written to {out}[/]")
    return out


def write_ollama_modelfile(
    gguf_path: str, output: str | None = None, system: str | None = None
) -> Path:
    gguf = Path(gguf_path)
    out = Path(output) if output else gguf.parent / "Modelfile"
    lines = [f"FROM {gguf.name}", ""]
    if system:
        lines += [f'SYSTEM """{system}"""', ""]
    lines += [
        "PARAMETER temperature 0.7",
        "# Ollama reads the chat template from the GGUF for known model families.",
        "# If replies look wrong, add a TEMPLATE block matching your base model.",
    ]
    out.write_text("\n".join(lines) + "\n")
    name = gguf.stem.lower().replace("_", "-")
    hint = f"ollama create {name} -f {out} && ollama run {name}"
    if shutil.which("ollama"):
        console.print(f"[green]Modelfile written[/]. Next:  {hint}")
    else:
        console.print(
            f"[green]Modelfile written[/]. Install Ollama (https://ollama.com), then:  {hint}"
        )
    return out
