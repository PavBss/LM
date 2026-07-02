import argparse
import importlib.util
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple


ROOT_DIR = Path(__file__).resolve().parent
SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
    "<USER>",
    "</USER>",
    "<MODEL>",
    "</MODEL>",
    "<think>",
    "</think>",
    "<|endtext|>",
    "<STOP>",
    "<SYSTEM>",
    "</SYSTEM>",
    "<MEMORY>",
    "</MEMORY>",
    "<TOOL_CALL>",
    "</TOOL_CALL>",
    "<TOOL_RESULT>",
    "</TOOL_RESULT>",
    "<PLAN>",
    "</PLAN>",
    "<ASK>",
    "</ASK>",
    "<OPTIONS>",
    "</OPTIONS>",
    "<OPT>",
    "</OPT>",
]


def project_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT_DIR / p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Clyx environment, data and checkpoint readiness.")
    parser.add_argument("--data_dir", type=str, default="data/prepared")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model_config", type=str, default="configs/model_117m.json")
    parser.add_argument("--min_free_gb", type=float, default=5.0)
    return parser.parse_args()


def add(results: List[Tuple[str, str, str]], status: str, name: str, detail: str) -> None:
    results.append((status, name, detail))


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("JSON root must be an object")
    return raw


def check_tokenizer(tokenizer_path: Path, results: List[Tuple[str, str, str]]) -> int:
    if not tokenizer_path.exists():
        add(results, "WARN", "tokenizer", f"missing: {tokenizer_path}")
        return 0
    if not package_available("tokenizers"):
        add(results, "WARN", "tokenizer", "tokenizers package is not installed; cannot inspect tokenizer.json")
        return 0
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    missing = [token for token in SPECIAL_TOKENS if tokenizer.token_to_id(token) is None]
    if missing:
        add(results, "FAIL", "tokenizer", f"missing special tokens: {', '.join(missing)}")
    else:
        add(results, "OK", "tokenizer", f"vocab_size={vocab_size}")
    return int(vocab_size)


def check_torch(results: List[Tuple[str, str, str]]):
    if not package_available("torch"):
        add(results, "FAIL", "torch", "torch is not installed")
        return None
    import torch

    detail = f"version={torch.__version__}"
    add(results, "OK", "torch", detail)
    if torch.cuda.is_available():
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        bf16 = False
        checker = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(checker):
            bf16 = bool(checker())
        add(results, "OK", "cuda", f"devices={len(names)}, names={names}, bf16={bf16}")
    else:
        add(results, "WARN", "cuda", "CUDA is not available; training will run on CPU or fail for distributed GPU mode")
    return torch


def check_checkpoint(path: Path, tokenizer_vocab: int, torch_module, results: List[Tuple[str, str, str]]) -> None:
    if not path.exists():
        add(results, "FAIL", "checkpoint", f"missing: {path}")
        return
    if torch_module is None:
        add(results, "WARN", "checkpoint", "torch missing; cannot inspect checkpoint")
        return
    try:
        try:
            checkpoint = torch_module.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch_module.load(path, map_location="cpu")
        config = checkpoint.get("model_config", {})
        step = int(checkpoint.get("step", 0))
        vocab = int(config.get("vocab_size", 0)) if isinstance(config, dict) else 0
        if tokenizer_vocab and vocab and tokenizer_vocab != vocab:
            add(results, "FAIL", "checkpoint", f"vocab mismatch: checkpoint={vocab}, tokenizer={tokenizer_vocab}")
        else:
            add(results, "OK", "checkpoint", f"step={step}, vocab_size={vocab or 'unknown'}")
    except Exception as exc:
        add(results, "FAIL", "checkpoint", f"{type(exc).__name__}: {exc}")


def main() -> None:
    args = parse_args()
    results: List[Tuple[str, str, str]] = []
    add(results, "OK", "python", f"{sys.version.split()[0]} on {platform.platform()}")

    for pkg in ("numpy", "requests", "urllib3", "tokenizers", "safetensors", "pytest", "tqdm"):
        add(results, "OK" if package_available(pkg) else "WARN", f"package:{pkg}", "installed" if package_available(pkg) else "missing")

    torch_module = check_torch(results)
    raw_dir = project_path(args.raw_dir)
    data_dir = project_path(args.data_dir)
    config_path = project_path(args.model_config)
    tokenizer_path = project_path(args.tokenizer) if args.tokenizer else data_dir / "tokenizer.json"

    add(results, "OK" if raw_dir.exists() else "WARN", "raw_dir", str(raw_dir))
    add(results, "OK" if data_dir.exists() else "WARN", "data_dir", str(data_dir))
    for name in ("dataset.txt", "qa.txt", "tool_dataset.jsonl"):
        path = raw_dir / name
        detail = f"{path.stat().st_size} bytes" if path.exists() else "missing"
        add(results, "OK" if path.exists() else "WARN", f"raw:{name}", detail)

    prepared_files = ("pretrain_train.bin", "pretrain_val.bin")
    for name in prepared_files:
        path = data_dir / name
        detail = f"{path.stat().st_size} bytes" if path.exists() else "missing; run prepare.py"
        add(results, "OK" if path.exists() else "WARN", f"prepared:{name}", detail)

    tokenizer_vocab = check_tokenizer(tokenizer_path, results)
    try:
        config = load_json(config_path)
        hidden = int(config.get("hidden_size", 0))
        heads = int(config.get("num_heads", 0))
        if heads and hidden % heads == 0:
            add(results, "OK", "model_config", f"{config_path}, hidden={hidden}, heads={heads}")
        else:
            add(results, "FAIL", "model_config", f"hidden_size must be divisible by num_heads: {config_path}")
    except Exception as exc:
        add(results, "FAIL", "model_config", f"{type(exc).__name__}: {exc}")

    free_gb = shutil.disk_usage(ROOT_DIR).free / (1024 ** 3)
    add(results, "OK" if free_gb >= args.min_free_gb else "WARN", "disk_free", f"{free_gb:.2f} GB")

    if args.checkpoint:
        check_checkpoint(project_path(args.checkpoint), tokenizer_vocab, torch_module, results)

    width = max(len(name) for _status, name, _detail in results)
    for status, name, detail in results:
        print(f"[{status}] {name:<{width}} {detail}")
    if any(status == "FAIL" for status, _name, _detail in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
