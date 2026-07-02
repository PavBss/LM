import argparse
import datetime as dt
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Optional

from main import ROOT_DIR, project_path, torch_load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained Clyx checkpoint for local Windows/Linux inference.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data/prepared")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--no_zip", action="store_true")
    return parser.parse_args()


def default_tokenizer_path(args: argparse.Namespace) -> Path:
    if args.tokenizer:
        return project_path(args.tokenizer)
    return project_path(args.data_dir) / "tokenizer.json"


def read_checkpoint_metadata(checkpoint_path: Path) -> Dict:
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    return {
        "step": int(checkpoint.get("step", 0)),
        "best_loss": float(checkpoint.get("best_loss", float("inf"))),
        "train_mode": str(checkpoint.get("train_mode", "unknown")),
        "profile": str(checkpoint.get("profile", "unknown")),
        "model_config": checkpoint.get("model_config", {}),
        "created_at": checkpoint.get("created_at"),
    }


def write_readme(out_dir: Path) -> None:
    text = """# Clyx local bundle

This folder contains a Clyx checkpoint and the minimal runtime files needed for local inference.

## Install

```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

On Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Chat

```bash
python run_chat.py --checkpoint checkpoint.pt --tokenizer tokenizer.json --device auto
```

## Check the bundle

```bash
python doctor.py --checkpoint checkpoint.pt --tokenizer tokenizer.json
python rescue_checkpoint.py --checkpoint checkpoint.pt --tokenizer tokenizer.json
```

## Agent

```bash
python run_agent.py --checkpoint checkpoint.pt --tokenizer tokenizer.json --agent_root . --dry_run_tools
```

Remove `--dry_run_tools` only when you want tool calls to execute after confirmation.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def zip_directory(out_dir: Path) -> Path:
    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(out_dir.parent))
    return zip_path


def main() -> None:
    args = parse_args()
    checkpoint_path = project_path(args.checkpoint)
    tokenizer_path = default_tokenizer_path(args)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    metadata = read_checkpoint_metadata(checkpoint_path)
    if args.out:
        out_dir = project_path(args.out)
    else:
        profile = metadata.get("profile", checkpoint_path.stem)
        date = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT_DIR / "exports" / f"clyx_{profile}_{date}"
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(checkpoint_path, out_dir / "checkpoint.pt")
    shutil.copy2(tokenizer_path, out_dir / "tokenizer.json")
    for name in (
        "main.py",
        "agent_runtime.py",
        "run_chat.py",
        "run_agent.py",
        "doctor.py",
        "rescue_checkpoint.py",
        "requirements.txt",
    ):
        shutil.copy2(ROOT_DIR / name, out_dir / name)

    (out_dir / "model_config.json").write_text(
        json.dumps(metadata.get("model_config", {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "exported_at": dt.datetime.now().isoformat(timespec="seconds"),
        "checkpoint_source": str(checkpoint_path),
        "tokenizer_source": str(tokenizer_path),
        **metadata,
    }
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(out_dir)

    print(f"[Export] Wrote bundle: {out_dir}")
    if not args.no_zip:
        zip_path = zip_directory(out_dir)
        print(f"[Export] Wrote zip: {zip_path}")


if __name__ == "__main__":
    main()
