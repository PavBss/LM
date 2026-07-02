import argparse
import json
from pathlib import Path

from main import ModelConfig, project_path, torch_load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a Clyx checkpoint and report common recovery problems.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="data/prepared/tokenizer.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = project_path(args.checkpoint)
    tokenizer_path = project_path(args.tokenizer)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    required = {"model_config", "model_state"}
    missing = sorted(required - set(checkpoint.keys()))
    report = {
        "checkpoint": str(checkpoint_path),
        "exists": True,
        "missing_required_keys": missing,
        "step": int(checkpoint.get("step", 0)),
        "best_loss": float(checkpoint.get("best_loss", float("inf"))),
        "train_mode": checkpoint.get("train_mode"),
        "profile": checkpoint.get("profile"),
        "tokenizer": str(tokenizer_path),
        "tokenizer_exists": tokenizer_path.exists(),
        "warnings": [],
    }

    if missing:
        report["warnings"].append("Checkpoint is missing required keys and cannot be loaded normally.")
    config_raw = checkpoint.get("model_config")
    if isinstance(config_raw, dict):
        try:
            config = ModelConfig(**config_raw)
            report["model_config"] = {
                "vocab_size": config.vocab_size,
                "hidden_size": config.hidden_size,
                "num_layers": config.num_layers,
                "num_heads": config.num_heads,
                "max_position_embeddings": config.max_position_embeddings,
            }
            if config.hidden_size % config.num_heads != 0:
                report["warnings"].append("hidden_size is not divisible by num_heads.")
        except Exception as exc:
            report["warnings"].append(f"Invalid model_config: {type(exc).__name__}: {exc}")
    else:
        report["warnings"].append("model_config is not a JSON object.")
    if not tokenizer_path.exists():
        report["warnings"].append("Tokenizer file is missing; inference and resume will fail.")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
