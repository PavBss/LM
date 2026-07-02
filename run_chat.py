import argparse
from pathlib import Path

from main import chat_shell, cuda_supports_bf16


def parse_args() -> argparse.Namespace:
    default_checkpoint = "checkpoint.pt" if Path("checkpoint.pt").exists() else None
    default_tokenizer = "tokenizer.json" if Path("tokenizer.json").exists() else None

    parser = argparse.ArgumentParser(description="Run Clyx chat locally on Windows/Linux.")
    parser.add_argument("--checkpoint", type=str, default=default_checkpoint)
    parser.add_argument("--tokenizer", type=str, default=default_tokenizer)
    parser.add_argument("--data_dir", type=str, default="data/prepared")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16", "none"], default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--loss_chunk_size", type=int, default=512)
    parser.add_argument("--max_answer_tokens", type=int, default=128)
    parser.add_argument("--max_think_tokens", type=int, default=64)
    parser.add_argument("--show_thinking", action="store_true")
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--mode", default="chat")
    parser.add_argument("--profile", default="assistant")
    args = parser.parse_args()

    if args.precision == "auto":
        import torch

        if torch.cuda.is_available():
            args.precision = "bf16" if cuda_supports_bf16() else "fp16"
        else:
            args.precision = "none"
    if args.repetition_penalty <= 0:
        parser.error("--repetition_penalty must be greater than 0")
    if args.max_new_tokens < 1:
        parser.error("--max_new_tokens must be >= 1")
    if args.max_answer_tokens < 0:
        parser.error("--max_answer_tokens must be >= 0")
    if args.max_think_tokens < 0:
        parser.error("--max_think_tokens must be >= 0")
    return args


def main() -> None:
    args = parse_args()
    chat_shell(args)


if __name__ == "__main__":
    main()
