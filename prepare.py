import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
ENDTEXT_TOKEN = "<|endtext|>"
STOP_TOKEN = "<STOP>"
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
    ENDTEXT_TOKEN,
    STOP_TOKEN,
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
    parser = argparse.ArgumentParser(description="Prepare Clyx tokenizer and datasets.")
    parser.add_argument("--raw_data", type=str, default="data/raw/dataset.txt")
    parser.add_argument("--qa_data", type=str, default="data/raw/qa.txt")
    parser.add_argument("--agent_data", type=str, default="data/raw/tool_dataset.jsonl")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--out_dir", type=str, default="data/prepared")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_think_tokens", type=int, default=64)
    parser.add_argument("--max_answer_tokens", type=int, default=128)
    parser.add_argument("--max_plan_tokens", type=int, default=256)
    parser.add_argument("--max_ask_tokens", type=int, default=256)
    parser.add_argument(
        "--tokenizer_mode",
        choices=["auto", "train", "reuse"],
        default="auto",
        help="auto reuses an existing tokenizer.json and trains one only when missing; train overwrites it; reuse requires it.",
    )
    parser.add_argument(
        "--sft_only",
        action="store_true",
        help="Only rebuild assistant/coding-agent SFT datasets and reuse the existing tokenizer.",
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u0400-\u04FF\u2013\u2014\u00AB\u00BB\u201C\u201D\u201E\u2026]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_end_markers(text: str) -> str:
    cleaned = clean_text(text)
    while True:
        before = cleaned
        for token in (STOP_TOKEN, ENDTEXT_TOKEN):
            if cleaned.endswith(token):
                cleaned = cleaned[: -len(token)].rstrip()
        if cleaned == before:
            return cleaned


def add_end_markers(text: str) -> str:
    cleaned = strip_end_markers(text)
    return f"{cleaned}{ENDTEXT_TOKEN}{STOP_TOKEN}" if cleaned else f"{ENDTEXT_TOKEN}{STOP_TOKEN}"


QAPair = Tuple[str, str, str]


def extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, re.DOTALL)
    return clean_text(match.group(1)) if match else clean_text(text)


def split_embedded_think(answer: str) -> Tuple[str, str]:
    match = re.search(r"<think>(.*?)</think>", answer, re.DOTALL)
    if not match:
        return "", strip_end_markers(answer)
    thought = clean_text(match.group(1))
    final = strip_end_markers(answer[: match.start()] + answer[match.end() :])
    return thought, final


def parse_qa_file(file_path: Path) -> List[QAPair]:
    qa_pairs: List[QAPair] = []
    if not file_path.exists():
        print(f"[Warning] QA file not found: {file_path}")
        return qa_pairs

    content = file_path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(
        r"<USER>(.*?)</USER>\s*\|\|\s*<think>(.*?)</think>\s*\|\|\s*<MODEL>(.*?)</MODEL>",
        re.DOTALL,
    )
    for question, thought, answer in pattern.findall(content):
        q = clean_text(question)
        t = clean_text(thought)
        a = strip_end_markers(answer)
        if q and a:
            qa_pairs.append((q, t, a))

    if not qa_pairs:
        old_pattern = re.compile(r"<USER>(.*?)</USER>\s*\|\|\s*<MODEL>(.*?)</MODEL>", re.DOTALL)
        for question, answer in old_pattern.findall(content):
            q = clean_text(question)
            t, a = split_embedded_think(answer)
            if q and a:
                qa_pairs.append((q, t, a))

    if not qa_pairs:
        for line in content.splitlines():
            parts = [part.strip() for part in line.split("||")]
            if len(parts) == 3:
                q = extract_tag(parts[0], "USER")
                t = extract_tag(parts[1], "think")
                a = extract_tag(parts[2], "MODEL")
            elif len(parts) == 2:
                q = extract_tag(parts[0], "USER")
                t, a = split_embedded_think(extract_tag(parts[1], "MODEL"))
            else:
                continue
            if q and a:
                qa_pairs.append((q, t, a))

    print(f"[Info] Parsed assistant QA pairs: {len(qa_pairs)}")
    return qa_pairs


def normalize_agent_record(record: Dict) -> str:
    if "text" in record:
        return add_end_markers(str(record["text"]))

    if "messages" in record and isinstance(record["messages"], list):
        parts = ["<BOS>"]
        for msg in record["messages"]:
            role = str(msg.get("role", "")).lower()
            content = str(msg.get("content", ""))
            if role == "system":
                parts.append(f"<SYSTEM>{content}</SYSTEM>")
            elif role == "user":
                parts.append(f"<USER>{content}</USER>")
            elif role == "think":
                parts.append(f"<think>{content}</think>")
            elif role in {"assistant", "model"}:
                parts.append(f"<MODEL>{content}</MODEL>")
            elif role == "tool_call":
                parts.append(f"<TOOL_CALL>{content}</TOOL_CALL>")
            elif role == "tool_result":
                parts.append(f"<TOOL_RESULT>{content}</TOOL_RESULT>")
            elif role == "plan":
                parts.append(f"<PLAN>{content}</PLAN>")
            elif role == "ask":
                parts.append(f"<ASK>{content}</ASK>")
            elif role == "options":
                opts = msg.get("options", content)
                opts_list = opts if isinstance(opts, list) else [opts]
                opts_xml = "".join(f"<OPT>{clean_text(str(o))}</OPT>" for o in opts_list if str(o).strip())
                if opts_xml:
                    parts.append(f"<OPTIONS>{opts_xml}</OPTIONS>")
        return add_end_markers("".join(parts))

    prompt = str(record.get("prompt", ""))
    if not prompt:
        return ""

    ask = clean_text(str(record.get("ask", "")))
    options = record.get("options", [])
    if isinstance(options, str):
        options = [options]
    plan = clean_text(str(record.get("plan", "")))
    user_answer = clean_text(str(record.get("user_answer", "")))
    plan_approval = record.get("plan_approval")

    tool_call = record.get("tool_call")
    tool_result = str(record.get("tool_result", ""))
    final = strip_end_markers(str(record.get("final", record.get("answer", ""))))
    think = clean_text(str(record.get("think", "")))
    final_think = clean_text(str(record.get("final_think", record.get("answer_think", ""))))

    parts = [f"<BOS><USER>{prompt}</USER>"]

    if ask:
        parts.append(f"<ASK>{ask}</ASK>")
        if options:
            opts_xml = "".join(f"<OPT>{clean_text(str(o))}</OPT>" for o in options if str(o).strip())
            if opts_xml:
                parts.append(f"<OPTIONS>{opts_xml}</OPTIONS>")
        if user_answer:
            parts.append(f"<USER>{user_answer}</USER>")

    if plan:
        parts.append(f"<PLAN>{plan}</PLAN>")
        if plan_approval:
            approval_text = clean_text(plan_approval) if isinstance(plan_approval, str) else "plan approved"
            parts.append(f"<TOOL_RESULT>{approval_text}</TOOL_RESULT>")

    if think:
        parts.append(f"<think>{think}</think>")
    if tool_call:
        parts.append(f"<MODEL><TOOL_CALL>{json.dumps(tool_call, ensure_ascii=False)}</TOOL_CALL></MODEL>")
    if tool_result:
        parts.append(f"<TOOL_RESULT>{tool_result}</TOOL_RESULT>")
    if final_think:
        parts.append(f"<think>{final_think}</think>")
    if final:
        parts.append(f"<MODEL>{final}</MODEL>")
    return add_end_markers("".join(parts))


def parse_agent_jsonl(file_path: Path) -> List[str]:
    samples: List[str] = []
    if not file_path.exists():
        print(f"[Info] Agent tool dataset not found, skipping: {file_path}")
        return samples

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[Warning] Bad JSONL line {line_no}: {exc}")
                continue
            text = normalize_agent_record(record)
            if text:
                samples.append(text)

    print(f"[Info] Parsed coding-agent tool samples: {len(samples)}")
    return samples


def load_tokenizer_lib():
    try:
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
    except ImportError:
        print("[Error] Install tokenizers first: pip install tokenizers")
        sys.exit(1)
    return Tokenizer, BPE, BpeTrainer, ByteLevel, ByteLevelDecoder


def write_tokenizer_corpus(raw_path: Path, qa_pairs: Sequence[QAPair], agent_samples: Sequence[str], out_dir: Path) -> Path:
    corpus_path = out_dir / "temp_tokenizer_corpus.txt"
    with corpus_path.open("w", encoding="utf-8", newline="\n") as out_f:
        if raw_path.exists():
            with raw_path.open("r", encoding="utf-8", errors="ignore") as raw_f:
                shutil.copyfileobj(raw_f, out_f, length=1024 * 1024)
                out_f.write("\n\n")
        for q, thought, a in qa_pairs:
            think_part = f"<think>{thought}</think>" if thought else ""
            out_f.write(f"<BOS><USER>{q}</USER>{think_part}<MODEL>{a}</MODEL>{ENDTEXT_TOKEN}{STOP_TOKEN}\n")
        for sample in agent_samples:
            out_f.write(sample)
            out_f.write("\n")
    return corpus_path


def train_tokenizer(raw_path: Path, qa_pairs: Sequence[QAPair], agent_samples: Sequence[str], vocab_size: int, out_dir: Path):
    Tokenizer, BPE, BpeTrainer, ByteLevel, ByteLevelDecoder = load_tokenizer_lib()
    print("[Info] Training ByteLevel BPE tokenizer...")

    corpus_path = write_tokenizer_corpus(raw_path, qa_pairs, agent_samples, out_dir)
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train(files=[str(corpus_path)], trainer=trainer)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer_path = out_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    corpus_path.unlink(missing_ok=True)
    print(f"[Success] Saved tokenizer: {tokenizer_path}")
    return tokenizer


def missing_special_tokens(tokenizer) -> List[str]:
    return [token for token in SPECIAL_TOKENS if tokenizer.token_to_id(token) is None]


def add_missing_special_tokens(tokenizer, tokenizer_path: Path) -> None:
    missing = missing_special_tokens(tokenizer)
    if missing:
        print(f"[Info] Tokenizer is missing special tokens: {', '.join(missing)}")
        print(f"[Info] Adding missing special tokens to tokenizer vocabulary...")
        tokenizer.add_special_tokens(missing)
        tokenizer.save(str(tokenizer_path))
        print(f"[Success] Updated tokenizer saved to {tokenizer_path}")


def load_existing_tokenizer(tokenizer_path: Path):
    Tokenizer, _BPE, _BpeTrainer, _ByteLevel, _ByteLevelDecoder = load_tokenizer_lib()
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found for reuse: {tokenizer_path}")
    print(f"[Info] Reusing tokenizer: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    add_missing_special_tokens(tokenizer, tokenizer_path)
    return tokenizer


def prepare_tokenizer(
    args: argparse.Namespace,
    raw_path: Path,
    qa_pairs: Sequence[QAPair],
    agent_samples: Sequence[str],
    out_dir: Path,
):
    tokenizer_path = out_dir / "tokenizer.json"
    tokenizer_mode = "reuse" if args.sft_only else args.tokenizer_mode
    if tokenizer_mode == "reuse" or (tokenizer_mode == "auto" and tokenizer_path.exists()):
        return load_existing_tokenizer(tokenizer_path), tokenizer_mode
    return train_tokenizer(raw_path, qa_pairs, agent_samples, args.vocab_size, out_dir), "train"


def process_pretrain_data(raw_path: Path, tokenizer, val_ratio: float, out_dir: Path):
    if not raw_path.exists():
        print(f"[Warning] Raw text not found, skipping pretrain data: {raw_path}")
        return None

    print(f"[Info] Reading raw text: {raw_path}")
    raw_text = raw_path.read_text(encoding="utf-8", errors="ignore")
    cleaned_text = clean_text(raw_text)
    print("[Info] Tokenizing raw text...")
    ids = np.asarray(tokenizer.encode(cleaned_text).ids, dtype=np.uint16)
    if ids.size < 2:
        print("[Warning] Raw text produced fewer than 2 tokens.")
        return None

    val_size = max(1, int(ids.size * val_ratio))
    train = ids[:-val_size]
    val = ids[-val_size:]
    train.tofile(out_dir / "pretrain_train.bin")
    val.tofile(out_dir / "pretrain_val.bin")
    print(f"[Success] Pretrain train tokens: {len(train):,}")
    print(f"[Success] Pretrain val tokens:   {len(val):,}")
    return len(train), len(val)


def encode_assistant_sample(tokenizer, question: str, thought: str, answer: str) -> Tuple[np.ndarray, np.ndarray]:
    prompt = f"<BOS><USER>{question}</USER>"
    think_part = f"<think>{thought}</think>" if thought else ""
    full = f"{prompt}{think_part}<MODEL>{answer}</MODEL>{ENDTEXT_TOKEN}{STOP_TOKEN}"
    prompt_ids = tokenizer.encode(prompt).ids
    full_ids = tokenizer.encode(full).ids
    input_ids = full_ids[:-1]
    targets = full_ids[1:]
    labels = [-100] * max(0, len(prompt_ids) - 1) + targets[max(0, len(prompt_ids) - 1):]
    return np.asarray(input_ids, dtype=np.uint16), np.asarray(labels, dtype=np.int32)


def encode_full_supervised_sample(tokenizer, sample_text: str) -> Tuple[np.ndarray, np.ndarray]:
    ids = tokenizer.encode(sample_text).ids
    return np.asarray(ids[:-1], dtype=np.uint16), np.asarray(ids[1:], dtype=np.int32)


def save_ragged_dataset(samples: Sequence[Tuple[np.ndarray, np.ndarray]], prefix: Path) -> int:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.zeros(len(samples) + 1, dtype=np.int64)
    total = 0
    for i, (input_ids, _labels) in enumerate(samples):
        total += len(input_ids)
        offsets[i + 1] = total

    tokens_path = prefix.with_suffix(".tokens.bin")
    labels_path = prefix.with_suffix(".labels.bin")
    offsets_path = prefix.with_suffix(".offsets.npy")

    tokens = np.memmap(tokens_path, dtype=np.uint16, mode="w+", shape=(total,))
    labels = np.memmap(labels_path, dtype=np.int32, mode="w+", shape=(total,))
    pos = 0
    for input_ids, sample_labels in samples:
        end = pos + len(input_ids)
        tokens[pos:end] = input_ids
        labels[pos:end] = sample_labels
        pos = end
    tokens.flush()
    labels.flush()
    np.save(offsets_path, offsets)
    return len(samples)


def split_samples(samples: Sequence[Tuple[np.ndarray, np.ndarray]], val_ratio: float, seed: int):
    if len(samples) == 1:
        return list(samples), list(samples)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(samples))
    val_count = max(1, int(len(samples) * val_ratio)) if len(samples) > 1 else 0
    val_idx = set(order[-val_count:].tolist()) if val_count else set()
    train, val = [], []
    for idx, sample in enumerate(samples):
        (val if idx in val_idx else train).append(sample)
    return train, val


def process_sft_data(
    qa_pairs: Sequence[QAPair],
    agent_samples: Sequence[str],
    tokenizer,
    val_ratio: float,
    out_dir: Path,
    seed: int,
):
    assistant_samples: List[Tuple[np.ndarray, np.ndarray]] = []
    for question, thought, answer in qa_pairs:
        input_ids, labels = encode_assistant_sample(tokenizer, question, thought, answer)
        if len(input_ids) > 1:
            assistant_samples.append((input_ids, labels))

    agent_encoded: List[Tuple[np.ndarray, np.ndarray]] = []
    for sample in agent_samples:
        input_ids, labels = encode_full_supervised_sample(tokenizer, sample)
        if len(input_ids) > 1:
            agent_encoded.append((input_ids, labels))

    stats: Dict[str, Tuple[int, int]] = {}
    for profile, samples in {"assistant": assistant_samples, "coding_agent": agent_encoded}.items():
        if not samples:
            print(f"[Info] No SFT samples for profile: {profile}")
            continue
        train, val = split_samples(samples, val_ratio, seed)
        train_count = save_ragged_dataset(train, out_dir / f"finetune_{profile}_train")
        val_count = save_ragged_dataset(val, out_dir / f"finetune_{profile}_val")
        stats[profile] = (train_count, val_count)
        avg_len = sum(len(x) for x, _ in samples) / len(samples)
        print(f"[Success] {profile}: train={train_count:,}, val={val_count:,}, avg_tokens={avg_len:.1f}")
    return stats


def file_stats(path: Path) -> Dict:
    if not path.exists():
        return {"exists": False, "bytes": 0, "lines": 0}
    lines = 0
    characters = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            lines += 1
            characters += len(line)
    return {
        "exists": True,
        "bytes": path.stat().st_size,
        "lines": lines,
        "characters": characters,
    }


def optional_json(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "path": str(path)}


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text).ids)


def length_report(args: argparse.Namespace, tokenizer, qa_pairs: Sequence[QAPair], agent_samples: Sequence[str]) -> Dict:
    max_think = int(getattr(args, "max_think_tokens", 64))
    max_answer = int(getattr(args, "max_answer_tokens", 128))
    max_plan = int(getattr(args, "max_plan_tokens", 256))
    max_ask = int(getattr(args, "max_ask_tokens", max_plan))
    report = {
        "limits": {
            "think_tokens": max_think,
            "answer_tokens": max_answer,
            "plan_tokens": max_plan,
            "ask_tokens": max_ask,
        },
        "qa": {"max_think_tokens": 0, "max_answer_tokens": 0, "over_limit": []},
        "agent": {"max_think_tokens": 0, "max_answer_tokens": 0, "max_plan_tokens": 0, "max_ask_tokens": 0, "over_limit": []},
    }

    for idx, (_question, thought, answer) in enumerate(qa_pairs, 1):
        think_tokens = token_count(tokenizer, thought)
        answer_tokens = token_count(tokenizer, answer)
        report["qa"]["max_think_tokens"] = max(report["qa"]["max_think_tokens"], think_tokens)
        report["qa"]["max_answer_tokens"] = max(report["qa"]["max_answer_tokens"], answer_tokens)
        if think_tokens > max_think or answer_tokens > max_answer:
            report["qa"]["over_limit"].append({
                "line": idx,
                "think_tokens": think_tokens,
                "answer_tokens": answer_tokens,
            })

    for idx, sample in enumerate(agent_samples, 1):
        think_chunks = re.findall(r"<think>(.*?)</think>", sample, flags=re.DOTALL)
        answer_chunks = [
            chunk
            for chunk in re.findall(r"<MODEL>(.*?)</MODEL>", sample, flags=re.DOTALL)
            if "<TOOL_CALL>" not in chunk
        ]
        plan_chunks = re.findall(r"<PLAN>(.*?)</PLAN>", sample, flags=re.DOTALL)
        ask_chunks = re.findall(r"<ASK>(.*?)</ASK>", sample, flags=re.DOTALL)
        sample_over = []
        for chunk_idx, chunk in enumerate(think_chunks, 1):
            count = token_count(tokenizer, chunk)
            report["agent"]["max_think_tokens"] = max(report["agent"]["max_think_tokens"], count)
            if count > max_think:
                sample_over.append({"field": f"think[{chunk_idx}]", "tokens": count})
        for chunk_idx, chunk in enumerate(plan_chunks, 1):
            count = token_count(tokenizer, chunk)
            report["agent"]["max_plan_tokens"] = max(report["agent"]["max_plan_tokens"], count)
            if count > max_plan:
                sample_over.append({"field": f"plan[{chunk_idx}]", "tokens": count})
        for chunk_idx, chunk in enumerate(ask_chunks, 1):
            count = token_count(tokenizer, chunk)
            report["agent"]["max_ask_tokens"] = max(report["agent"]["max_ask_tokens"], count)
            if count > max_ask:
                sample_over.append({"field": f"ask[{chunk_idx}]", "tokens": count})
        for chunk_idx, chunk in enumerate(answer_chunks, 1):
            count = token_count(tokenizer, chunk)
            report["agent"]["max_answer_tokens"] = max(report["agent"]["max_answer_tokens"], count)
            if count > max_answer:
                sample_over.append({"field": f"answer[{chunk_idx}]", "tokens": count})
        if sample_over:
            report["agent"]["over_limit"].append({"line": idx, "items": sample_over})

    return report


def write_metadata(
    out_dir: Path,
    args: argparse.Namespace,
    tokenizer,
    qa_pairs: Sequence[QAPair],
    agent_samples: Sequence[str],
    pretrain_stats,
    sft_stats: Dict[str, Tuple[int, int]],
) -> None:
    raw_path = project_path(args.raw_data)
    qa_path = project_path(args.qa_data)
    agent_path = project_path(args.agent_data)
    clean_report_path = raw_path.with_name("dataset_clean_report.json")
    warnings = []
    if not pretrain_stats:
        warnings.append("Pretrain dataset was not created.")
    if not qa_pairs:
        warnings.append("Assistant SFT dataset was not created because qa_data has no valid pairs.")
    if not agent_samples:
        warnings.append("Coding-agent SFT dataset was not created because agent_data has no valid samples.")
    for profile, (train_count, val_count) in sft_stats.items():
        if train_count == 0 or val_count == 0:
            warnings.append(f"SFT profile {profile} has an empty train or val split.")
    lengths = length_report(args, tokenizer, qa_pairs, agent_samples)
    if lengths["qa"]["over_limit"]:
        warnings.append(f"QA examples over token limits: {len(lengths['qa']['over_limit'])}")
    if lengths["agent"]["over_limit"]:
        warnings.append(f"Agent examples over token limits: {len(lengths['agent']['over_limit'])}")

    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": SPECIAL_TOKENS,
        "missing_special_tokens": missing_special_tokens(tokenizer),
        "tokenizer_mode": args.tokenizer_mode,
        "sft_only": bool(args.sft_only),
        "raw_data": str(raw_path),
        "qa_data": str(qa_path),
        "agent_data": str(agent_path),
        "raw_stats": file_stats(raw_path),
        "clean_report": optional_json(clean_report_path),
        "qa_stats": file_stats(qa_path),
        "agent_stats": file_stats(agent_path),
        "qa_pairs": len(qa_pairs),
        "agent_samples": len(agent_samples),
        "lengths": lengths,
        "pretrain": {
            "train_tokens": int(pretrain_stats[0]),
            "val_tokens": int(pretrain_stats[1]),
        } if pretrain_stats else None,
        "sft": {
            profile: {"train_samples": int(train), "val_samples": int(val)}
            for profile, (train, val) in sft_stats.items()
        },
        "warnings": warnings,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if warnings:
        print("[Warnings]")
        for warning in warnings:
            print(f"- {warning}")


def main() -> None:
    args = parse_args()
    out_dir = project_path(args.out_dir)
    raw_path = project_path(args.raw_data)
    qa_path = project_path(args.qa_data)
    agent_path = project_path(args.agent_data)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Clyx data preparation")
    print("=" * 72)
    print(f"Root:      {ROOT_DIR}")
    print(f"Raw text:  {raw_path}")
    print(f"QA:        {qa_path}")
    print(f"Agent:     {agent_path}")
    print(f"Output:    {out_dir}")

    qa_pairs = parse_qa_file(qa_path)
    agent_samples = parse_agent_jsonl(agent_path)
    tokenizer, args.tokenizer_mode = prepare_tokenizer(args, raw_path, qa_pairs, agent_samples, out_dir)
    pretrain_stats = None if args.sft_only else process_pretrain_data(raw_path, tokenizer, args.val_ratio, out_dir)
    sft_stats = process_sft_data(qa_pairs, agent_samples, tokenizer, args.val_ratio, out_dir, args.seed)
    write_metadata(out_dir, args, tokenizer, qa_pairs, agent_samples, pretrain_stats, sft_stats)

    print("\n" + "=" * 72)
    print("Dataset summary")
    print("=" * 72)
    print(f"Vocabulary: {tokenizer.get_vocab_size():,}")
    if pretrain_stats:
        print(f"Pretrain:   train={pretrain_stats[0]:,} tokens, val={pretrain_stats[1]:,} tokens")
    else:
        print("Pretrain:   not created")
    for profile, (train_count, val_count) in sft_stats.items():
        print(f"SFT {profile}: train={train_count:,}, val={val_count:,}")
    print("[Done] Prepared files are in data/prepared.")


if __name__ == "__main__":
    main()
