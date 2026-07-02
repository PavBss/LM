import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None

try:
    import safetensors.torch
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


ROOT_DIR = Path(__file__).resolve().parent
SPECIAL_TOKENS = {
    "PAD": "<PAD>",
    "UNK": "<UNK>",
    "BOS": "<BOS>",
    "EOS": "<EOS>",
    "USER": "<USER>",
    "USER_END": "</USER>",
    "MODEL": "<MODEL>",
    "MODEL_END": "</MODEL>",
    "THINK": "<think>",
    "THINK_END": "</think>",
    "ENDTEXT": "<|endtext|>",
    "STOP": "<STOP>",
    "SYSTEM": "<SYSTEM>",
    "SYSTEM_END": "</SYSTEM>",
    "MEMORY": "<MEMORY>",
    "MEMORY_END": "</MEMORY>",
    "TOOL_CALL": "<TOOL_CALL>",
    "TOOL_CALL_END": "</TOOL_CALL>",
    "TOOL_RESULT": "<TOOL_RESULT>",
    "TOOL_RESULT_END": "</TOOL_RESULT>",
    "PLAN": "<PLAN>",
    "PLAN_END": "</PLAN>",
    "ASK": "<ASK>",
    "ASK_END": "</ASK>",
    "OPTIONS": "<OPTIONS>",
    "OPTIONS_END": "</OPTIONS>",
    "OPT": "<OPT>",
    "OPT_END": "</OPT>",
}


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    max_position_embeddings: int = 2048
    intermediate_size: int = 2048
    norm_eps: float = 1e-6
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = False


def project_path(path: Optional[str], default: Optional[str] = None) -> Optional[Path]:
    value = path if path is not None else default
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else ROOT_DIR / p


def rank0_print(*args, **kwargs) -> None:
    if get_rank() == 0:
        print(*args, **kwargs)


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_rank0() -> bool:
    return get_rank() == 0


def setup_distributed() -> Tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))

    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training needs CUDA in this script.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            local_rank = torch.cuda.current_device()
    return device, rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            pass
    major, _minor = torch.cuda.get_device_capability()
    return major >= 8


class LLMTokenizer:
    def __init__(self, tokenizer_path: Union[str, Path]):
        if Tokenizer is None:
            raise RuntimeError("Install tokenizers first: pip install tokenizers")
        tokenizer_path = Path(tokenizer_path)
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        for name, token in SPECIAL_TOKENS.items():
            setattr(self, f"{name.lower()}_id", self.tokenizer.token_to_id(token))

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids = self.tokenizer.encode(text).ids
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: Union[List[int], np.ndarray, torch.Tensor], skip_special_tokens: bool = False) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        elif isinstance(ids, np.ndarray):
            ids = ids.tolist()
        return self.tokenizer.decode([int(x) for x in ids], skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, device=device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len: int, device, dtype) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached or self.cos_cached.device != x.device:
            self._set_cos_sin_cache(seq_len, device=x.device, dtype=x.dtype)
        return self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.hidden_size % config.num_heads == 0
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.attn_dropout = config.attn_dropout
        self.resid_dropout = nn.Dropout(config.resid_dropout)

    def forward(
        self,
        x: torch.Tensor,
        rotary_emb: RotaryEmbedding,
        position_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, hidden = x.size()
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        max_pos = int(position_ids.max().item()) + 1
        cos, sin = rotary_emb(q, seq_len=max_pos)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        if past_key_value is not None:
            prev_k, prev_v = past_key_value
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)

        new_kv = (k, v) if use_cache else None
        is_causal = attn_mask is None and seq_len > 1
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.resid_dropout(self.out_proj(y)), new_kv


class SwiGLUMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.resid_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        rotary_emb: RotaryEmbedding,
        position_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = x
        attn_out, new_kv = self.attn(
            self.attn_norm(x),
            rotary_emb,
            position_ids,
            attn_mask=attn_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = residual + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, new_kv


def chunked_cross_entropy(
    logits_weight: torch.Tensor,
    hidden_states: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 512,
) -> Tuple[torch.Tensor, int, int]:
    _, _, hidden = hidden_states.shape
    flat_h = hidden_states.reshape(-1, hidden)
    flat_t = targets.reshape(-1)
    valid = flat_t != -100
    denom = valid.sum().clamp(min=1)
    total = torch.zeros((), device=hidden_states.device, dtype=torch.float32)
    correct = 0
    valid_count = int(valid.sum().item())
    for start in range(0, flat_h.size(0), chunk_size):
        end = start + chunk_size
        chunk_logits = flat_h[start:end].matmul(logits_weight.t())
        t_chunk = flat_t[start:end]
        total = total + F.cross_entropy(chunk_logits, t_chunk, ignore_index=-100, reduction="sum")
        
        preds = chunk_logits.argmax(dim=-1)
        valid_chunk = t_chunk != -100
        correct += int((preds[valid_chunk] == t_chunk[valid_chunk]).sum().item())
    return total / denom, correct, valid_count


class CustomTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(
            dim=config.hidden_size // config.num_heads,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("w3.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        loss_chunk_size: int = 512,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        bsz, seq_len = input_ids.size()
        if position_ids is None:
            if past_key_values is not None:
                past_len = past_key_values[0][0].size(2)
                position_ids = torch.arange(past_len, past_len + seq_len, dtype=torch.long, device=input_ids.device)
            else:
                position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(bsz, -1)

        attn_mask = None
        if attention_mask is not None:
            pad_mask = attention_mask.to(torch.bool).view(bsz, 1, 1, -1)
            causal_mask = torch.tril(torch.ones(seq_len, pad_mask.size(-1), device=input_ids.device, dtype=torch.bool))
            attn_mask = causal_mask.unsqueeze(0).unsqueeze(1) & pad_mask

        x = self.embed(input_ids)
        new_kvs = []
        for idx, block in enumerate(self.layers):
            if self.config.gradient_checkpointing and self.training and not use_cache:
                def custom_forward(hidden, pos_ids, mask):
                    return block(hidden, self.rotary_emb, pos_ids, mask, None, False)[0]

                x = checkpoint_utils.checkpoint(custom_forward, x, position_ids, attn_mask, use_reentrant=False)
                new_kvs.append(None)
            else:
                past_kv = past_key_values[idx] if past_key_values is not None else None
                x, new_kv = block(
                    x,
                    self.rotary_emb,
                    position_ids,
                    attn_mask=attn_mask,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                )
                new_kvs.append(new_kv)

        x = self.norm(x)
        if targets is not None:
            return chunked_cross_entropy(self.lm_head.weight, x, targets, chunk_size=loss_chunk_size), None
        return self.lm_head(x), new_kvs


class MemmapDataset(Dataset):
    def __init__(self, bin_path: Path, seq_len: int):
        if not bin_path.exists():
            raise FileNotFoundError(f"Missing pretrain dataset: {bin_path}")
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_len = int(seq_len)
        self.num_samples = max(0, (len(self.data) - 1) // self.seq_len)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = np.asarray(self.data[start:end], dtype=np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])


class RaggedSFTDataset(Dataset):
    def __init__(self, prefix: Path, seq_len: int):
        self.tokens_path = prefix.with_suffix(".tokens.bin")
        self.labels_path = prefix.with_suffix(".labels.bin")
        self.offsets_path = prefix.with_suffix(".offsets.npy")
        for path in (self.tokens_path, self.labels_path, self.offsets_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing SFT dataset file: {path}")
        self.offsets = np.load(self.offsets_path)
        total = int(self.offsets[-1])
        self.tokens = np.memmap(self.tokens_path, dtype=np.uint16, mode="r", shape=(total,))
        self.labels = np.memmap(self.labels_path, dtype=np.int32, mode="r", shape=(total,))
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return max(0, len(self.offsets) - 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = int(self.offsets[idx])
        end = int(self.offsets[idx + 1])
        x = np.asarray(self.tokens[start:end], dtype=np.int64)
        y = np.asarray(self.labels[start:end], dtype=np.int64)
        if len(x) > self.seq_len:
            x = x[-self.seq_len :]
            y = y[-self.seq_len :]
        return torch.from_numpy(x), torch.from_numpy(y)


def sft_collate(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]], pad_token_id: int):
    max_len = max(len(x) for x, _ in batch)
    inputs, labels, masks = [], [], []
    for x, y in batch:
        pad_len = max_len - len(x)
        inputs.append(torch.cat([x, torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        labels.append(torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)]))
        masks.append(torch.cat([torch.ones(len(x), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
    return torch.stack(inputs), torch.stack(labels), torch.stack(masks)


def profile_name(profile: str) -> str:
    return profile.replace("-", "_")


def build_dataloaders(args: argparse.Namespace, tokenizer: LLMTokenizer, world_size: int):
    data_dir = project_path(args.data_dir)
    num_workers = args.num_workers
    if num_workers < 0:
        cpu_count = os.cpu_count() or 1
        num_workers = max(1, min(4, cpu_count // max(1, world_size)))

    if args.mode in {"pretrain", "extend-context"}:
        train_dataset = MemmapDataset(data_dir / "pretrain_train.bin", args.seq_len)
        val_dataset = MemmapDataset(data_dir / "pretrain_val.bin", args.seq_len)
        collate_fn = None
    elif args.mode == "sft":
        prof = profile_name(args.profile)
        train_dataset = RaggedSFTDataset(data_dir / f"finetune_{prof}_train", args.seq_len)
        val_dataset = RaggedSFTDataset(data_dir / f"finetune_{prof}_val", args.seq_len)
        collate_fn = lambda batch: sft_collate(batch, tokenizer.pad_id)
    else:
        raise ValueError(f"Unsupported training mode: {args.mode}")

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    return train_loader, val_loader, train_sampler


def get_learning_rate(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return args.lr * step / max(1, args.warmup_steps)
    if step >= args.max_steps:
        return args.min_lr
    ratio = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return args.min_lr + coeff * (args.lr - args.min_lr)


def configure_optimizers(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    decay, no_decay = set(), set()
    whitelist = (nn.Linear, nn.Embedding)
    blacklist = (RMSNorm, nn.LayerNorm)
    for module_name, module in model.named_modules():
        for param_name, _param in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            if param_name.endswith("bias"):
                no_decay.add(full_name)
            elif param_name.endswith("weight") and isinstance(module, whitelist):
                decay.add(full_name)
            elif param_name.endswith("weight") and isinstance(module, blacklist):
                no_decay.add(full_name)

    params = {name: param for name, param in model.named_parameters()}
    missing = params.keys() - (decay | no_decay)
    if missing:
        raise RuntimeError(f"Uncategorized parameters: {sorted(missing)}")

    decay_names = [name for name in sorted(decay) if name in params]
    no_decay_names = [name for name in sorted(no_decay) if name in params]
    groups = [
        {"params": [params[name] for name in decay_names], "weight_decay": args.weight_decay},
        {"params": [params[name] for name in no_decay_names], "weight_decay": 0.0},
    ]
    use_fused = False
    optimizer_kwargs = {"lr": args.lr, "betas": (args.beta1, args.beta2)}
    if torch.cuda.is_available():
        try:
            optimizer = torch.optim.AdamW(groups, **optimizer_kwargs, fused=True)
            use_fused = True
        except (TypeError, RuntimeError) as exc:
            rank0_print(f"[Optimizer] AdamW fused unavailable: {exc}")
            optimizer = torch.optim.AdamW(groups, **optimizer_kwargs)
    else:
        optimizer = torch.optim.AdamW(groups, **optimizer_kwargs)
    rank0_print(f"[Optimizer] AdamW fused={use_fused}, decay={len(decay_names)}, no_decay={len(no_decay_names)}")
    return optimizer


def unwrap_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def source_hash() -> str:
    digest = hashlib.sha256()
    tracked_paths = [
        ROOT_DIR / "main.py",
        ROOT_DIR / "prepare.py",
        ROOT_DIR / "agent_runtime.py",
        ROOT_DIR / "download_data.py",
        ROOT_DIR / "setup_kaggle.py",
    ]
    tracked_paths.extend(sorted((ROOT_DIR / "configs").glob("*.json")))
    for path in tracked_paths:
        if not path.exists():
            continue
        digest.update(path.relative_to(ROOT_DIR).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_checkpoint(path: Path, device: torch.device) -> Tuple[Optional[ModelConfig], Dict[str, Any], Optional[Dict[str, Any]], int, float, str]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
        
    if path.suffix == ".safetensors":
        if not HAS_SAFETENSORS:
            raise ImportError("safetensors is required to load .safetensors files.")
        checkpoint = safetensors.torch.load_file(path, device=str(device))
    else:
        checkpoint = torch_load(path, map_location=device)
        
    if "model_state" in checkpoint:
        config = ModelConfig(**checkpoint["model_config"]) if "model_config" in checkpoint else None
        return (
            config,
            checkpoint["model_state"],
            checkpoint.get("optimizer_state"),
            int(checkpoint.get("step", 0)),
            float(checkpoint.get("best_loss", float("inf"))),
            str(checkpoint.get("train_mode", "pretrain")),
        )
    else:
        # Это "голый" файл весов, без конфигурации и оптимизатора (например, чистый .safetensors от HuggingFace)
        return (None, checkpoint, None, 0, float("inf"), "pretrain")


def checkpoint_names(args: argparse.Namespace) -> Tuple[str, str]:
    if args.mode == "extend-context":
        base = "base_4096"
    elif args.mode == "sft":
        base = profile_name(args.profile)
    else:
        base = "base"
    return f"{base}_best.pt", f"{base}_latest.pt"


def load_model_config_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Model config must be a JSON object: {path}")
    allowed = set(ModelConfig.__dataclass_fields__.keys())
    return {key: value for key, value in raw.items() if key in allowed}


def write_training_state(
    model: nn.Module,
    step: int,
    best_loss: float,
    args: argparse.Namespace,
    checkpoint_path: Path,
    reason: str,
) -> None:
    raw_model = unwrap_model(model)
    state = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "reason": reason,
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "best_loss": float(best_loss),
        "mode": args.mode,
        "profile": args.profile,
        "seq_len": int(args.seq_len),
        "precision": args.precision,
        "max_steps": int(args.max_steps),
        "model_config": asdict(raw_model.config),
        "source_hash": source_hash(),
    }
    (checkpoint_path.parent / "training_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def archive_latest_checkpoint(save_path: Path, step: int, keep_last_n: int) -> None:
    if keep_last_n <= 0 or "latest" not in save_path.stem:
        return
    archive_path = save_path.with_name(f"{save_path.stem}_step_{step}{save_path.suffix}")
    if archive_path == save_path:
        return
    shutil.copy2(save_path, archive_path)
    archived = sorted(
        save_path.parent.glob(f"{save_path.stem}_step_*{save_path.suffix}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for old_path in archived[keep_last_n:]:
        old_path.unlink(missing_ok=True)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_loss: float,
    args: argparse.Namespace,
    save_path: Path,
    save_safetensors: bool = False,
    reason: str = "manual",
) -> None:
    if not is_rank0():
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = unwrap_model(model)
    payload = {
        "step": step,
        "best_loss": best_loss,
        "train_mode": args.mode,
        "profile": args.profile,
        "model_config": asdict(raw_model.config),
        "model_state": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    torch.save(payload, save_path)
    rank0_print(f"[Checkpoint] Saved {save_path}")

    # Copy tokenizer to checkpoint directory
    tokenizer_path = project_path(getattr(args, "tokenizer", None)) or (project_path(args.data_dir) / "tokenizer.json")
    if tokenizer_path.exists():
        shutil.copy2(tokenizer_path, save_path.parent / "tokenizer.json")

    write_training_state(model, step, best_loss, args, save_path, reason)
    archive_latest_checkpoint(save_path, step, int(getattr(args, "keep_last_n_checkpoints", 0)))
    if save_safetensors and HAS_SAFETENSORS:
        st_path = save_path.with_suffix(".safetensors")
        safetensors.torch.save_file(raw_model.state_dict(), st_path)
        rank0_print(f"[Checkpoint] Saved {st_path}")


def make_model_config(
    args: argparse.Namespace,
    tokenizer: LLMTokenizer,
    checkpoint_config: Optional[ModelConfig] = None,
    config_path: Optional[Path] = None,
) -> ModelConfig:
    if checkpoint_config is None:
        config_values = load_model_config_json(config_path)
        config = ModelConfig(**config_values)
        if args.hidden_size is not None:
            config.hidden_size = args.hidden_size
        if args.num_layers is not None:
            config.num_layers = args.num_layers
        if args.num_heads is not None:
            config.num_heads = args.num_heads
        if args.intermediate_size is not None:
            config.intermediate_size = args.intermediate_size
        config.vocab_size = tokenizer.vocab_size
        config.max_position_embeddings = max(config.max_position_embeddings, args.seq_len)
        config.gradient_checkpointing = args.gradient_checkpointing or config.gradient_checkpointing
    else:
        config = checkpoint_config
        config.vocab_size = tokenizer.vocab_size
        config.max_position_embeddings = max(config.max_position_embeddings, getattr(args, "seq_len", 0))
        config.gradient_checkpointing = getattr(args, "gradient_checkpointing", config.gradient_checkpointing)
    if config.hidden_size % config.num_heads != 0:
        raise ValueError(f"hidden_size ({config.hidden_size}) must be divisible by num_heads ({config.num_heads}).")
    return config


def batch_to_device(batch, device: torch.device):
    if len(batch) == 3:
        input_ids, targets, attn_mask = batch
        return (
            input_ids.to(device, non_blocking=True),
            targets.to(device, non_blocking=True),
            attn_mask.to(device, non_blocking=True),
        )
    input_ids, targets = batch
    return input_ids.to(device, non_blocking=True), targets.to(device, non_blocking=True), None


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, args: argparse.Namespace, device: torch.device) -> Tuple[float, float]:
    was_training = model.training
    model.eval()
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    try:
        for idx, batch in enumerate(loader):
            if idx >= args.eval_steps:
                break
            input_ids, targets, attn_mask = batch_to_device(batch, device)
            with precision_context(args, device):
                (loss, correct, valid_t), _ = model(input_ids, targets=targets, attention_mask=attn_mask, loss_chunk_size=args.loss_chunk_size)
            totals[0] += loss.detach().to(dtype=torch.float64)
            totals[1] += 1.0
            totals[2] += correct
            totals[3] += valid_t

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)

        avg_loss = float((totals[0] / totals[1]).item()) if totals[1].item() > 0 else float("inf")
        avg_acc = float((totals[2] / totals[3]).item()) if totals[3].item() > 0 else 0.0
        return avg_loss, avg_acc
    finally:
        if was_training:
            model.train()
        else:
            model.eval()


def precision_context(args: argparse.Namespace, device: torch.device):
    if device.type != "cuda" or args.precision == "none":
        return nullcontext()
    if args.precision == "bf16":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.amp.autocast(device_type="cuda", dtype=torch.float16)


def default_checkpoint(args: argparse.Namespace) -> Optional[Path]:
    checkpoints = project_path(args.checkpoint_dir)
    candidates = [
        checkpoints / "assistant_best.pt",
        checkpoints / "base_4096_best.pt",
        checkpoints / "base_best.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def write_eval_sample(
    model: nn.Module,
    tokenizer: LLMTokenizer,
    args: argparse.Namespace,
    device: torch.device,
    logs_dir: Path,
    step: int,
) -> None:
    if not getattr(args, "sample_after_eval", False):
        return
    prompt = args.sample_prompt or "<BOS><USER>Коротко объясни, что такое Clyx.</USER><MODEL>"
    max_new_tokens = min(max(1, int(getattr(args, "sample_max_new_tokens", 128))), effective_generation_tokens(args))
    was_training = model.training
    try:
        chunks = list(
            generate_streaming(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                device=device,
                loss_chunk_size=args.loss_chunk_size,
                max_answer_tokens=getattr(args, "max_answer_tokens", None),
                max_think_tokens=getattr(args, "max_think_tokens", None),
            )
        )
        sample_path = logs_dir / "eval_samples.jsonl"
        record = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "step": int(step),
            "prompt": prompt,
            "completion": "".join(chunks).strip(),
        }
        with sample_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        rank0_print(f"[EvalSample] Wrote {sample_path}")
    except Exception as exc:
        rank0_print(f"[EvalSample] Skipped: {type(exc).__name__}: {exc}")
    finally:
        if was_training:
            model.train()
        else:
            model.eval()


def resize_embeddings_in_state_dict(state_dict: Dict[str, Any], new_vocab_size: int, hidden_size: int) -> bool:
    resized = False
    for key in ["embed.weight", "lm_head.weight"]:
        if key in state_dict:
            old_weight = state_dict[key]
            old_vocab_size = old_weight.size(0)
            if old_vocab_size < new_vocab_size:
                rank0_print(f"[Info] Resizing {key} in checkpoint from {old_vocab_size} to {new_vocab_size}")
                new_weight = torch.zeros((new_vocab_size, hidden_size), dtype=old_weight.dtype, device=old_weight.device)
                new_weight[:old_vocab_size, :] = old_weight
                mean_weight = old_weight.mean(dim=0, keepdim=True)
                new_weight[old_vocab_size:, :] = mean_weight
                state_dict[key] = new_weight
                resized = True
            elif old_vocab_size > new_vocab_size:
                rank0_print(f"[Warning] Checkpoint {key} has larger vocab size {old_vocab_size} than config {new_vocab_size}. Truncating to fit.")
                state_dict[key] = old_weight[:new_vocab_size, :]
                resized = True
    return resized


def train_pipeline(args: argparse.Namespace) -> None:
    device, rank, local_rank, world_size = setup_distributed()
    set_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    data_dir = project_path(args.data_dir)
    checkpoint_dir = project_path(args.checkpoint_dir)
    logs_dir = project_path(args.logs_dir)
    tokenizer_path = project_path(args.tokenizer) if args.tokenizer else data_dir / "tokenizer.json"
    checkpoint_path = project_path(args.checkpoint) if args.checkpoint else None
    model_config_path = project_path(args.model_config) if args.model_config else None
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = LLMTokenizer(tokenizer_path)
    checkpoint_config = None
    model_state = None
    optimizer_state = None
    step_start = 0
    best_loss = float("inf")
    checkpoint_mode = None
    if checkpoint_path:
        checkpoint_config, model_state, optimizer_state, step_start, best_loss, checkpoint_mode = load_checkpoint(checkpoint_path, device)
        rank0_print(f"[Checkpoint] Loaded {checkpoint_path}, step={step_start}, mode={checkpoint_mode}")

    config = make_model_config(args, tokenizer, checkpoint_config, model_config_path)
    model = CustomTransformer(config)
    if model_state is not None:
        if resize_embeddings_in_state_dict(model_state, config.vocab_size, config.hidden_size):
            rank0_print("[Info] Embeddings were resized. Clearing optimizer state to avoid shape mismatch.")
            optimizer_state = None
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        if missing or unexpected:
            rank0_print(f"[Checkpoint] Missing keys={len(missing)}, unexpected keys={len(unexpected)}")
    model.to(device)

    raw_params = sum(p.numel() for p in model.parameters())
    rank0_print("=" * 72)
    rank0_print(f"Clyx training: mode={args.mode}, profile={args.profile}, world_size={world_size}")
    if checkpoint_config is None and model_config_path and model_config_path.exists():
        rank0_print(f"Model config: {model_config_path}")
    elif checkpoint_config is not None:
        rank0_print("Model config: loaded from checkpoint")
    rank0_print(f"Device: {device}, precision={args.precision}, parameters={raw_params / 1e6:.2f}M")
    rank0_print(
        f"Architecture: hidden={config.hidden_size}, layers={config.num_layers}, "
        f"heads={config.num_heads}, mlp={config.intermediate_size}, ctx={config.max_position_embeddings}"
    )
    rank0_print(f"Data: {data_dir}")
    rank0_print("=" * 72)

    optimizer = configure_optimizers(model, args)
    if optimizer_state and (args.resume_optimizer or checkpoint_mode == args.mode):
        try:
            optimizer.load_state_dict(optimizer_state)
            rank0_print("[Optimizer] Restored optimizer state.")
        except Exception as exc:
            rank0_print(f"[Optimizer] Could not restore optimizer state: {exc}")

    if getattr(args, "compile", False) and hasattr(torch, "compile") and device.type == "cuda":
        rank0_print("[Compile] torch.compile enabled.")
        model = torch.compile(model)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    train_loader, val_loader, train_sampler = build_dataloaders(args, tokenizer, world_size)
    rank0_print(f"[Dataset] train_batches={len(train_loader)}, val_batches={len(val_loader)}")
    if len(train_loader) == 0:
        raise RuntimeError("Training dataset is too small for the selected seq_len/batch_size. Lower --seq_len or add more data.")

    use_scaler = device.type == "cuda" and args.precision == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    best_name, latest_name = checkpoint_names(args)
    csv_path = logs_dir / "training_log.csv"
    if is_rank0() and not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp", "mode", "profile", "step", "train_loss", "val_loss", "lr", "seconds_per_step"])

    step = step_start if args.resume_steps else 0
    epoch = 0
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    train_iter = iter(train_loader)
    model.train()
    t0 = time.time()
    last_time_save = time.time()
    last_loss = float("nan")
    train_correct_accum = 0
    train_valid_accum = 0

    try:
        while step < args.max_steps:
            lr = get_learning_rate(step, args)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0

            for micro_step in range(args.grad_accum):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    epoch += 1
                    if train_sampler is not None:
                        train_sampler.set_epoch(epoch)
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                input_ids, targets, attn_mask = batch_to_device(batch, device)
                sync_ctx = model.no_sync() if isinstance(model, DDP) and micro_step < args.grad_accum - 1 else nullcontext()
                with sync_ctx:
                    with precision_context(args, device):
                        (loss, correct, valid_t), _ = model(
                            input_ids,
                            targets=targets,
                            attention_mask=attn_mask,
                            loss_chunk_size=args.loss_chunk_size,
                        )
                        loss = loss / args.grad_accum
                    total_loss += float(loss.detach().item())
                    train_correct_accum += correct
                    train_valid_accum += valid_t
                    if use_scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            step += 1
            last_loss = total_loss * args.grad_accum

            if step % args.log_interval == 0 and is_rank0():
                elapsed = (time.time() - t0) / max(1, args.log_interval)
                t0 = time.time()
                train_acc = train_correct_accum / max(1, train_valid_accum)
                print(f"step {step:6d}/{args.max_steps:6d} | loss {last_loss:.4f} | acc {train_acc:.4f} | lr {lr:.2e} | {elapsed:.2f}s/step", flush=True)
                train_correct_accum = 0
                train_valid_accum = 0

            if step % args.eval_interval == 0:
                eval_model = unwrap_model(model) if isinstance(model, DDP) else model
                val_loss, val_acc = evaluate_loss(eval_model, val_loader, args, device)
                if is_rank0():
                    print(f"[Eval] step={step}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, train_loss={last_loss:.4f}")
                    with csv_path.open("a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([
                            dt.datetime.now().isoformat(timespec="seconds"),
                            args.mode,
                            args.profile,
                            step,
                            f"{last_loss:.6f}",
                            f"{val_loss:.6f}",
                            f"{lr:.8e}",
                            "",
                        ])
                    if val_loss < best_loss:
                        best_loss = val_loss
                        save_checkpoint(
                            model,
                            optimizer,
                            step,
                            best_loss,
                            args,
                            checkpoint_dir / best_name,
                            args.save_safetensors,
                            reason="best_eval",
                        )
                    eval_sample_model = unwrap_model(model) if isinstance(model, DDP) else model
                    write_eval_sample(eval_sample_model, tokenizer, args, device, logs_dir, step)
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                t0 = time.time()

            if step % args.save_interval == 0:
                save_checkpoint(model, optimizer, step, best_loss, args, checkpoint_dir / latest_name, False, reason="step_interval")
                last_time_save = time.time()
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

            time_save_minutes = float(getattr(args, "time_save_interval_minutes", 0.0))
            if time_save_minutes > 0 and (time.time() - last_time_save) >= time_save_minutes * 60.0:
                save_checkpoint(model, optimizer, step, best_loss, args, checkpoint_dir / latest_name, False, reason="time_interval")
                last_time_save = time.time()
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

    except KeyboardInterrupt:
        rank0_print("\n[Interrupted] Saving latest checkpoint...")
        try:
            if getattr(args, "save_on_keyboard_interrupt", True):
                save_checkpoint(model, optimizer, step, best_loss, args, checkpoint_dir / latest_name, False, reason="keyboard_interrupt")
        except Exception as e:
            rank0_print(f"Failed to save checkpoint during interrupt: {e}")
    except Exception as e:
        rank0_print(f"\n[Error] Training failed: {e}")
        raise
    finally:
        cleanup_distributed()


class StreamingDecoder:
    def __init__(self, tokenizer: LLMTokenizer):
        self.tokenizer = tokenizer
        self.tokens: List[int] = []
        self.printed_len = 0

    def put(self, token_id: int) -> str:
        self.tokens.append(int(token_id))
        text = self.tokenizer.decode(self.tokens)
        if text.endswith("\ufffd"):
            return ""
        new_text = text[self.printed_len :]
        self.printed_len = len(text)
        return new_text


KNOWN_STREAM_TAGS = (
    "<think>",
    "</think>",
    "<MODEL>",
    "</MODEL>",
    "<TOOL_CALL>",
    "</TOOL_CALL>",
    "<PLAN>",
    "</PLAN>",
    "<ASK>",
    "</ASK>",
    "<OPTIONS>",
    "</OPTIONS>",
    "<OPT>",
    "</OPT>",
    "<|endtext|>",
    "<STOP>",
    "<EOS>",
)


def remove_unstable_partial_tag(text: str) -> str:
    last_open = text.rfind("<")
    if last_open == -1 or ">" in text[last_open:]:
        return text
    tail = text[last_open:]
    if any(tag.startswith(tail) for tag in KNOWN_STREAM_TAGS):
        return text[:last_open]
    return text


def render_visible_generation(text: str, show_thinking: bool = False, show_tool_calls: bool = False) -> str:
    visible = text
    if not show_thinking:
        visible = re.sub(r"<think>.*?</think>", "", visible, flags=re.DOTALL)
        visible = re.sub(r"<think>.*$", "", visible, flags=re.DOTALL)
        visible = visible.replace("</think>", "")
    if not show_tool_calls:
        visible = re.sub(r"<TOOL_CALL>.*?</TOOL_CALL>", "", visible, flags=re.DOTALL)
        visible = re.sub(r"<TOOL_CALL>.*$", "", visible, flags=re.DOTALL)
        visible = visible.replace("</TOOL_CALL>", "")
    # Plan mode tags: show the content with readable headers, strip raw tags.
    visible = visible.replace("<PLAN>", "\n[Plan]\n").replace("</PLAN>", "")
    visible = visible.replace("<ASK>", "\n[Question] ").replace("</ASK>", "")
    # Render <OPTIONS><OPT>a</OPT><OPT>b</OPT></OPTIONS> as a numbered list.
    def _render_options(match: re.Match) -> str:
        opts = re.findall(r"<OPT>(.*?)</OPT>", match.group(1), flags=re.DOTALL)
        lines = [f"  {i}) {opt.strip()}" for i, opt in enumerate(opts, 1)]
        return "\n" + "\n".join(lines) if lines else ""

    visible = re.sub(r"<OPTIONS>(.*?)</OPTIONS>", _render_options, visible, flags=re.DOTALL)
    # Clean up any half-streamed options block.
    visible = re.sub(r"<OPTIONS>.*$", "", visible, flags=re.DOTALL)
    visible = visible.replace("<OPTIONS>", "").replace("</OPTIONS>", "")
    visible = visible.replace("<OPT>", "").replace("</OPT>", "")
    visible = visible.replace("<MODEL>", "").replace("</MODEL>", "")
    visible = visible.replace("<|endtext|>", "").replace("<STOP>", "").replace("<EOS>", "")
    return remove_unstable_partial_tag(visible)


class GenerationDisplay:
    def __init__(self, show_thinking: bool = False, show_tool_calls: bool = False):
        self.show_thinking = show_thinking
        self.show_tool_calls = show_tool_calls
        self.raw = ""
        self.printed_len = 0

    def put(self, chunk: str) -> str:
        self.raw += chunk
        visible = render_visible_generation(
            self.raw,
            show_thinking=self.show_thinking,
            show_tool_calls=self.show_tool_calls,
        )
        if len(visible) < self.printed_len:
            return ""
        delta = visible[self.printed_len :]
        self.printed_len = len(visible)
        return delta


def apply_repetition_penalty(logits: torch.Tensor, token_ids: Sequence[int], penalty: float) -> None:
    if penalty == 1.0 or not token_ids:
        return
    if penalty <= 0:
        raise ValueError("repetition_penalty must be greater than 0.")
    unique_ids = torch.as_tensor(sorted(set(int(token_id) for token_id in token_ids)), dtype=torch.long, device=logits.device)
    scores = logits[unique_ids]
    logits[unique_ids] = torch.where(scores < 0, scores * penalty, scores / penalty)


def effective_generation_tokens(args: argparse.Namespace) -> int:
    max_new = max(1, int(getattr(args, "max_new_tokens", 128)))
    max_answer = int(getattr(args, "max_answer_tokens", 0) or 0)
    max_think = int(getattr(args, "max_think_tokens", 0) or 0)
    max_plan = int(getattr(args, "max_plan_tokens", 0) or 0)
    max_ask = int(getattr(args, "max_ask_tokens", 0) or 0)
    if max_answer > 0 or max_think > 0 or max_plan > 0 or max_ask > 0:
        return min(max_new, max(1, max_answer + max_think + max_plan + max_ask + 32))
    return max_new


@torch.no_grad()
def generate_streaming(
    model: nn.Module,
    tokenizer: LLMTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    device: torch.device,
    loss_chunk_size: int = 512,
    max_answer_tokens: Optional[int] = None,
    max_think_tokens: Optional[int] = None,
) -> Iterable[str]:
    model.eval()
    raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ctx_len = unwrap_model(model).config.max_position_embeddings
    if len(raw_ids) > ctx_len - 1:
        raw_ids = raw_ids[-(ctx_len - 1) :]
    input_ids = torch.tensor([raw_ids], dtype=torch.long, device=device)
    generated: List[int] = []
    past_key_values = None
    decoder = StreamingDecoder(tokenizer)
    stop_ids = {tokenizer.eos_id, tokenizer.stop_id, tokenizer.endtext_id}
    tool_call_end_id = getattr(tokenizer, "tool_call_end_id", None)
    if tool_call_end_id is not None:
        stop_ids.add(tool_call_end_id)
    # Plan mode: stop after a complete plan or options block so the runtime can act on it.
    for stop_attr in ("plan_end_id", "options_end_id"):
        stop_token_id = getattr(tokenizer, stop_attr, None)
        if stop_token_id is not None:
            stop_ids.add(stop_token_id)
    think_id = getattr(tokenizer, "think_id", None)
    think_end_id = getattr(tokenizer, "think_end_id", None)
    in_think = False
    think_tokens = 0
    answer_tokens = 0
    answer_limit = int(max_answer_tokens or 0)
    think_limit = int(max_think_tokens or 0)

    def append_and_stream(token_id: int) -> str:
        generated.append(token_id)
        return decoder.put(token_id)

    for _ in range(max_new_tokens):
        if in_think and think_limit > 0 and think_tokens >= think_limit and think_end_id is not None:
            next_token = int(think_end_id)
            in_think = False
            piece = append_and_stream(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)
            if piece:
                yield piece
            continue

        model_input = input_ids[:, -1:] if past_key_values is not None else input_ids
        logits, past_key_values = model(model_input, past_key_values=past_key_values, use_cache=True, loss_chunk_size=loss_chunk_size)
        logits = logits[0, -1, :]
        apply_repetition_penalty(logits, raw_ids + generated, repetition_penalty)
        if temperature > 0:
            logits = logits / temperature
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[-1]] = -float("inf")
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(probs, dim=-1)
                remove = cumulative > top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                logits[sorted_indices[remove]] = -float("inf")
            next_token = int(torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).item())
        else:
            next_token = int(torch.argmax(logits, dim=-1).item())

        if next_token in stop_ids:
            break

        is_think_start = think_id is not None and next_token == int(think_id)
        is_think_end = think_end_id is not None and next_token == int(think_end_id)
        if is_think_start:
            in_think = True
        elif is_think_end:
            in_think = False
        elif in_think:
            if think_limit > 0 and think_tokens >= think_limit:
                if think_end_id is None:
                    break
                next_token = int(think_end_id)
                in_think = False
            else:
                think_tokens += 1
        elif answer_limit > 0:
            if answer_tokens >= answer_limit:
                break
            answer_tokens += 1

        piece = append_and_stream(next_token)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)
        if piece:
            yield piece


def compact_memory(tokenizer: LLMTokenizer, memory: str, old_turns: Sequence[Tuple[str, str]], max_tokens: int) -> str:
    parts = [memory] if memory else []
    for role, content in old_turns:
        label = "User" if role == "user" else ("Tool" if role == "tool" else "Assistant")
        compact = " ".join(content.split())
        parts.append(f"{label}: {compact}")
    text = "\n".join(part for part in parts if part)
    ids = tokenizer.encode(text)
    if len(ids) > max_tokens:
        ids = ids[-max_tokens:]
    return tokenizer.decode(ids)


def render_chat_prompt(tokenizer: LLMTokenizer, system_prompt: str, history: List[Tuple[str, str]], memory: str, ctx_len: int, max_new: int):
    def render_assistant_turn(content: str) -> str:
        if any(tag in content for tag in ("<think>", "<MODEL>", "<TOOL_CALL>", "<PLAN>", "<ASK>")):
            return content
        return f"<MODEL>{content}</MODEL>"

    def render() -> str:
        parts = ["<BOS>"]
        if system_prompt:
            parts.append(f"<SYSTEM>{system_prompt}</SYSTEM>")
        if memory:
            parts.append(f"<MEMORY>{memory}</MEMORY>")
        for role, content in history:
            if role == "user":
                parts.append(f"<USER>{content}</USER>")
            elif role == "tool":
                parts.append(f"<TOOL_RESULT>{content}</TOOL_RESULT>")
            else:
                parts.append(render_assistant_turn(content))
        return "".join(parts)

    prompt = render()
    reserve = max_new + min(128, max(16, ctx_len // 16))
    while len(tokenizer.encode(prompt)) + reserve > ctx_len and len(history) > 4:
        old = history[:2]
        del history[:2]
        memory = compact_memory(tokenizer, memory, old, max_tokens=500)
        prompt = render()
    while len(tokenizer.encode(prompt)) + reserve > ctx_len and memory:
        ids = tokenizer.encode(memory)
        memory = tokenizer.decode(ids[-max(64, len(ids) // 2) :])
        prompt = render()
    return prompt, memory


def choose_inference_checkpoint(args: argparse.Namespace) -> Path:
    explicit = project_path(args.checkpoint) if args.checkpoint else None
    if explicit:
        return explicit
    found = default_checkpoint(args)
    if found:
        return found
    raise FileNotFoundError("No checkpoint found. Train first or pass --checkpoint.")


def resolve_inference_device(args: argparse.Namespace, device: Optional[torch.device] = None) -> torch.device:
    if device is not None:
        return device
    requested = getattr(args, "device", "auto")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_for_inference(args: argparse.Namespace, device: Optional[torch.device] = None):
    device = resolve_inference_device(args, device)
    data_dir = project_path(args.data_dir)
    checkpoint_path = choose_inference_checkpoint(args)

    if args.tokenizer:
        tokenizer_path = project_path(args.tokenizer)
    elif (checkpoint_path.parent / "tokenizer.json").exists():
        tokenizer_path = checkpoint_path.parent / "tokenizer.json"
    else:
        tokenizer_path = data_dir / "tokenizer.json"

    tokenizer = LLMTokenizer(tokenizer_path)
    config, model_state, _optim, _step, _best, _mode = load_checkpoint(checkpoint_path, device)
    model_config_path = project_path(args.model_config) if getattr(args, "model_config", None) else None
    config = make_model_config(args, tokenizer, config, model_config_path)
    if model_state:
        resize_embeddings_in_state_dict(model_state, config.vocab_size, config.hidden_size)
    model = CustomTransformer(config)
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()
    if getattr(args, "compile", False) and hasattr(torch, "compile") and device.type == "cuda":
        print("[Compile] torch.compile enabled for inference.")
        model = torch.compile(model)
    return model, tokenizer, device, checkpoint_path


def chat_shell(args: argparse.Namespace) -> None:
    model, tokenizer, device, checkpoint_path = load_model_for_inference(args)
    system_prompt = args.system_prompt or "You are Clyx, a concise and useful assistant."
    history: List[Tuple[str, str]] = []
    memory = ""
    ctx_len = unwrap_model(model).config.max_position_embeddings
    print(f"[Chat] Loaded {checkpoint_path} on {device}. Type exit to stop.")
    while True:
        try:
            user_input = input("\nUser > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            break
        history.append(("user", user_input))
        max_new_tokens = effective_generation_tokens(args)
        prompt, memory = render_chat_prompt(tokenizer, system_prompt, history, memory, ctx_len, max_new_tokens)
        print("Clyx > ", end="", flush=True)
        answer_parts: List[str] = []
        display = GenerationDisplay(show_thinking=bool(getattr(args, "show_thinking", False)))
        for chunk in generate_streaming(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            device=device,
            loss_chunk_size=args.loss_chunk_size,
            max_answer_tokens=getattr(args, "max_answer_tokens", None),
            max_think_tokens=getattr(args, "max_think_tokens", None),
        ):
            answer_parts.append(chunk)
            visible = display.put(chunk)
            if visible:
                print(visible, end="", flush=True)
        print()
        history.append(("assistant", "".join(answer_parts).strip()))


def print_menu(args: argparse.Namespace) -> None:
    print("Clyx commands:")
    print("  python doctor.py")
    print("  python prepare.py")
    print("  python prepare.py --sft_only")
    print("  python main.py --mode pretrain --model_config configs/model_testagentic.json --seq_len 512 --batch_size 2 --grad_accum 8")
    print("  torchrun --standalone --nproc_per_node=2 main.py --mode pretrain --model_config configs/model_117m.json --seq_len 2048 --batch_size 2 --grad_accum 8")
    print("  torchrun --standalone --nproc_per_node=2 main.py --mode extend-context --checkpoint checkpoints/base_best.pt --seq_len 4096 --batch_size 1 --grad_accum 8")
    print("  torchrun --standalone --nproc_per_node=2 main.py --mode sft --profile assistant --checkpoint checkpoints/base_4096_best.pt --seq_len 4096 --batch_size 1 --grad_accum 8")
    print("  python main.py --mode sft --profile coding-agent --checkpoint checkpoints/base_best.pt --seq_len 512 --max_answer_tokens 128 --max_think_tokens 64")
    print("  python main.py --mode chat --checkpoint checkpoints/assistant_best.pt")
    print("  python export_bundle.py --checkpoint checkpoints/assistant_best.pt --tokenizer data/prepared/tokenizer.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clyx LLM trainer and runtime.")
    parser.add_argument("--mode", choices=["menu", "pretrain", "extend-context", "sft", "chat", "agent"], default="menu")
    parser.add_argument("--profile", choices=["assistant", "coding-agent"], default="assistant")
    parser.add_argument("--data_dir", type=str, default="data/prepared")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--logs_dir", type=str, default="logs")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model_config", type=str, default="configs/model_117m.json")

    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--time_save_interval_minutes", type=float, default=0.0)
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--loss_chunk_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=-1)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16", "none"], default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--resume_optimizer", action="store_true")
    parser.add_argument("--resume_steps", action="store_true")
    parser.add_argument("--save_safetensors", action="store_true")
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument("--save_on_keyboard_interrupt", action=argparse.BooleanOptionalAction, default=True)
    else:
        parser.add_argument("--save_on_keyboard_interrupt", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--intermediate_size", type=int, default=None)

    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_answer_tokens", type=int, default=128)
    parser.add_argument("--max_think_tokens", type=int, default=64)
    parser.add_argument("--max_plan_tokens", type=int, default=256)
    parser.add_argument("--max_ask_tokens", type=int, default=256)
    parser.add_argument("--show_thinking", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--sample_after_eval", action="store_true")
    parser.add_argument("--sample_prompt", type=str, default=None)
    parser.add_argument("--sample_max_new_tokens", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--agent_root", type=str, default=None)
    parser.add_argument("--dry_run_tools", action="store_true")
    parser.add_argument("--agent_tool_log", type=str, default="logs/agent_tools.jsonl")
    args = parser.parse_args()

    if args.precision == "auto":
        if torch.cuda.is_available():
            args.precision = "bf16" if cuda_supports_bf16() else "fp16"
        else:
            args.precision = "none"
    if args.repetition_penalty <= 0:
        parser.error("--repetition_penalty must be greater than 0")
    if args.time_save_interval_minutes < 0:
        parser.error("--time_save_interval_minutes must be >= 0")
    if args.keep_last_n_checkpoints < 0:
        parser.error("--keep_last_n_checkpoints must be >= 0")
    if args.sample_max_new_tokens < 1:
        parser.error("--sample_max_new_tokens must be >= 1")
    if args.max_new_tokens < 1:
        parser.error("--max_new_tokens must be >= 1")
    if args.max_answer_tokens < 0:
        parser.error("--max_answer_tokens must be >= 0")
    if args.max_think_tokens < 0:
        parser.error("--max_think_tokens must be >= 0")
    if args.max_plan_tokens < 0:
        parser.error("--max_plan_tokens must be >= 0")
    if args.max_ask_tokens < 0:
        parser.error("--max_ask_tokens must be >= 0")
    if args.mode == "extend-context" and args.seq_len < 4096:
        args.seq_len = 4096
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "menu":
        print_menu(args)
        return
    if args.mode in {"pretrain", "extend-context", "sft"}:
        train_pipeline(args)
        return
    if args.mode == "chat":
        chat_shell(args)
        return
    if args.mode == "agent":
        from agent_runtime import run_agent

        run_agent(args)
        return


if __name__ == "__main__":
    main()
