<p align="center">
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" width="200"/>
</p>

<h1 align="center">Clyx — Custom Language Model Training Pipeline</h1>

<p align="center">
  <strong>Pipeline for building a custom dataset and fine-tuning a language model from scratch.</strong>
</p>

<p align="center">
  <a href="https://huggingface.co/syntropic-clx"><img src="https://img.shields.io/badge/Org-syntropic--clx-yellow" alt="Organization"/></a>
  <a href="https://huggingface.co/syntropic-clx/Clyx_0.2-115.67M-BASE"><img src="https://img.shields.io/badge/Model-Clyx_0.2--115.67M--BASE-blue" alt="Model"/></a>
  <img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="License"/>
  <img src="https://img.shields.io/badge/GPU-RTX_6000_Pro_Blackwell-76B900" alt="GPU"/>
</p>

---

## What It Does

- **Scrapes and builds a custom dataset** from web sources
- **Trains a tokenizer** from scratch (ByteLevel BPE, 40k vocab)
- **Fine-tunes a language model** on the prepared dataset
- **Runs on GPU infrastructure** (tested on NVIDIA RTX 6000 Pro Blackwell, 96GB VRAM)

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| Model Framework | Hugging Face Transformers |
| Tokenizer | Hugging Face Tokenizers (ByteLevel BPE) |
| Model Weights | Safetensors |
| Optimization | AdamW (fused), Cosine LR with warmup |

## Status

**Early-stage / personal project.** Trained a small model (**Clyx 0.2 — 115.67M BASE**) as a proof of concept.

---

## Clyx 0.2 — 115.67M BASE

> 🤗 **[syntropic-clx/Clyx_0.2-115.67M-BASE](https://huggingface.co/syntropic-clx/Clyx_0.2-115.67M-BASE)** on Hugging Face

Causal language model trained **from scratch** on ~1.57B tokens. Custom Transformer with RoPE, RMSNorm, SwiGLU. No external pretrained weights.

### Architecture

| Parameter | Value |
|---|---|
| Parameters | 115.67M |
| Hidden size | 768 |
| Layers | 12 |
| Attention heads | 12 |
| MLP dim | 2048 |
| Positional encoding | RoPE |
| Normalization | RMSNorm |
| Activation | SwiGLU |
| Context window | 2048 tokens |
| Vocabulary | 40,000 (ByteLevel BPE) |
| Precision | bfloat16 |

### Training

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (fused) |
| LR schedule | Cosine with warmup |
| Learning rate | 3e-4 → 3e-5 |
| Batch size | 64 × 4 grad accum = 256 |
| Steps | 3,000 |
| Tokens | ~1.57B |
| Val loss | 1.4565 |
| Hardware | NVIDIA RTX PRO 6000 |
| Data | Russian text, English text, Python and C/C++ code (~10 GB raw) |

### Inference

Base model — continues text, does not answer questions or follow instructions.

```python
temperature        = 0.7
top_k              = 50
top_p              = 0.9
repetition_penalty = 1.15
max_new_tokens     = 1024
```

### How to Use

**Option 1 — via ClyxBox**
```bash
pip install clyxbox
```
```python
from clyxbox import ClyxModel, ClyxTokenizer

tokenizer = ClyxTokenizer.from_pretrained("syntropic-clx/Clyx_0.2-115.67M-BASE")
model     = ClyxModel.from_pretrained("syntropic-clx/Clyx_0.2-115.67M-BASE")
model.eval()

prompt = "Once upon a time in a dark forest"
ids    = tokenizer.encode(prompt, return_tensors="pt")
out    = model.generate(ids, max_new_tokens=200, temperature=0.7, top_p=0.9)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

**Option 2 — load weights manually**
```python
from safetensors.torch import load_file

state_dict = load_file("model.safetensors")
```

### Special Tokens

| Token | Role |
|---|---|
| `<BOS>` | Beginning of sequence |
| `<STOP>` | End of sequence |
| `<USER>` / `</USER>` | User turn |
| `<MODEL>` / `</MODEL>` | Model turn |
| `<SYSTEM>` / `</SYSTEM>` | System prompt |
| `<PAD>` | Padding |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Scrape and build dataset
python download_data.py

# Prepare tokenized data for training
python prepare.py

# Full training pipeline
python main.py --config configs/model_117m.json

# Interactive chat
python run_chat.py

# Agent mode
python run_agent.py

# Export model
python export_bundle.py
```

## Model Configuration

Edit files in `configs/` to adjust architecture. Example — `model_117m.json`:

```json
{
  "name": "clyx_117m",
  "vocab_size": 32000,
  "hidden_size": 768,
  "num_layers": 12,
  "num_heads": 12,
  "max_position_embeddings": 2048,
  "intermediate_size": 2048,
  "norm_eps": 1e-6,
  "tie_word_embeddings": true
}
```

## Links

- 🤗 **Organization:** [syntropic-clx](https://huggingface.co/syntropic-clx)
- 🤗 **Model:** [Clyx_0.2-115.67M-BASE](https://huggingface.co/syntropic-clx/Clyx_0.2-115.67M-BASE)

## License

This project is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) for details.
