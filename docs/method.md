# Full Pipeline: QLoRA Fine-Tune of a 30B MoE

## Overview

```
Qwen3-30B-A3B-Instruct-2507 (HuggingFace)
         |
         v
  QLoRA on Vast.ai A100 80GB
  (4-bit NF4 quantization, BF16 compute)
         |
         v
   LoRA adapter (~52 MB safetensors)
         |
         v
  llama.cpp: convert adapter to GGUF
  llama.cpp: merge adapter GGUF into base GGUF
         |
         v
  18 GB merged GGUF (Q4_K_M quantization)
         |
         v
  Ollama: serve with Modelfile
  (chat template + RENDERER/PARSER for tool calling)
         |
         v
  ~13 tokens/second on CPU (AMD EPYC, 64 GB RAM)
```

---

## Stage 1: Base Model

**Qwen3-30B-A3B-Instruct-2507** is a Mixture-of-Experts (MoE) model.

Key MoE property: it has 30B total parameters but only ~3B are active per token.
This is what makes CPU inference viable at 13 t/s - only a fraction of the network runs each step.

The instruct variant is used because it has the chat template baked in, which Ollama needs
to know how to format system/user/assistant turns and tool calls.

HuggingFace cache for the full model: ~57 GB. This sits on the training machine (Vast.ai).
The final GGUF (merged + quantized): ~18 GB. This is what you serve.

---

## Stage 2: QLoRA Training

**QLoRA** = Quantized Low-Rank Adaptation.

The base model is loaded in 4-bit NF4 quantization (reduces 60 GB down to ~15 GB VRAM).
The LoRA adapter adds a small set of trainable rank-decomposition matrices on top.
Only the adapter weights are updated - base model weights are frozen.

### LoRA configuration (v2 - what worked)

```yaml
r: 16                          # rank - controls adapter capacity
lora_alpha: 32                 # scaling factor (alpha/r = 2.0)
lora_dropout: 0.05
target_modules:
  - q_proj                     # query projection
  - k_proj                     # key projection
  - v_proj                     # value projection
  - o_proj                     # output projection
  - gate_proj                  # MLP gate
  - up_proj                    # MLP up
  - down_proj                  # MLP down
```

Targeting all 7 modules (not just q_proj + v_proj) was the single biggest change from v1 to v2.
The eval loss dropped from 0.509 to 0.076 - an 85% improvement.

### Training hyperparameters (v2)

| Parameter | Value | Reason |
|---|---|---|
| Quantization | 4-bit NF4 | Fits 30B into 80 GB VRAM |
| Compute dtype | BF16 | Better than FP16 for stability on A100 |
| Optimizer | paged_adamw_8bit | Low VRAM optimizer |
| Learning rate | 1e-4 | Stable for personality training |
| Batch size | 1 | Memory constraint |
| Gradient accumulation | 4 | Effective batch = 4 |
| Epochs | 5 | Better convergence than 3 |
| Warmup steps | 10 | Smooth LR ramp |
| Weight decay | 0.01 | Light regularization |
| Max grad norm | 0.3 | Gradient clipping |
| Max length | 1024 | Handles multi-turn conversations |
| Gradient checkpointing | True | Memory saving |

### Flash Attention 2

Flash Attention 2 (Tri Dao, Stanford 2023) tiles attention into SRAM blocks instead of
materializing the full N x N attention matrix in GPU HBM (high bandwidth memory).

Memory: O(N) instead of O(N^2).
Speed: ~2-3x faster attention layers.
Requirement: Ampere+ GPU (A100, RTX 3090/4090) + `flash-attn` package.

Enable with: `attn_implementation="flash_attention_2"` in `AutoModelForCausalLM.from_pretrained()`.
Install with: `pip install flash-attn --no-build-isolation` (5-10 min compile on Vast.ai).

The training script auto-detects whether flash-attn is installed and falls back gracefully.
In our runs, Flash Attention 2 was not used (compile time was too slow for the job budget).
Training still completed successfully using standard attention.

---

## Stage 3: Adapter Merge

After training, you have:
- The base model GGUF (already quantized, 18 GB)
- A LoRA adapter in safetensors format (~52 MB)

The adapter must be merged into the base weights before Ollama can serve it.

**Two paths:**

**Path A (recommended for CPU machines): llama.cpp GGUF merge**
```bash
# Convert safetensors adapter to GGUF adapter
python3 llama.cpp/convert_lora_to_gguf.py \
  --base /path/to/hf/base/model \
  my-adapter/ \
  --outfile my-adapter.gguf

# Merge GGUF adapter into base GGUF
llama.cpp/build/bin/llama-export-lora \
  --model qwen3-30b-base.gguf \
  --lora my-adapter.gguf \
  --output my-merged.gguf
```

`llama-export-lora` uses minimal RAM and does NOT load the full model into memory.
This is the correct path for low-RAM machines like a CPU inference server.

**Path B: Python safetensors merge (see scripts/merge_adapter.py)**
This loads the full model in FP16 (~60 GB RAM) and merges in Python.
Only use this if you have a machine with 64+ GB RAM.

See `scripts/merge_adapter.sh` for the full shell pipeline (Path A).

---

## Stage 4: Ollama Deployment

Ollama serves GGUF models. To serve a fine-tuned model from a raw GGUF:

1. You MUST provide a `Modelfile` with the full chat template.
   Without it, Ollama doesn't know how to format system/user/assistant turns
   and the model generates garbage (literal random tokens).

2. For Qwen3 MoE fine-tunes, you MUST add `RENDERER` and `PARSER` directives.
   Without them, tool calls come back as raw `<tool_call>` XML text in the `content`
   field instead of structured `tool_calls` objects. The OpenAI Agents SDK cannot
   execute raw text tool calls.

The Modelfile in this repo is a working template. Copy the TEMPLATE block from:
```bash
ollama show qwen3:30b-a3b --modelfile | grep -A 100 'TEMPLATE'
```

See `Modelfile` and `docs/deployment.md` for the complete setup.
