# Deployment: llama.cpp Merge, GGUF, and Ollama

## Overview

After training, you have a LoRA adapter in safetensors format.
To serve it with Ollama, you need to merge it into the base model as a GGUF.

Two merge paths exist. Choose based on your machine's RAM:

| Path | Method | RAM required | Tool |
|---|---|---|---|
| A (recommended) | GGUF-to-GGUF merge | ~4 GB RAM | llama.cpp llama-export-lora |
| B (fallback) | Python safetensors merge | 60+ GB RAM | scripts/merge_adapter.py |

---

## Path A: llama.cpp GGUF Merge (Recommended)

This is the correct method for CPU machines with limited RAM.
`llama-export-lora` does NOT load the full model into memory - it streams the merge.

### Build llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release -j $(nproc)
```

### Step 1: Convert safetensors adapter to GGUF

```bash
python3 llama.cpp/convert_lora_to_gguf.py \
  --base /path/to/hf/base/model \    # HuggingFace model directory (downloaded during training)
  ./my-adapter/ \                      # your adapter directory
  --outfile ./my-adapter.gguf
```

The HF base model directory is the 57 GB HuggingFace cache from the training step.
If you want to do this on your local machine without re-downloading, scp the HF cache:
```bash
scp -P <PORT> -r root@ssh<N>.vast.ai:~/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/ ~/.cache/huggingface/hub/
```
This is 57 GB. Only worth it if you plan multiple training iterations.

Alternative: Use the base GGUF you already have and skip Path A entirely - use Path B.

### Step 2: Merge GGUF adapter into base GGUF

```bash
llama.cpp/build/bin/llama-export-lora \
  --model qwen3-30b-base.gguf \       # 18 GB base GGUF
  --lora my-adapter.gguf \             # output from Step 1
  --output my-merged.gguf              # result: 18 GB merged GGUF
```

Output is another 18 GB GGUF. Adapter weights are baked in.

---

## Path B: Python Safetensors Merge

Use `scripts/merge_adapter.py`. Requires 60+ GB RAM (loads the full FP16 model).
Better for machines with large RAM but no need to compile llama.cpp.

```bash
python3 scripts/merge_adapter.py
```

Edit the path constants at the top of the script before running.
Output: safetensors merged model directory. Then convert to GGUF separately:

```bash
python3 llama.cpp/convert_hf_to_gguf.py \
  ./merged-model-directory/ \
  --outfile my-merged.gguf \
  --outtype q4_k_m
```

---

## The Modelfile: Critical Details

Without a correct Modelfile, Ollama will either:
- Generate garbage (no chat template)
- Return tool calls as text (no RENDERER/PARSER)

The `Modelfile` in this repo is a working template. Key sections:

### FROM directive
```
FROM /absolute/path/to/my-merged.gguf
```
Point to your merged GGUF file. Must be an absolute path.

### TEMPLATE block
The full Qwen3 chat template. Do NOT trim or modify this.
Get the authoritative version from the base Ollama model:
```bash
ollama show qwen3:30b-a3b --modelfile | grep -A 100 'TEMPLATE'
```

### RENDERER and PARSER (critical for tool calling)
```
RENDERER qwen3.5
PARSER qwen3.5
```
Use `qwen3.5` exactly. Confirmed working on Ollama 0.20.2+.
Without these, tool calls appear as raw XML in the `content` field.

### PARAMETER stop tokens
```
PARAMETER stop <|im_start|>
PARAMETER stop <|im_end|>
```
Qwen3 uses these as turn boundaries. Without them, the model may generate
across turn boundaries and hallucinate both sides of the conversation.

---

## Creating the Ollama Model

```bash
ollama create my-finetuned-model -f Modelfile
```

Verify the model loaded correctly:
```bash
ollama show my-finetuned-model --modelfile | grep -E 'RENDERER|PARSER|FROM'
```

---

## Testing Tool Calling

Before putting the model into production, verify tool calls work:

```bash
curl -s http://localhost:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "my-finetuned-model",
    "messages": [{"role": "user", "content": "log 4 minutes of meditation"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "log_mindfulness",
        "description": "Log a mindfulness session",
        "parameters": {
          "type": "object",
          "properties": {"duration_mins": {"type": "integer"}},
          "required": ["duration_mins"]
        }
      }
    }]
  }' | python3 -m json.tool | grep -A 10 '"tool_calls"'
```

The response MUST contain a `tool_calls` array. If `tool_calls` is empty or absent and
you see `<tool_call>` in `message.content`, RENDERER/PARSER are not working.

Fix for an already-created model (without recreating):
```bash
cat > /tmp/fix.modelfile << 'EOF'
FROM my-finetuned-model
RENDERER qwen3.5
PARSER qwen3.5
EOF
ollama create my-finetuned-model -f /tmp/fix.modelfile
```

---

## Benchmarking Inference Speed

```bash
# Simple throughput test
ollama run my-finetuned-model "Explain quantum entanglement in 200 words" --verbose

# Watch tokens/second in the output
# Expected on AMD EPYC 64 GB: 13-18 t/s
# Expected on Apple M4 Max 128 GB: ~20-30 t/s (estimated)
```

---

## Serving via OpenAI-Compatible API

Ollama exposes an OpenAI-compatible endpoint at port 11434:

```bash
# Chat completion
curl http://localhost:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"my-finetuned-model","messages":[{"role":"user","content":"Hello"}]}'
```

Any SDK that supports a custom base URL works:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
response = client.chat.completions.create(
    model="my-finetuned-model",
    messages=[{"role": "user", "content": "Hello"}]
)
```

---

## Cloudflare R2 Backup (Optional)

18 GB GGUFs are large to re-derive. Back up after each successful merge:

```bash
# Configure rclone with R2 credentials first
rclone copy my-merged.gguf r2:your-bucket/models/
rclone copy my-adapter/ r2:your-bucket/adapters/v1/
```

R2 has no egress fees. Storage costs ~$0.015/GB/month ($0.27/month for the 18 GB model).
