# Hardware: Why A100 and How to Rent One

## Why the 30B MoE Needs 80 GB VRAM

MoE (Mixture-of-Experts) is confusing. The model has 30B parameters but only ~3B are active
per forward pass during inference. That 3B active-param property is why CPU inference is viable.

**Training is different.** QLoRA fine-tuning requires loading ALL 30B parameters, not just
the active experts. In 4-bit quantization (NF4), 30B parameters consume roughly 15-18 GB.
But the optimizer state, gradient checkpoints, activation cache, and LoRA adapter weights
add another 30-40 GB. VRAM peak during our training runs: **59.8 GB**.

| GPU | VRAM | Result |
|---|---|---|
| RTX 3090 | 24 GB | OOM during model load |
| A6000 | 48 GB | OOM during training setup |
| A100 SXM4 | 80 GB | Works. 59.8 GB peak, 20 GB headroom |

The RTX 3090 and A6000 failures are not edge cases - they are guaranteed OOM for this model
at 4-bit with full target modules. Do not attempt without 80 GB VRAM.

**With Flash Attention 2 enabled:** VRAM peak drops by ~10-15 GB. Still need 65+ GB minimum.
The A100 remains the right choice for safety margin.

---

## Vast.ai Search Recipe

Search for the cheapest A100 with enough disk for the HuggingFace cache (~58 GB) plus
the training workspace and outputs (another 100+ GB).

```bash
# Install vastai CLI
pip install vastai

# Log in
vastai login

# Search - sort by price ascending
vastai search offers 'gpu_ram>=75 num_gpus=1 disk_space>=200 cuda_vers>=13.0 verified=true reliability>=0.95' -o 'dph asc'
```

**Filter explanation:**
- `gpu_ram>=75` - A100 80GB minimum (75 gives a small buffer for listing discrepancies)
- `num_gpus=1` - single GPU is enough and cheapest
- `disk_space>=200` - HF cache 58 GB + model 18 GB + workspace + buffer
- `cuda_vers>=13.0` - matches the PyTorch Docker image we use
- `verified=true` - Vast.ai has spot-checked these machines
- `reliability>=0.95` - avoids flaky hosts (crucial for training runs)

**Expected price range:** $0.75-1.50/hr for A100 80GB on Vast.ai.

Pick the cheapest instance with `reliability >= 0.95`. The machine labelled "Montana" in our
runs had 99.9% reliability - worth filtering for high reliability even at a small premium.

---

## Provision / SSH / Train / Download / Destroy Flow

**Critical rule:** The instance bills from creation, not from SSH. Every minute counts once
you create the instance. Steps 1-3 cost nothing. Steps 4-11 are on the clock.

### Step 1: Verify local readiness (no cost)

```bash
# Confirm training data exists
wc -l data/my_train.jsonl
# Expected: 500 lines for v2-equivalent training

# Confirm vastai CLI works
vastai show user
# Check credit balance - need at least $1.50
```

### Step 2: Search and pick an instance (no cost)

```bash
vastai search offers 'gpu_ram>=75 num_gpus=1 disk_space>=200 cuda_vers>=13.0 verified=true reliability>=0.95' -o 'dph asc'
# Note the ID of the cheapest option with reliability >= 0.95
```

### Step 3: Create instance and wait for "running" (billing starts here)

```bash
vastai create instance <ID> --image vastai/pytorch:cuda-13.0.2-auto --disk 200 --ssh

# Poll until status = "running" (usually 2-5 minutes)
vastai show instances
```

### Step 4: Test SSH before uploading anything

```bash
ssh -o ConnectTimeout=10 -i ~/.ssh/id_ed25519 -p <PORT> root@ssh<N>.vast.ai \
  "echo 'SSH OK' && nvidia-smi | head -5"
```

If SSH fails within 2 minutes of instance status "running": destroy and try a different instance.
Host-level SSH failures indicate a bad machine. Don't debug - move on.

### Step 5: Upload training data and script

```bash
scp -P <PORT> data/my_train.jsonl scripts/finetune.py root@ssh<N>.vast.ai:/workspace/
```

### Step 6: Install dependencies

```bash
ssh -p <PORT> root@ssh<N>.vast.ai \
  "/venv/main/bin/pip install transformers peft datasets accelerate bitsandbytes psutil pandas pyarrow 2>&1 | tail -5"

# Optional: Flash Attention 2 (start this first - takes 5-10 min to compile)
ssh -p <PORT> root@ssh<N>.vast.ai \
  "/venv/main/bin/pip install flash-attn --no-build-isolation 2>&1 | tail -3"
```

Install WITHOUT `--no-deps`. In v1 we used `--no-deps` and training failed due to missing
transitive dependencies. This is one of the 9 failure causes in `docs/failures.md`.

### Step 7: Train

```bash
ssh -p <PORT> root@ssh<N>.vast.ai \
  "cd /workspace && export HF_TOKEN='<your_hf_token>' && /venv/main/bin/python finetune.py 2>&1 | tee training.log"

# Monitor progress in another terminal
ssh -p <PORT> root@ssh<N>.vast.ai "tail -f /workspace/training-progress.jsonl"
```

HuggingFace will download Qwen3-30B-A3B-Instruct-2507 (~57 GB) to the instance cache.
This download takes 10-15 minutes the first time. Set HF_TOKEN or you'll hit rate limits.

Total wall time target: 20-35 minutes (5 min setup, 10-15 min HF download, 10-33 min training).

### Step 8: Download adapter immediately

```bash
# From your local machine or server
scp -P <PORT> -r root@ssh<N>.vast.ai:/workspace/my-adapter/ ./my-adapter/

# Verify it downloaded
ls -la ./my-adapter/
# Should contain: adapter_model.safetensors (~52 MB), adapter_config.json, tokenizer files
```

### Step 9: Destroy the instance (billing stops only when destroyed)

```bash
vastai destroy instance <ID>

# Verify it's gone
vastai show instances
```

**Do not forget this step.** An idle A100 at $1/hr costs $24/day.

---

## Local Inference Machine Requirements

The merged GGUF (~18 GB) runs on CPU. Requirements:

- RAM: 24+ GB (model stays in RAM during inference)
- CPU: Any modern x86_64 or ARM (AMD EPYC, Intel Xeon, Apple Silicon all work)
- Ollama installed

Our machine (AMD EPYC, 64 GB RAM) runs at 13-18 t/s.
Apple M4 Max 128 GB would run faster (~20-30 t/s estimated).

---

## Future: Local Fine-Tuning Options

| Option | Approx Cost (local currency) | Memory | Can Fine-Tune 30B |
|---|---|---|---|
| Mac Studio M4 Max 128 GB | $6,649 | 128 GB unified | Yes (via MLX) |
| PC + RTX 4090 | $2,700-4,700 | 24 GB VRAM | No - cloud still needed |
| DGX Spark | ~$7,200 | 128 GB unified | Yes |

Until local hardware is available, Vast.ai at $1/run is the right answer.
