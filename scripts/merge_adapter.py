#!/usr/bin/env python3
"""Merge LoRA adapter into base model (Python / safetensors path).

This is Path B from docs/deployment.md. Use this if you have 60+ GB RAM
and do not want to compile llama.cpp.

For CPU machines with limited RAM, use Path A (merge_adapter.sh) instead.
Path A uses llama-export-lora which streams the merge without loading the
full model into memory.

USAGE:
  Edit MODEL_ID, ADAPTER_DIR, and MERGED_DIR below, then:
  python3 scripts/merge_adapter.py

  After this completes, convert to GGUF:
  python3 llama.cpp/convert_hf_to_gguf.py ./merged-model/ --outfile my-merged.gguf --outtype q4_k_m
"""
import time, os, sys

print("[{}] Starting merge...".format(time.strftime("%H:%M:%S")))

# Use minimal memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# CONFIGURATION - edit before running
# ============================================================

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"   # HuggingFace model ID
ADAPTER_DIR = "./my-adapter"                       # path to your LoRA adapter
MERGED_DIR = "./merged-model"                      # output directory

# ============================================================

print("[{}] Loading base model in FP16 (CPU, low memory mode)...".format(time.strftime("%H:%M:%S")))
print("  This will use ~60 GB RAM + swap. May take 10-20 minutes on a CPU machine.", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
print("[{}] Base model loaded.".format(time.strftime("%H:%M:%S")), flush=True)

print("[{}] Loading LoRA adapter from {}...".format(time.strftime("%H:%M:%S"), ADAPTER_DIR))
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
print("[{}] Adapter loaded. Merging...".format(time.strftime("%H:%M:%S")), flush=True)

model = model.merge_and_unload()
print("[{}] Merged. Saving to {}...".format(time.strftime("%H:%M:%S"), MERGED_DIR), flush=True)

os.makedirs(MERGED_DIR, exist_ok=True)
model.save_pretrained(MERGED_DIR, safe_serialization=True)
AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True).save_pretrained(MERGED_DIR)

print("[{}] DONE. Merged model at {}".format(time.strftime("%H:%M:%S"), MERGED_DIR))
print("\nNext: convert to GGUF:")
print("  python3 llama.cpp/convert_hf_to_gguf.py {} --outfile my-merged.gguf --outtype q4_k_m".format(MERGED_DIR))
