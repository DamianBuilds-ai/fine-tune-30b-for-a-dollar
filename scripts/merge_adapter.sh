#!/usr/bin/env bash
# merge_adapter.sh - Path A: GGUF-to-GGUF adapter merge (recommended)
#
# This is the low-RAM path. llama-export-lora streams the merge without
# loading the full 18 GB model into memory at once.
#
# USAGE:
#   bash scripts/merge_adapter.sh <adapter_dir> <base_gguf> <output_gguf>
#
# EXAMPLE:
#   bash scripts/merge_adapter.sh ./my-adapter/ qwen3-30b-base.gguf my-merged.gguf
#
# REQUIRES:
#   - llama.cpp built locally (see below)
#   - HuggingFace base model directory (for step 1)
#
# BUILD llama.cpp:
#   git clone https://github.com/ggerganov/llama.cpp
#   cd llama.cpp
#   cmake -B build
#   cmake --build build --config Release -j $(nproc)
#
# ENVIRONMENT VARIABLES:
#   LLAMA_CPP_DIR  - path to llama.cpp repo (default: ./llama.cpp)
#   HF_BASE_DIR    - path to HuggingFace base model directory (for step 1)
#                    (default: ~/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/latest)

set -euo pipefail

ADAPTER_DIR="${1:?Usage: $0 <adapter_dir> <base_gguf> <output_gguf>}"
BASE_GGUF="${2:?Usage: $0 <adapter_dir> <base_gguf> <output_gguf>}"
OUTPUT_GGUF="${3:?Usage: $0 <adapter_dir> <base_gguf> <output_gguf>}"

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-./llama.cpp}"
HF_BASE_DIR="${HF_BASE_DIR:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/latest}"
ADAPTER_GGUF="${ADAPTER_DIR%.}/adapter.gguf"

echo "=============================="
echo "QLoRA Adapter Merge Pipeline"
echo "=============================="
echo "Adapter dir : $ADAPTER_DIR"
echo "Base GGUF   : $BASE_GGUF"
echo "Output GGUF : $OUTPUT_GGUF"
echo "llama.cpp   : $LLAMA_CPP_DIR"
echo ""

# Validate inputs
if [ ! -d "$ADAPTER_DIR" ]; then
    echo "ERROR: Adapter directory not found: $ADAPTER_DIR"
    exit 1
fi

if [ ! -f "$BASE_GGUF" ]; then
    echo "ERROR: Base GGUF not found: $BASE_GGUF"
    echo "  The base GGUF should be the Qwen3-30B quantized model (~18 GB)."
    echo "  Download via: ollama pull qwen3:30b-a3b (then locate in Ollama's blobs dir)"
    exit 1
fi

if [ ! -f "$LLAMA_CPP_DIR/build/bin/llama-export-lora" ]; then
    echo "ERROR: llama-export-lora not found at $LLAMA_CPP_DIR/build/bin/llama-export-lora"
    echo "  Build llama.cpp first:"
    echo "    git clone https://github.com/ggerganov/llama.cpp"
    echo "    cd llama.cpp && cmake -B build && cmake --build build --config Release -j \$(nproc)"
    exit 1
fi

# Step 1: Convert safetensors adapter to GGUF
echo "[$(date +%H:%M:%S)] Step 1: Converting safetensors adapter to GGUF..."
if [ ! -d "$HF_BASE_DIR" ]; then
    echo "ERROR: HuggingFace base model not found at $HF_BASE_DIR"
    echo "  Set HF_BASE_DIR to your cached HuggingFace model directory."
    echo "  Or use Path B (merge_adapter.py) which downloads automatically."
    exit 1
fi

python3 "$LLAMA_CPP_DIR/convert_lora_to_gguf.py" \
    --base "$HF_BASE_DIR" \
    "$ADAPTER_DIR" \
    --outfile "$ADAPTER_GGUF"

echo "[$(date +%H:%M:%S)] Adapter GGUF created: $ADAPTER_GGUF"

# Step 2: Merge adapter GGUF into base GGUF
echo "[$(date +%H:%M:%S)] Step 2: Merging adapter into base GGUF..."
echo "  This is streaming (low RAM). Progress may take several minutes."

"$LLAMA_CPP_DIR/build/bin/llama-export-lora" \
    --model "$BASE_GGUF" \
    --lora "$ADAPTER_GGUF" \
    --output "$OUTPUT_GGUF"

echo "[$(date +%H:%M:%S)] Merge complete: $OUTPUT_GGUF"
ls -lh "$OUTPUT_GGUF"

echo ""
echo "=============================="
echo "DONE. Next steps:"
echo "=============================="
echo "1. Edit Modelfile: set FROM to $(pwd)/$OUTPUT_GGUF"
echo "2. Create Ollama model:"
echo "   ollama create my-finetuned-model -f Modelfile"
echo "3. Test it:"
echo "   ollama run my-finetuned-model \"Hello\""
echo ""
echo "See docs/deployment.md for Modelfile setup and tool calling verification."
