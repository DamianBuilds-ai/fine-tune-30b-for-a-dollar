#!/usr/bin/env bash
# vastai_provision.sh - Automated Vast.ai instance search and setup
#
# This script wraps the Vast.ai CLI to find a suitable A100 and print
# the SSH command you need. It does NOT automatically run training -
# that requires manual supervision (HF download, monitoring, destroy).
#
# USAGE:
#   bash scripts/vastai_provision.sh
#
# REQUIRES:
#   - vastai CLI: pip install vastai
#   - vastai account: vast.ai (load $2+ credit)
#   - HF_TOKEN: export HF_TOKEN="your_huggingface_token"
#
# After running this script:
#   1. Copy the scp command to upload your training files
#   2. SSH into the instance
#   3. Install deps and run finetune.py
#   4. Download adapter
#   5. DESTROY the instance (this script does NOT destroy it)

set -euo pipefail

echo "========================================"
echo "Vast.ai A100 Provisioner"
echo "========================================"
echo ""

# Check vastai CLI is available
if ! command -v vastai &> /dev/null; then
    echo "ERROR: vastai CLI not found."
    echo "  Install: pip install vastai"
    echo "  Login:   vastai login"
    exit 1
fi

# Check credit balance
echo "[1/4] Checking Vast.ai credit balance..."
vastai show user 2>/dev/null | grep -E 'credit|balance' || echo "  (could not parse balance - check vast.ai dashboard)"
echo ""

# Search for A100 instances
echo "[2/4] Searching for A100 80GB instances..."
echo "  Filter: gpu_ram>=75, disk>=200GB, cuda>=13.0, verified, reliability>=0.95"
echo ""
vastai search offers \
    'gpu_ram>=75 num_gpus=1 disk_space>=200 cuda_vers>=13.0 verified=true reliability>=0.95' \
    -o 'dph asc' 2>/dev/null | head -20

echo ""
echo "========================================"
read -rp "Enter instance ID to rent (or Ctrl-C to cancel): " INSTANCE_ID
echo ""

# Create instance
echo "[3/4] Creating instance $INSTANCE_ID with 200GB disk..."
vastai create instance "$INSTANCE_ID" \
    --image vastai/pytorch:cuda-13.0.2-auto \
    --disk 200 \
    --ssh

echo ""
echo "[4/4] Waiting for instance to start..."
echo "  Polling every 10 seconds..."
for i in $(seq 1 30); do
    STATUS=$(vastai show instances 2>/dev/null | grep "$INSTANCE_ID" | awk '{print $5}' || echo "unknown")
    echo "  [$i/30] Status: $STATUS"
    if [[ "$STATUS" == "running" ]]; then
        break
    fi
    sleep 10
done

echo ""
echo "========================================"
echo "Instance ready. SSH details:"
vastai show instances 2>/dev/null | grep "$INSTANCE_ID"
echo ""
echo "Get the SSH command:"
echo "  vastai show instances"
echo ""
echo "Then upload files and train:"
echo "  scp -P <PORT> data/train.jsonl scripts/finetune.py root@ssh<N>.vast.ai:/workspace/"
echo "  ssh -p <PORT> root@ssh<N>.vast.ai"
echo "  /venv/main/bin/pip install transformers peft datasets accelerate bitsandbytes psutil"
echo "  export HF_TOKEN=\$HF_TOKEN"
echo "  /venv/main/bin/python finetune.py 2>&1 | tee training.log"
echo ""
echo "IMPORTANT: Destroy the instance after training to stop billing:"
echo "  vastai destroy instance $INSTANCE_ID"
echo "========================================"
