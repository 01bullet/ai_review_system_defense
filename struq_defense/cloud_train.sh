#!/bin/bash
# ============================================================
# StruQ Defense — Cloud Training Script (Generic)
# ============================================================
# Hardware: RTX 4090 24GB, 20 vCPU, 90GB RAM, 50GB SSD
#
# Usage:
#   1. 本机: python struq_defense/download_model.py
#   2. 打包: tar czf struq_cloud.tar.gz struq_defense/ data/ ai_scientist/ models/
#   3. 上传解压: tar xzf struq_cloud.tar.gz
#   4. 运行: bash struq_defense/cloud_train.sh
# ============================================================
set -e

echo "============================================================"
echo "  StruQ Defense — Cloud Training Setup"
echo "============================================================"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'checking...')"
echo "  VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'checking...')"
echo "  Python: $(python --version)"
echo ""

# ---- Local model ----
LOCAL_MODEL="models/qwen2.5-7b"
export STRUQ_LOCAL_MODEL="$LOCAL_MODEL"

if [ ! -d "$LOCAL_MODEL" ] || [ ! -f "$LOCAL_MODEL/config.json" ]; then
    echo "ERROR: Local model not found at $LOCAL_MODEL"
    echo "Run locally first: python struq_defense/download_model.py"
    exit 1
fi
echo "Model: $LOCAL_MODEL ($(du -sh $LOCAL_MODEL | cut -f1))"

# ---- Step 1: Install dependencies ----
echo "[1/4] Installing Python dependencies..."
pip install transformers peft accelerate bitsandbytes datasets tensorboard --quiet
echo "  Done."

# ---- Step 2: Verify GPU ----
echo "[2/4] Verifying environment..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'  GPU: {props.name}')
    print(f'  VRAM: {props.total_memory / 1024**3:.1f} GB')
import bitsandbytes as bnb
print(f'  bitsandbytes: {bnb.__version__}')
# Test local model loading (no network)
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('$LOCAL_MODEL', trust_remote_code=True)
print(f'  Tokenizer: {len(t)} vocab — local load OK')
"
echo ""

# ---- Step 3: Verify dataset ----
echo "[3/4] Verifying dataset..."
python -c "
import json, os
path = 'data/struq_dataset.json'
if not os.path.exists(path):
    print('  ERROR: Dataset not found!')
    exit(1)
with open(path) as f:
    ds = json.load(f)
print(f'  Dataset: {len(ds)} entries')
from collections import Counter
types = Counter(d['type'] for d in ds)
for t, c in types.most_common():
    print(f'    {t}: {c}')
"
echo ""

# ---- Step 4: Train ----
echo "[4/4] Starting structured instruction tuning..."
echo "  Model: $LOCAL_MODEL (local)"
echo "  Method: QLoRA (r=16, alpha=32)"
echo "  Epochs: 3 | Batch: 2x4=8 | Seq: 8192"
echo "  Estimate: 1-2 hours"
echo ""

python -m struq_defense.run train \
    --epochs 3 \
    --lr 2e-4 \
    --merge \
    --merge-output models/struq/struq_merged_model

echo ""
echo "============================================================"
echo "  Training complete!"
echo "============================================================"
echo ""
echo "  Outputs:"
echo "    LoRA adapter: models/struq/struq_lora_adapter/"
echo "    Merged model: models/struq/struq_merged_model/"
echo ""
echo "  Download results:"
echo "    scp -r user@host:~/AI-Scientist-main/models/struq/ ./models/"
echo ""
