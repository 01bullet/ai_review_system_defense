#!/bin/bash
# ============================================================
# V3 Training — Cloud (RTX 4090 24GB) — AutoDL
# ============================================================
# Two-phase training:
#   Phase A (Review):  PeerRead human alignment, no attacks, from base model
#   Phase B (Defense): API expansion + defense, from Phase A adapter
#
# seq_len=8192 — fits complete PeerRead papers on 24GB VRAM
#
# Usage:
#   1. Upload changed files to AutoDL:
#      - struq_defense/config_v3.py
#      - struq_defense/train_v3.py
#      - struq_defense/train_v3_cloud.sh (this file)
#   2. Run: bash struq_defense/train_v3_cloud.sh
# ============================================================
set -e

echo "============================================================"
echo "  V3 Training — 8192 seq_len — RTX 4090 24GB"
echo "  Phase A: Review Capability (base model → PeerRead)"
echo "  Phase B: Defense Capability (Phase A → attacks)"
echo "============================================================"

# ============================================================
# Directory config
# ============================================================
AUTODL_TMP="/root/autodl-tmp"
HF_CACHE="$AUTODL_TMP/huggingface"
MODEL_OUTPUT="$AUTODL_TMP/struq_output"
LOCAL_MODEL="$AUTODL_TMP/models/qwen2.5-7b"
MODELSCOPE_CACHE="$AUTODL_TMP/modelscope_cache"

export HF_HOME="$HF_CACHE"
export HF_HUB_CACHE="$HF_CACHE/hub"
export STRUQ_LOCAL_MODEL="$LOCAL_MODEL"
export STRUQ_V3_MODELS_DIR="$AUTODL_TMP/struq_output/v3"
mkdir -p "$HF_CACHE" "$MODEL_OUTPUT" "$MODELSCOPE_CACHE" "$(dirname $LOCAL_MODEL)"

echo ""
echo "  Data disk: $AUTODL_TMP ($(df -h $AUTODL_TMP | tail -1 | awk '{print $4}') available)"
echo "  Model output: $STRUQ_V3_MODELS_DIR"
echo ""

# ============================================================
# Step 0: Prepare model
# ============================================================
echo "[0/6] Preparing base model..."
if [ -d "$LOCAL_MODEL" ] && [ -f "$LOCAL_MODEL/config.json" ]; then
    echo "  Local model found: $LOCAL_MODEL ($(du -sh $LOCAL_MODEL | cut -f1))"
    echo "  Skipping download"
else
    echo "  Downloading Qwen2.5-7B from ModelScope (~15GB, ~30-60 min)..."
    pip install modelscope -q 2>&1 | tail -1

    python -c "
import os, sys, shutil
from pathlib import Path

target = Path('$LOCAL_MODEL')
target.mkdir(parents=True, exist_ok=True)
os.environ['MODELSCOPE_CACHE'] = '$MODELSCOPE_CACHE'

print(f'[ModelScope] Downloading Qwen2.5-7B ...')
from modelscope import snapshot_download
tmp_dir = snapshot_download('Qwen/Qwen2.5-7B', cache_dir='$MODELSCOPE_CACHE')

tmp = Path(tmp_dir)
for f in tmp.rglob('*'):
    if f.is_file():
        dest = target / f.name
        if not dest.exists():
            shutil.move(str(f), str(dest))
shutil.rmtree(tmp_dir, ignore_errors=True)

assert (target / 'config.json').exists(), 'Model download failed!'
total = sum(f.stat().st_size for f in target.rglob('*') if f.is_file())
print(f'Model downloaded: {total/1e9:.1f} GB')
print('OK')
"
    echo "  Model ready"
fi
echo ""

# ============================================================
# Step 1: GPU check
# ============================================================
echo "[1/6] GPU info..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# ============================================================
# Step 2: Dependencies
# ============================================================
echo "[2/6] Installing dependencies..."
pip install transformers peft accelerate bitsandbytes tensorboard -q 2>&1 | tail -3

# flash-attn for memory-efficient attention at seq_len=8192
echo "  Installing flash-attn (for 8192 seq_len)..."
pip install flash-attn --no-build-isolation -q 2>&1 | tail -3 || {
    echo "  WARNING: flash-attn install failed, falling back to sdpa"
    echo "  This may cause OOM at 8192 seq_len. Consider reducing seq_len."
}
echo "  Done"
echo ""

# Fix OMP_NUM_THREADS (AutoDL sometimes sets it to garbage)
export OMP_NUM_THREADS=8
# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================
# Step 3: Environment verification
# ============================================================
echo "[3/6] Verifying environment..."
python -c "
import torch, bitsandbytes as bnb
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
print(f'  bitsandbytes: {bnb.__version__}')

# Verify local model
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('$LOCAL_MODEL', trust_remote_code=True)
print(f'  Tokenizer: {len(t)} vocab')
print(f'  Model: $LOCAL_MODEL — local load OK')
print(f'  Training seq_len: 8192 (was 1024)')

# Verify PeerRead dataset
import os
pr = os.environ.get('STRUQ_PEERREAD_DIR', 'example_papers/PeerRead-master/PeerRead-master/data')
acl = os.path.join(pr, 'acl_2017/train/reviews')
if os.path.isdir(acl):
    count = len([f for f in os.listdir(acl) if f.endswith('.json')])
    print(f'  PeerRead ACL 2017: {count} reviews found')
else:
    print(f'  WARNING: PeerRead not found at {pr}')
"
echo ""

# ============================================================
# Step 4: Phase A — Review Capability
# ============================================================
echo "[4/6] Phase A: Human-Review Alignment (from base model)"
echo "  Papers: 242 (ACL + CoNLL + ICLR)"
echo "  Attacks: None"
echo "  label_masking: True"
echo "  max_seq_length: 8192"
echo "  Epochs: 3 | Batch: 1x8=8 | LR: 5e-5"
echo "  Starting from: Qwen2.5-7B (base — no adapter)"
echo ""
echo "  Estimate: ~4-8 hours"
echo ""

python -m struq_defense.train_v3 --phase human_align

echo ""
echo "  Phase A complete!"
echo "  Adapter saved to: $STRUQ_V3_MODELS_DIR/v3a_human_align/"
echo ""

# ============================================================
# Step 5: Phase B — Defense Capability
# ============================================================
echo "[5/6] Phase B: API Expansion + Light Defense (from Phase A)"
echo "  Papers: 549 (ICLR + arxiv CS.AI)"
echo "  Attacks: 37% (naive + completion + format + boundary)"
echo "  Starting from: Phase A adapter"
echo "  Epochs: 3 | Batch: 1x8=8 | LR: 3e-5"
echo ""
echo "  ⚠ This phase calls DeepSeek API for arxiv paper reviews"
echo "  Set DEEPSEEK_API_KEY or use --skip-reviews flag"
echo ""
echo "  Estimate: ~6-12 hours (including API calls)"
echo ""

if [ -n "$DEEPSEEK_API_KEY" ]; then
    python -m struq_defense.train_v3 --phase api_expand
else
    echo "  WARNING: DEEPSEEK_API_KEY not set. Using --skip-reviews."
    echo "  arxiv papers will use synthetic placeholder reviews."
    python -m struq_defense.train_v3 --phase api_expand --skip-reviews
fi

echo ""
echo "  Phase B complete!"
echo "  Adapter saved to: $STRUQ_V3_MODELS_DIR/v3b_api_defense/"
echo ""

# ============================================================
# Step 6: Summary
# ============================================================
echo "[6/6] Training complete!"
echo ""
echo "============================================================"
echo "  Output Files"
echo "============================================================"
echo ""
echo "  Phase A (Review Only):"
echo "    $STRUQ_V3_MODELS_DIR/v3a_human_align/"
echo ""
echo "  Phase B (Review + Defense):"
echo "    $STRUQ_V3_MODELS_DIR/v3b_api_defense/"
echo ""
echo "  To download to local:"
echo "    cd $STRUQ_V3_MODELS_DIR"
echo "    tar czf v3a_review.tar.gz v3a_human_align/"
echo "    tar czf v3b_defense.tar.gz v3b_api_defense/"
echo ""
echo "  Then download via AutoDL file manager"
echo "============================================================"
