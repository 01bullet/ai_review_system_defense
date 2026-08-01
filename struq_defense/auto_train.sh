#!/bin/bash
# ============================================================
# StruQ Defense — AutoDL Cloud Training Script
# ============================================================
# AutoDL 环境: RTX 4090 24GB, PyTorch 2.8.0, CUDA 12.8
# 数据盘: /root/autodl-tmp (50GB)
#
# 使用方法:
#   1. 上传 struq_cloud.tar.gz (~440MB, 不含模型) 到 AutoDL
#   2. 解压: tar xzf struq_cloud.tar.gz
#   3. 运行: bash struq_defense/auto_train.sh
#   → 脚本会自动从 ModelScope 下载模型 (~15GB, 约 30-60 分钟)
# ============================================================
set -e

echo "============================================================"
echo "  StruQ Defense — AutoDL Training"
echo "============================================================"

# ============================================================
# 目录配置
# ============================================================
AUTODL_TMP="/root/autodl-tmp"
HF_CACHE="$AUTODL_TMP/huggingface"
MODEL_OUTPUT="$AUTODL_TMP/struq_output"
LOCAL_MODEL="$AUTODL_TMP/models/qwen2.5-7b"   # 数据盘，空间充足
MODELSCOPE_CACHE="$AUTODL_TMP/modelscope_cache"

export HF_HOME="$HF_CACHE"
export HF_HUB_CACHE="$HF_CACHE/hub"
export STRUQ_LOCAL_MODEL="$LOCAL_MODEL"
mkdir -p "$HF_CACHE" "$MODEL_OUTPUT" "$MODELSCOPE_CACHE" "$(dirname $LOCAL_MODEL)"

# ============================================================
# Step 0: 准备模型 (本地优先 → ModelScope 下载)
# ============================================================
echo ""
echo "[0/5] Preparing model..."
echo "  数据盘: $AUTODL_TMP ($(df -h $AUTODL_TMP | tail -1 | awk '{print $4}') 可用)"

if [ -d "$LOCAL_MODEL" ] && [ -f "$LOCAL_MODEL/config.json" ]; then
    echo "  Local model found: $LOCAL_MODEL ($(du -sh $LOCAL_MODEL | cut -f1))"
    echo "  ✓ 使用本地模型，跳过下载"
else
    echo "  Local model not found. Downloading from ModelScope..."
    echo "  Target: $LOCAL_MODEL"
    echo "  Size: ~15GB, 预计 30-60 分钟"
    echo ""

    # Install modelscope if needed
    pip install modelscope -q 2>&1 | tail -1

    # Download via ModelScope to DATA DISK (not system disk!)
    python -c "
import os, sys, shutil
from pathlib import Path

target = Path('$LOCAL_MODEL')
target.mkdir(parents=True, exist_ok=True)

# Use data disk for cache, NOT system disk
os.environ['MODELSCOPE_CACHE'] = '$MODELSCOPE_CACHE'

print(f'[ModelScope] Downloading Qwen2.5-7B ...')
print(f'  Target: {target}')
print(f'  Cache:  $MODELSCOPE_CACHE')
from modelscope import snapshot_download
tmp_dir = snapshot_download('Qwen/Qwen2.5-7B', cache_dir='$MODELSCOPE_CACHE')

# Move files from nested cache dir to flat target
tmp = Path(tmp_dir)
for f in tmp.rglob('*'):
    if f.is_file():
        dest = target / f.name
        if not dest.exists():
            shutil.move(str(f), str(dest))
# Cleanup
shutil.rmtree(tmp_dir, ignore_errors=True)

# Verify
assert (target / 'config.json').exists(), 'Model download failed!'
total = sum(f.stat().st_size for f in target.rglob('*') if f.is_file())
print(f'Model downloaded: {total/1e9:.1f} GB to {target}')
print('OK')
"
    echo ""
    echo "  ✓ 模型下载完成"
fi
echo ""

# ============================================================
# Step 1: GPU 检查
# ============================================================
echo "[1/5] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ============================================================
# Step 2: 网络配置（仅 pip）
# ============================================================
echo "[2/5] Configuring pip mirror..."
if curl -s --connect-timeout 5 https://mirrors.tuna.tsinghua.edu.cn > /dev/null 2>&1; then
    PIP_INDEX="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
    echo "  pip: 清华源"
elif curl -s --connect-timeout 5 https://pypi.org > /dev/null 2>&1; then
    PIP_INDEX=""
    echo "  pip: 直连"
else
    PIP_INDEX=""
    echo "  pip: 直连 (可能失败)"
fi
echo ""

# ============================================================
# Step 3: 安装依赖
# ============================================================
echo "[3/5] Installing dependencies..."
if [ -n "$PIP_INDEX" ]; then
    pip install transformers peft accelerate bitsandbytes tensorboard -q -i "$PIP_INDEX" 2>&1 | tail -3
else
    pip install transformers peft accelerate bitsandbytes tensorboard -q 2>&1 | tail -3
fi
echo "  Done."
echo ""

# ============================================================
# Step 4: 环境验证
# ============================================================
echo "[4/5] Verifying environment..."
python -c "
import os, sys
import torch
import bitsandbytes as bnb
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
print(f'  bitsandbytes: {bnb.__version__}')

# 验证本地模型
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('$LOCAL_MODEL', trust_remote_code=True)
print(f'  Tokenizer: {len(t)} vocab')
print(f'  Model path: $LOCAL_MODEL')
print(f'  ✓ 本地模型可加载，训练将完全离线')
"

# 验证数据集
python -c "
import json, os
path = 'data/struq_dataset.json'
if not os.path.exists(path):
    print('  WARNING: Dataset not found at data/struq_dataset.json')
    print('  Run locally first: python -m struq_defense.run build-dataset')
else:
    with open(path) as f:
        ds = json.load(f)
    from collections import Counter
    types = Counter(d['type'] for d in ds)
    print(f'  Dataset: {len(ds)} entries')
    for t, c in types.most_common():
        print(f'    {t}: {c}')
"
echo ""

# ============================================================
# Step 5: 训练
# ============================================================
echo "[5/5] Starting training..."
echo "  Model: $LOCAL_MODEL (本地)"
echo "  Method: QLoRA (r=16, alpha=32)"
echo "  Epochs: 3 | Batch: 2x4=8 | Seq: 8192"
echo "  Estimate: 1-2 hours"
echo ""
echo "  监控: tensorboard --logdir $MODEL_OUTPUT/struq_lora_adapter --port 6006"
echo ""

python -m struq_defense.run train \
    --epochs 3 \
    --lr 2e-4 \
    --output "$MODEL_OUTPUT/struq_lora_adapter" \
    --merge \
    --merge-output "$MODEL_OUTPUT/struq_merged_model"

echo ""
echo "============================================================"
echo "  Training complete!"
echo "============================================================"
echo ""
echo "  输出: $MODEL_OUTPUT/"
echo "    LoRA:  $MODEL_OUTPUT/struq_lora_adapter/"
echo "    Merge: $MODEL_OUTPUT/struq_merged_model/"
echo ""
echo "  下载: AutoDL 网页 → 文件管理 → $AUTODL_TMP/struq_output/"
echo ""
