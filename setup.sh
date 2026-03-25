#!/bin/bash
# Run ONCE in JupyterHub terminal: cd ~/llmpt && bash setup.sh
set -e
echo "🔧 Fixing environment..."

# ── CUDA Setup (Fixes DeepSpeed compilation errors) ──────────────────
# Search for nvcc to find CUDA_HOME
NVCC_PATH=$(which nvcc 2>/dev/null || true)

if [ -n "$NVCC_PATH" ]; then
    export CUDA_HOME=$(dirname $(dirname $NVCC_PATH))
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export DS_BUILD_OPS=1
    echo "  Found CUDA at $CUDA_HOME (nvcc: $NVCC_PATH)"
else
    echo "  ⚠️ nvcc (CUDA compiler) not found. Unsetting CUDA_HOME to bypass DeepSpeed metadata checks..."
    unset CUDA_HOME
    export DS_BUILD_OPS=0
fi

# Upgrade base tools before installing heavy hitters
pip install -q --upgrade pip setuptools wheel

# Detect torch version and force upgrade to 2.4.0 if too old
# (Accelerate 1.0+ requires torch 2.4+ for torch.amp.GradScaler)
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "0.0.0")
echo "  Detected torch=${TORCH_VER}"

if [[ "$(printf '%s\n' "2.4.0" "$TORCH_VER" | sort -V | head -n1)" != "2.4.0" ]]; then
    echo "  🚀 Upgrading torch to 2.4.0 (required for accelerate/transformers compatibility)..."
    pip install --upgrade -q torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
fi

# Install transformers + trl compatible with torch 2.4/2.5 (--no-deps, no torch pull)
pip install --no-deps -q "transformers>=4.46.0,<4.50.0" "trl>=0.11.0,<0.14.0"

# Install remaining deps normally
pip install -q \
    accelerate datasets peft bitsandbytes huggingface_hub rich deepspeed vllm \
    "filelock>=3.12.0" openpyxl \
    sympy==1.12 latex2sympy2 word2number regex pebble timeout-decorator

echo ""
echo "✅ Final versions:"
python -c "
import torch, transformers, trl, accelerate
print(f'  torch={torch.__version__}')
print(f'  transformers={transformers.__version__}')
print(f'  trl={trl.__version__}')
print(f'  accelerate={accelerate.__version__}')
# Quick sanity: can we import Trainer?
from transformers import Trainer
print('  ✅ Trainer imports OK')
"
