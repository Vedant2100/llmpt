#!/bin/bash
# Run ONCE in JupyterHub terminal: cd ~/llmpt && bash setup.sh
set -e
echo "🔧 Fixing environment..."

# Detect installed torch version and match torchvision/torchaudio to it
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
echo "  Detected torch=${TORCH_VER}"

# Map torch version to compatible torchvision
case "$TORCH_VER" in
    2.4.*)  TV_VER="0.19.1" ; TA_VER="2.4.1" ;;
    2.5.*)  TV_VER="0.20.1" ; TA_VER="2.5.1" ;;
    *)      TV_VER="" ; TA_VER="" ;;  # skip if unknown
esac

# Fix torchvision/torchaudio to match torch (--no-deps so they don't pull torch)
if [ -n "$TV_VER" ]; then
    echo "  Installing torchvision==${TV_VER} torchaudio==${TA_VER} to match torch"
    pip install --no-deps -q \
        "torchvision==${TV_VER}+cu121" "torchaudio==${TA_VER}+cu121" \
        --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || \
    pip install --no-deps -q \
        "torchvision==${TV_VER}" "torchaudio==${TA_VER}" \
        --index-url https://download.pytorch.org/whl/cu121
fi

# Install transformers + trl compatible with torch 2.4/2.5 (--no-deps, no torch pull)
pip install --no-deps -q "transformers>=4.46.0,<4.50.0" "trl>=0.11.0,<0.14.0"

# Install remaining deps normally
pip install -q \
    accelerate datasets peft bitsandbytes huggingface_hub \
    "filelock>=3.12.0" openpyxl \
    sympy==1.12 latex2sympy2 word2number regex pebble timeout-decorator

echo ""
echo "✅ Final versions:"
python -c "
import torch, transformers, trl, torchvision
print(f'  torch={torch.__version__}')
print(f'  torchvision={torchvision.__version__}')
print(f'  transformers={transformers.__version__}')
print(f'  trl={trl.__version__}')
# Quick sanity: can we import Trainer?
from transformers import Trainer
print('  ✅ Trainer imports OK')
"
