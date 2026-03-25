#!/bin/bash
# Run ONCE in JupyterHub terminal: cd ~/llmpt && bash setup.sh
set -e
echo "🔧 Fixing environment..."

# ── CUDA Setup (Fixes DeepSpeed compilation errors) ──────────────────
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
if [ -d "$CUDA_HOME" ]; then
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    echo "  Found CUDA at $CUDA_HOME"
else
    echo "  ⚠️ CUDA_HOME ($CUDA_HOME) not found. DeepSpeed might fail to compile kernels."
fi

# Force DeepSpeed to try building ops if nvcc is found
export DS_BUILD_OPS=1

# Detect installed torch version and match torchvision/torchaudio to it
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
echo "  Detected torch=${TORCH_VER}"

# We are running Math/Text models, so torchvision and torchaudio are completely unnecessary.
# In fact, broken torchvision C++ extensions crash HuggingFace 'transformers' on import.
# Let's cleanly rip them out of the container so they stop causing tracebacks!
pip uninstall -y torchvision torchaudio || true

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
import torch, transformers, trl
print(f'  torch={torch.__version__}')
print(f'  transformers={transformers.__version__}')
print(f'  trl={trl.__version__}')
# Quick sanity: can we import Trainer?
from transformers import Trainer
print('  ✅ Trainer imports OK')
"
