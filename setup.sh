#!/bin/bash
# Run ONCE in JupyterHub terminal: cd ~/llmpt && bash setup.sh
set -e
echo "🔧 Fixing environment..."

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
