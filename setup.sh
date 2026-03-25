#!/bin/bash
# Run this ONCE in the JupyterHub terminal before using the notebook.
# Usage: bash setup.sh

set -e
echo "🔧 Fixing environment..."

# 1. Restore torch to match torchvision/torchaudio
pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. Pin transformers + trl to versions compatible with torch 2.5.x
pip install -q "transformers>=4.46.0,<5.0.0" "trl>=0.11.0,<0.14.0"

# 3. Install other deps
pip install -q accelerate datasets peft bitsandbytes huggingface_hub \
    "filelock>=3.12.0" openpyxl \
    sympy==1.12 latex2sympy2 word2number regex pebble timeout-decorator

echo ""
echo "✅ Done. Versions installed:"
python -c "import torch, transformers, trl; print(f'  torch={torch.__version__}  transformers={transformers.__version__}  trl={trl.__version__}')"
echo ""
echo "Now open the notebook and run all cells."
