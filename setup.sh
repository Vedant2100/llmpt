#!/bin/bash
# Run this ONCE in the JupyterHub terminal before using the notebook.
# Usage: cd ~/llmpt && bash setup.sh

set -e
echo "🔧 Fixing environment..."

# Step 1: Force-reinstall torch first (from cu121 index, ignoring deps)
pip install --force-reinstall --no-deps -q \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Step 2: Install everything else WITHOUT letting pip touch torch
pip install --no-deps -q "transformers>=4.46.0,<5.0.0" "trl>=0.11.0,<0.14.0"
pip install -q \
    accelerate datasets peft bitsandbytes huggingface_hub \
    "filelock>=3.12.0" openpyxl \
    sympy==1.12 latex2sympy2 word2number regex pebble timeout-decorator

echo ""
echo "✅ Versions installed:"
python -c "
import torch, transformers, trl
print(f'  torch={torch.__version__}')
print(f'  transformers={transformers.__version__}')
print(f'  trl={trl.__version__}')
assert '2.5.1' in torch.__version__, f'ERROR: torch is {torch.__version__}, expected 2.5.1!'
print()
print('All good. Restart kernel, then run the notebook.')
"
