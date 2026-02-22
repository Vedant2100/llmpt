"""Shared utilities: GPU cleanup, seeding, answer extraction, storage."""

import gc
import os
import re
import glob
import shutil
import random
import subprocess

import torch
import numpy as np


def clear_gpu():
    """Clear GPU memory, resilient to corrupted CUDA context."""
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"⚠️ CUDA cleanup warning (non-fatal): {e}")


def set_random_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🎲 Random seeds set to {seed}")


def extract_answer_from_response(text: str) -> str:
    """
    Unified answer extraction for GSM8K.

    Returns the last number found in *text* (after ``####`` if present),
    or ``"None"`` when no number is found.  Must be used consistently
    during both training and evaluation.
    """
    if "####" in text:
        text = text.split("####")[-1].strip()
    nums = re.findall(r'\d+', text)
    return nums[-1] if nums else "None"


def cleanup_storage():
    """Aggressive cleanup to free disk space (pip cache, HF cache, artifacts)."""
    print('🧹 Purging caches & training artifacts...')

    subprocess.run(
        ['pip', 'cache', 'purge'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    hf_cache = os.path.expanduser('~/.cache/huggingface/hub')
    if os.path.exists(hf_cache):
        shutil.rmtree(hf_cache, ignore_errors=True)
        print('  ✓ Cleared HuggingFace cache')

    artifacts = [
        'reward_model*', 'coldstart_dorado*',
        'dorado_final*', 'dorado_round_*', 'runs/',
    ]
    for pattern in artifacts:
        for path in glob.glob(pattern):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
    print('  ✓ Removed training artifacts')

    total, used, free = shutil.disk_usage('/')
    print(f'\n📊 Disk: {used/total:.1%} used  ({free/1024**3:.2f} GB free)')
    if free / 1024**3 < 2:
        print('   ⚠️ Still low on space')
    else:
        print('   ✅ Sufficient space')
