"""Gaussian noise degradation: grainy low-light photos and noisy scans.

Deterministic: the noise pattern comes from a seed built from the doc_id and
the noise level, so the same (doc_id, sigma) always gives the same image.
"""
import hashlib

import numpy as np
from PIL import Image

# tuned on 20 docs: Tesseract survival 0.97 / 0.47 / 0.04 (sharp cliff past sigma 40)
NOISE_LEVELS = [40, 42, 43]   # mild, medium, heavy


def seed_for(doc_id, sigma):
    """Stable integer seed from doc_id + level (Python's hash() is not stable)."""
    h = hashlib.sha256(f"{doc_id}|noise|{sigma}".encode()).hexdigest()
    return int(h[:8], 16)


def noise(img, sigma, doc_id):
    """Add zero-mean Gaussian noise with standard deviation `sigma` (pixel units)."""
    rng = np.random.default_rng(seed_for(doc_id, sigma))
    arr = np.asarray(img, dtype=np.float32)
    noisy = arr + rng.normal(0.0, sigma, size=arr.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), mode=img.mode)


def condition_name(sigma):
    return f"noise_s{sigma}"