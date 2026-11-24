
import hashlib

import numpy as np
from PIL import Image


NOISE_LEVELS = [40, 42, 43]  


def seed_for(doc_id, sigma):
    h = hashlib.sha256(f"{doc_id}|noise|{sigma}".encode()).hexdigest()
    return int(h[:8], 16)


def noise(img, sigma, doc_id):
    rng = np.random.default_rng(seed_for(doc_id, sigma))
    arr = np.asarray(img, dtype=np.float32)
    noisy = arr + rng.normal(0.0, sigma, size=arr.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), mode=img.mode)


def condition_name(sigma):
    return f"noise_s{sigma}"