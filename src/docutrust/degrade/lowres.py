"""Low-resolution degradation: photos taken from far away, thumbnails.

The image is shrunk by `factor` and scaled back to its original size, so the
OCR sees an image of the same size that has lost fine detail.
Deterministic: resizing has no randomness.
"""
from PIL import Image

# tuned on 20 docs: Tesseract survival 0.97 / 0.29 / 0.15 (matches blur 0.98 / 0.32 / 0.17)
LOWRES_LEVELS = [0.55, 0.35, 0.3]   # scale factor, lower = worse


def lowres(img, factor):
    w, h = img.size
    small = img.resize((max(1, int(w * factor)), max(1, int(h * factor))),
                       Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def condition_name(factor):
    return f"lowres_f{factor}"