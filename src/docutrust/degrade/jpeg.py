"""JPEG compression degradation: WhatsApp-forwarded and heavily compressed images.

Deterministic: the JPEG encoder gives the same output for the same input and quality.
"""
import io

from PIL import Image

# tuned on 20 docs: Tesseract survival 0.97 / 0.89 / 0.67.
# JPEG never gets "heavy" on printed text, even at quality 1; we keep its natural range.
JPEG_LEVELS = [10, 5, 1]   # quality, lower = worse


def jpeg(img, quality):
    """Encode as JPEG at `quality` (1-95) and decode back."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out.convert(img.mode)


def condition_name(quality):
    return f"jpeg_q{quality}"