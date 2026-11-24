
import io

from PIL import Image

JPEG_LEVELS = [10, 5, 1]  


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