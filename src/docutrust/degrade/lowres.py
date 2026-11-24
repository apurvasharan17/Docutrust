
from PIL import Image


LOWRES_LEVELS = [0.55, 0.35, 0.3]   


def lowres(img, factor):
    w, h = img.size
    small = img.resize((max(1, int(w * factor)), max(1, int(h * factor))),
                       Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def condition_name(factor):
    return f"lowres_f{factor}"