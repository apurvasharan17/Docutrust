"""Blur degradation. Deterministic: same image + same radius = same output."""
from PIL import ImageFilter

BLUR_LEVELS = [1.0, 1.5, 2.0]  # mild, medium, heavy


def blur(img, radius):
    return img.filter(ImageFilter.GaussianBlur(radius))


def condition_name(radius):
    return f"blur_r{radius}"