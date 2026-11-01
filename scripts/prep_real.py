"""Real receipts: fix orientation, resize, run Tesseract.

Run from repo root:
  python3 scripts/prep_real.py
Reads   data/raw/<receipt>.jpg
Writes  data/real/<receipt>.png             (upright, at most 2000 px on the long side)
        data/ocr/real_ocr.jsonl             (condition "real", same format as every OCR file)
Gold labels are written by hand in data/real/<receipt>.json.
"""
import json
import os
import sys

import pytesseract
from PIL import Image, ImageOps

sys.path.insert(0, "src")
from docutrust.ocr.tesseract import ocr_image

RECEIPTS = ["receipt_002"]      # receipt_001 is handwritten: out of scope
RAW_DIR = "data/raw"
OUT_DIR = "data/real"
MAX_SIDE = 2000                 # phone photos are huge; this keeps OCR fast and light


def prepare(name):
    img = Image.open(os.path.join(RAW_DIR, name + ".jpg"))
    img = ImageOps.exif_transpose(img)          # phone photos store rotation in metadata
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))     # keeps the aspect ratio
    out = os.path.join(OUT_DIR, name + ".png")
    img.convert("RGB").save(out)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    version = str(pytesseract.get_tesseract_version())
    with open("data/ocr/real_ocr.jsonl", "w") as f:
        for name in RECEIPTS:
            path = prepare(name)
            result = ocr_image(path)
            f.write(json.dumps({
                "doc_id": name,
                "condition": "real",
                "image_path": path,
                "ocr_engine": "tesseract",
                "ocr_version": version,
                "ocr_text": result["text"],
                "words": result["words"],
            }) + "\n")
            print(f"{name}: {Image.open(path).size} px, {len(result['words'])} words")
    print("wrote data/ocr/real_ocr.jsonl")


if __name__ == "__main__":
    main()