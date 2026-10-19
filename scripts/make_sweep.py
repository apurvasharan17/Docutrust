"""Degradation sweep: noise, JPEG and low-res at 3 levels each, then OCR.

Same job as make_blur.py, for the other degradation types.
Run from repo root (Mac, ~5-10 min):
  python3 scripts/make_sweep.py
Outputs, per condition (e.g. noise_s42):
  data/degraded/<condition>/doc_001.png ...
  data/ocr/<condition>_ocr.jsonl
"""
import glob
import json
import os
import sys
import time

import pytesseract
from PIL import Image

sys.path.insert(0, "src")
from docutrust.degrade import jpeg as jpeg_mod
from docutrust.degrade import lowres as lowres_mod
from docutrust.degrade import noise as noise_mod
from docutrust.ocr.tesseract import ocr_image

CLEAN_DIR = "data/synthetic"

# type -> (function(img, level, doc_id), levels, condition_name)
SWEEP = {
    "noise":  (lambda img, lvl, doc: noise_mod.noise(img, lvl, doc),
               noise_mod.NOISE_LEVELS, noise_mod.condition_name),
    "jpeg":   (lambda img, lvl, doc: jpeg_mod.jpeg(img, lvl),
               jpeg_mod.JPEG_LEVELS, jpeg_mod.condition_name),
    "lowres": (lambda img, lvl, doc: lowres_mod.lowres(img, lvl),
               lowres_mod.LOWRES_LEVELS, lowres_mod.condition_name),
}


def main():
    version = str(pytesseract.get_tesseract_version())
    paths = sorted(glob.glob(os.path.join(CLEAN_DIR, "*.png")))
    os.makedirs("data/ocr", exist_ok=True)

    for kind, (fn, levels, name_of) in SWEEP.items():
        for level in levels:
            t0 = time.time()
            cond = name_of(level)
            img_dir = f"data/degraded/{cond}"
            ocr_path = f"data/ocr/{cond}_ocr.jsonl"
            os.makedirs(img_dir, exist_ok=True)

            with open(ocr_path, "w") as f:
                for path in paths:
                    doc_id = os.path.splitext(os.path.basename(path))[0]
                    out_img = os.path.join(img_dir, doc_id + ".png")
                    fn(Image.open(path), level, doc_id).save(out_img)

                    result = ocr_image(out_img)
                    f.write(json.dumps({
                        "doc_id": doc_id,
                        "condition": cond,
                        "image_path": out_img,
                        "ocr_engine": "tesseract",
                        "ocr_version": version,
                        "ocr_text": result["text"],
                        "words": result["words"],
                    }) + "\n")

            print(f"{cond}: {len(paths)} docs -> {ocr_path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()