
import glob
import json
import os
import sys

import pytesseract
from PIL import Image

sys.path.insert(0, "src")
from docutrust.degrade.blur import BLUR_LEVELS, blur, condition_name
from docutrust.ocr.tesseract import ocr_image

CLEAN_DIR = "data/synthetic"


def main():
    version = str(pytesseract.get_tesseract_version())
    paths = sorted(glob.glob(os.path.join(CLEAN_DIR, "*.png")))
    os.makedirs("data/ocr", exist_ok=True)

    for radius in BLUR_LEVELS:
        cond = condition_name(radius)
        img_dir = f"data/degraded/{cond}"
        ocr_path = f"data/ocr/{cond}_ocr.jsonl"
        os.makedirs(img_dir, exist_ok=True)

        with open(ocr_path, "w") as f:
            for path in paths:
                doc_id = os.path.splitext(os.path.basename(path))[0]
                out_img = os.path.join(img_dir, doc_id + ".png")
                blur(Image.open(path), radius).save(out_img)

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

        print(f"{cond}: {len(paths)} docs -> {ocr_path}")


if __name__ == "__main__":
    main()