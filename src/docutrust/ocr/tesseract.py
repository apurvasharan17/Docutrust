"""OCR stage: run Tesseract on every synthetic invoice and cache the result.

Output: data/ocr/clean_ocr.jsonl  (one JSON object per line, one line per doc)
This file is what gets uploaded to Kaggle — the LLM stage never touches images.
"""
import glob
import json
import os

import pytesseract
from PIL import Image

IN_DIR = "data/synthetic"
OUT_PATH = "data/ocr/clean_ocr.jsonl"


def ocr_image(path):
    img = Image.open(path)

    # full text, keeps line breaks (this is what the LLM reads)
    text = pytesseract.image_to_string(img)

    # per-word boxes + confidences (for the OCR-confidence signal later)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    words = []
    for word, conf in zip(data["text"], data["conf"]):
        conf = float(conf)
        if word.strip() and conf >= 0:  # -1 = layout box, not a word
            words.append({"word": word, "conf": conf})

    return {"text": text, "words": words}


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    version = str(pytesseract.get_tesseract_version())
    paths = sorted(glob.glob(os.path.join(IN_DIR, "*.png")))

    with open(OUT_PATH, "w") as f:
        for path in paths:
            doc_id = os.path.splitext(os.path.basename(path))[0]
            result = ocr_image(path)
            record = {
                "doc_id": doc_id,
                "condition": "clean",
                "image_path": path,
                "ocr_engine": "tesseract",
                "ocr_version": version,
                "ocr_text": result["text"],
                "words": result["words"],
            }
            f.write(json.dumps(record) + "\n")
            print(f"{doc_id}: {len(result['words'])} words")

    print(f"wrote {len(paths)} docs to {OUT_PATH}")


if __name__ == "__main__":
    main()