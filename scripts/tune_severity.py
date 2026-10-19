"""Pick degradation severities BEFORE spending any GPU time.

For each degradation type and each candidate level, degrade the first N clean
invoices in memory, run Tesseract, and measure "survival": the share of gold
field values whose words all still appear in the OCR output. The existing blur
levels are included as the reference to match (mild / medium / heavy).

Run from repo root (takes a few minutes, Mac only):
  python3 scripts/tune_severity.py
Nothing is written to disk.
"""
import glob
import json
import os
import sys
import time

import pytesseract
from PIL import Image

sys.path.insert(0, "src")
from docutrust.degrade.blur import blur
from docutrust.degrade.jpeg import jpeg
from docutrust.degrade.lowres import lowres
from docutrust.degrade.noise import noise
from docutrust.signals.logprob import FIELDS
from docutrust.signals.ocr import match_conf, value_words

N_DOCS = 20


CANDIDATES = {
    "noise sigma": (lambda img, lvl, doc: noise(img, lvl, doc), [41, 42, 43]),
}

def ocr_words(img):
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    return [{"word": w, "conf": float(c)}
            for w, c in zip(data["text"], data["conf"])
            if w.strip() and float(c) >= 0]


def survives(gold_value, words):
    """True if every word of the gold value is found in the OCR words."""
    vws = value_words(str(gold_value))
    return bool(vws) and all(match_conf(v, words) is not None for v in vws)


def main():
    paths = sorted(glob.glob("data/synthetic/*.png"))[:N_DOCS]
    docs = []
    for p in paths:
        doc_id = os.path.splitext(os.path.basename(p))[0]
        with open(p.replace(".png", ".json")) as f:
            gold = json.load(f)
        docs.append((doc_id, Image.open(p).copy(), gold))

    print(f"survival = share of gold values still readable by Tesseract ({N_DOCS} docs)\n")
    for name, (fn, levels) in CANDIDATES.items():
        print(name)
        for lvl in levels:
            t0 = time.time()
            hits = total = 0
            for doc_id, img, gold in docs:
                words = ocr_words(fn(img, lvl, doc_id))
                for field in FIELDS:
                    total += 1
                    hits += survives(gold[field], words)
            print(f"   {str(lvl):>5}   survival={hits / total:.2f}   ({time.time() - t0:.0f}s)")
        print()


if __name__ == "__main__":
    main()