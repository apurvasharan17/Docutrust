"""LLM output + gold labels  ->  one record per field, for one condition.

Run from repo root:
  python3 scripts/build_records.py clean
  python3 scripts/build_records.py blur_r1.0
Reads  data/llm/<condition>_llm.jsonl
       data/ocr/<condition>_ocr.jsonl           (Tesseract words + confidences)
       data/ocr_easy/<condition>_easyocr.jsonl  (EasyOCR words, second engine)
Writes data/records/<condition>_records.jsonl
Signals: logprob/margin, OCR confidence, grounding, validation rules,
         doc-level OCR confidence, OCR engine agreement
"""
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, "src")
from docutrust.normalize import label
from docutrust.schema import FieldRecord
from docutrust.signals.logprob import FIELDS, field_signals, parse_values, sha256
from docutrust.signals.ocr import ocr_signals, doc_ocr_conf, engine_agreement
from docutrust.signals.validation import validation_signals

GOLD_DIR = "data/synthetic"  # gold is the same for every condition


def load_words(path):
    """doc_id -> list of {word, conf} from one OCR output file."""
    words = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            words[r["doc_id"]] = r["words"]
    return words


def build(condition):
    llm_path = f"data/llm/{condition}_llm.jsonl"
    out_path = f"data/records/{condition}_records.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    counts = {"CORRECT": 0, "WRONG": 0, "MISS": 0, "NO_SIGNAL": 0}
    tess_words = load_words(f"data/ocr/{condition}_ocr.jsonl")
    easy_words = load_words(f"data/ocr_easy/{condition}_easyocr.jsonl")

    with open(llm_path) as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            raw = json.loads(line)
            doc = raw["doc_id"]
            with open(os.path.join(GOLD_DIR, doc + ".json")) as g:
                gold = json.load(g)
            values = parse_values(raw["response"])
            # rules look across fields (e.g. CGST == SGST), so compute once per doc
            checks = validation_signals(values, raw["ocr_text"])

            for field in FIELDS:
                value = values.get(field)
                sig = field_signals(raw["response"], raw["logprobs"], field)
                if sig is None:
                    counts["NO_SIGNAL"] += 1
                    sig = {}
                # signals that do not depend on logprobs: every field gets them
                sig.update(ocr_signals(value, tess_words[doc]))
                sig["validation_passed"] = checks[field]
                sig["doc_ocr_conf"] = doc_ocr_conf(tess_words[doc])
                sig["engine_agreement"] = engine_agreement(value, easy_words[doc])

                lab = label(value, gold[field], field)
                counts[lab] += 1

                rec = FieldRecord(
                    doc_id=doc,
                    field_name=field,
                    ocr_text_sha256=sha256(raw["ocr_text"]),
                    condition=condition,
                    raw_value=value,
                    gold_value=gold[field],
                    label=lab,
                    is_correct=(lab == "CORRECT"),
                    model=raw["model"],
                    ollama_version=raw["ollama_version"],
                    kv_cache_type=raw["kv_cache"],
                    hardware=raw["gpu"],
                    run_timestamp=now,
                    **sig,
                )
                f_out.write(json.dumps(rec.to_dict()) + "\n")

    print(f"{condition}: {counts} -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python3 scripts/build_records.py <condition>")
    build(sys.argv[1])