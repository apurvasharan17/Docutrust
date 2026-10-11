"""LLM output + gold labels  ->  one record per field, for one condition.

Run from repo root:
  python3 scripts/build_records.py clean
  python3 scripts/build_records.py blur_r1.0
Reads  data/llm/<condition>_llm.jsonl
Writes data/records/<condition>_records.jsonl
"""
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, "src")
from docutrust.normalize import label
from docutrust.schema import FieldRecord
from docutrust.signals.logprob import FIELDS, field_signals, parse_values, sha256

GOLD_DIR = "data/synthetic"  # gold is the same for every condition


def build(condition):
    llm_path = f"data/llm/{condition}_llm.jsonl"
    out_path = f"data/records/{condition}_records.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    counts = {"CORRECT": 0, "WRONG": 0, "MISS": 0, "NO_SIGNAL": 0}

    with open(llm_path) as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            raw = json.loads(line)
            with open(os.path.join(GOLD_DIR, raw["doc_id"] + ".json")) as g:
                gold = json.load(g)
            values = parse_values(raw["response"])

            for field in FIELDS:
                sig = field_signals(raw["response"], raw["logprobs"], field)
                if sig is None:
                    counts["NO_SIGNAL"] += 1
                    sig = {}
                lab = label(values.get(field), gold[field], field)
                counts[lab] += 1

                rec = FieldRecord(
                    doc_id=raw["doc_id"],
                    field_name=field,
                    ocr_text_sha256=sha256(raw["ocr_text"]),
                    condition=condition,
                    raw_value=values.get(field),
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