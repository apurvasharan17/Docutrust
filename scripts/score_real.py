"""Apply the fusion model trained on SYNTHETIC data to real receipts.

Run from repo root (after build_records.py real):
  python3 scripts/score_real.py

The models are trained on Qwen's answered fields from all 13 synthetic
conditions, then applied unchanged to the real receipts: the model has never
seen a real receipt. With only a few receipts this is a case study, not a
statistical result, so every field is printed.

Fields the LLM returned as null are shown but not scored (the fusion model
covers answered fields only). On real receipts a null can be correct, when the
field is genuinely not printed.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "scripts")
from fuse import CONDITIONS, FEATURES, feature_value, new_model

REAL_RECORDS = "data/records/real_records.jsonl"


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def features(rows):
    return np.array([[feature_value(r, f) for f in FEATURES] for r in rows])


def main():
    train = []
    for c in CONDITIONS:
        train += load(f"data/records/{c}_records.jsonl")
    train = [r for r in train if r["raw_value"] is not None]
    X = features(train)
    y = np.array([1 if r["is_correct"] else 0 for r in train])
    lp = [FEATURES.index("logprob_geo")]

    fused = new_model().fit(X, y)
    platt = new_model().fit(X[:, lp], y)
    print(f"trained on {len(train)} synthetic answered fields (Qwen, 13 conditions)\n")

    real = load(REAL_RECORDS)
    for doc in sorted({r["doc_id"] for r in real}):
        print(f"=== {doc} ===")
        print(f"{'field':16s}{'model answer':24s}{'true value':24s}{'label':9s}"
              f"{'raw':>7s}{'platt':>7s}{'fused':>7s}")
        for r in (r for r in real if r["doc_id"] == doc):
            ans = "null" if r["raw_value"] is None else str(r["raw_value"])
            gold = "null" if r["gold_value"] is None else str(r["gold_value"])
            raw = r.get("logprob_geo")
            raw_txt = f"{raw:.3f}" if raw is not None else "  -"
            if r["raw_value"] is None:
                p_txt = f_txt = "  -"                 # not scored: answered fields only
            else:
                x = features([r])
                p_txt = f"{platt.predict_proba(x[:, lp])[0, 1]:.3f}"
                f_txt = f"{fused.predict_proba(x)[0, 1]:.3f}"
            print(f"{r['field_name']:16s}{ans[:22]:24s}{gold[:22]:24s}{r['label']:9s}"
                  f"{raw_txt:>7s}{p_txt:>7s}{f_txt:>7s}")
        print()

    print("raw = logprob_geo, platt = recalibrated logprob, fused = all signals")
    print("good sign: CORRECT fields high, WRONG fields clearly lower under 'fused'")


if __name__ == "__main__":
    main()