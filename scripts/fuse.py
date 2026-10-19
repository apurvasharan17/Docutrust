"""Fusion: combine all confidence signals into one calibrated score.

Run from repo root (after build_records.py for every condition):
  python3 scripts/fuse.py
Writes results/fusion_summary.json

SCOPE: answered fields only (the LLM returned a value).
  On the synthetic set every field is present, so every null is wrong by
  construction. A model that saw nulls would just learn "null -> wrong",
  which says nothing about real receipts where fields can be truly absent.
  Confident nulls are reported separately, not hidden inside the fusion.

THREE SCORES per field, all evaluated out-of-fold:
  raw_logprob : logprob_geo as is                 (the baseline we criticise)
  platt       : logistic regression on logprob_geo alone
                (fair baseline: gets the same re-scaling the fused model gets)
  fused       : logistic regression on all signals (our method)

CROSS-VALIDATION groups by doc_id: the same invoice (clean or blurred) is
never in train and test at the same time.
"""
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "scripts")
from compute_ece import expected_calibration_error, auroc, N_BINS

CONDITIONS = ["clean", "blur_r1.0", "blur_r1.5", "blur_r2.0"]
FEATURES = ["logprob_geo", "logprob_min", "margin_min", "ocr_conf_min",
            "engine_agreement", "validation_passed", "doc_ocr_conf", "grounded"]
OPTIONAL = ["self_consistency"]   # used automatically once it is in the records
FOLDS = 5


def load_answered():
    rows = []
    for c in CONDITIONS:
        with open(f"data/records/{c}_records.jsonl") as f:
            rows += [json.loads(line) for line in f]
    return [r for r in rows if r["raw_value"] is not None]


def feature_value(r, name):
    """Answered field with no OCR support for its value -> 0.0 (no evidence)."""
    v = r.get(name)
    return 0.0 if v is None else float(v)


def new_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def oof_predict(X, y, groups):
    """Out-of-fold probabilities: each row is scored by a model that never saw its doc."""
    pred = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=FOLDS).split(X, y, groups):
        model = new_model().fit(X[train], y[train])
        pred[test] = model.predict_proba(X[test])[:, 1]
    return pred


def safe_auroc(scores, labels):
    if len(set(labels)) < 2:
        return None          # e.g. clean: every answered field is correct
    return auroc(scores, labels)


def main():
    rows = load_answered()
    feats = FEATURES + [f for f in OPTIONAL if any(r.get(f) is not None for r in rows)]

    X = np.array([[feature_value(r, f) for f in feats] for r in rows])
    y = np.array([1 if r["is_correct"] else 0 for r in rows])
    groups = np.array([r["doc_id"] for r in rows])
    cond = np.array([r["condition"] for r in rows])
    lp = feats.index("logprob_geo")

    scores = {
        "raw_logprob": X[:, lp],
        "platt": oof_predict(X[:, [lp]], y, groups),
        "fused": oof_predict(X, y, groups),
    }

    print(f"answered fields: {len(rows)}")
    print(f"features: {feats}\n")
    summary = {"features": feats, "n_answered": len(rows), "folds": FOLDS,
               "conditions": {}}

    for c in CONDITIONS + ["all"]:
        mask = np.ones(len(y), dtype=bool) if c == "all" else cond == c
        labels = [bool(v) for v in y[mask]]
        res = {"n": int(mask.sum()), "accuracy": float(y[mask].mean())}
        print(f"{c}: n={res['n']}, accuracy={res['accuracy']:.3f}")

        for name, s in scores.items():
            confs = [float(v) for v in s[mask]]
            ece = expected_calibration_error(confs, labels, N_BINS)
            au = safe_auroc(confs, labels)
            res[name] = {"avg_conf": float(np.mean(confs)), "ece": ece, "auroc": au}
            au_txt = f"{au:.3f}" if au is not None else "  -  "
            print(f"   {name:12s} avg_conf={np.mean(confs):.3f}  "
                  f"ECE={ece:.3f}  AUROC={au_txt}")
        summary["conditions"][c] = res
        print()

    # which signals does the fused model lean on? (fit once on all data)
    full = new_model().fit(X, y)
    weights = dict(zip(feats, full[-1].coef_[0].tolist()))
    print("signal weights (standardised, + means 'more likely correct'):")
    for f, w in sorted(weights.items(), key=lambda t: -abs(t[1])):
        print(f"   {f:18s} {w:+.3f}")
    summary["weights"] = weights

    with open("results/fusion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nsaved results/fusion_summary.json")


if __name__ == "__main__":
    main()