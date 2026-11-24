
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "scripts")
from compute_ece import expected_calibration_error, auroc, N_BINS

FAMILIES = {
    "blur":   ["blur_r1.0", "blur_r1.5", "blur_r2.0"],
    "noise":  ["noise_s40", "noise_s42", "noise_s43"],
    "jpeg":   ["jpeg_q10", "jpeg_q5", "jpeg_q1"],
    "lowres": ["lowres_f0.55", "lowres_f0.35", "lowres_f0.3"],
}
CONDITIONS = ["clean"] + [c for fam in FAMILIES.values() for c in fam]
TRANSFER_TRAIN = ["clean"] + FAMILIES["blur"]
TRANSFER_TEST = FAMILIES["noise"] + FAMILIES["jpeg"] + FAMILIES["lowres"]

FEATURES = ["logprob_geo", "logprob_min", "margin_min", "ocr_conf_min",
            "engine_agreement", "validation_passed", "doc_ocr_conf", "grounded"]
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


def oof_predict(X, y, groups, train_mask, test_mask):
    """Out-of-fold scores. Folds are split by doc_id; within each fold the model
    trains only on rows in train_mask and scores only rows in test_mask."""
    pred = np.full(len(y), np.nan)
    for train, test in GroupKFold(n_splits=FOLDS).split(X, y, groups):
        train = train[train_mask[train]]
        test = test[test_mask[test]]
        if len(test) == 0:
            continue
        model = new_model().fit(X[train], y[train])
        pred[test] = model.predict_proba(X[test])[:, 1]
    return pred


def safe_auroc(scores, labels):
    if len(set(labels)) < 2:
        return None          # e.g. clean: every answered field is correct
    return auroc(scores, labels)


def evaluate(title, scores, y, cond, conds):
    """Print and return ECE / AUROC per condition (+ all listed conditions together)."""
    print(f"=== {title} ===")
    out = {}
    for c in conds + ["all"]:
        mask = np.isin(cond, conds) if c == "all" else cond == c
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
        out[c] = res
    print()
    return out


def main():
    rows = load_answered()
    X = np.array([[feature_value(r, f) for f in FEATURES] for r in rows])
    y = np.array([1 if r["is_correct"] else 0 for r in rows])
    groups = np.array([r["doc_id"] for r in rows])
    cond = np.array([r["condition"] for r in rows])
    lp = [FEATURES.index("logprob_geo")]
    everything = np.ones(len(y), dtype=bool)

    print(f"answered fields: {len(rows)}   features: {FEATURES}\n")
    summary = {"features": FEATURES, "n_answered": len(rows), "folds": FOLDS}

    # A) in-distribution: every condition can appear in training (other docs)
    scores_a = {
        "raw_logprob": X[:, lp[0]],
        "platt": oof_predict(X[:, lp], y, groups, everything, everything),
        "fused": oof_predict(X, y, groups, everything, everything),
    }
    summary["in_distribution"] = evaluate(
        "A) in-distribution: 5-fold CV over all 13 conditions",
        scores_a, y, cond, CONDITIONS)

    # save every field's out-of-fold fused score: the Explorer shows these, so each
    # number in the UI comes from a model that never saw that document
    with open("results/fusion_oof.jsonl", "w") as f:
        for r, s in zip(rows, scores_a["fused"]):
            f.write(json.dumps({"condition": r["condition"], "doc_id": r["doc_id"],
                                "field_name": r["field_name"], "fused": float(s)}) + "\n")

    # B) transfer: train on clean + blur only, test on unseen degradation types
    train_mask = np.isin(cond, TRANSFER_TRAIN)
    test_mask = np.isin(cond, TRANSFER_TEST)
    scores_b = {
        "raw_logprob": X[:, lp[0]],
        "platt": oof_predict(X[:, lp], y, groups, train_mask, test_mask),
        "fused": oof_predict(X, y, groups, train_mask, test_mask),
    }
    summary["transfer"] = evaluate(
        "B) transfer: trained on clean + blur, tested on noise / JPEG / low-res",
        scores_b, y, cond, TRANSFER_TEST)

    # which signals does the fused model lean on? (fit once on all data)
    full = new_model().fit(X, y)
    weights = dict(zip(FEATURES, full[-1].coef_[0].tolist()))
    print("signal weights (standardised, + means 'more likely correct'):")
    for f, w in sorted(weights.items(), key=lambda t: -abs(t[1])):
        print(f"   {f:18s} {w:+.3f}")
    summary["weights"] = weights

    with open("results/fusion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nsaved results/fusion_summary.json")


if __name__ == "__main__":
    main()