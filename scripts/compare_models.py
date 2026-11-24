
import json
import sys

import numpy as np

sys.path.insert(0, "scripts")
from compute_ece import expected_calibration_error, N_BINS
from fuse import CONDITIONS as QWEN_CONDITIONS
from fuse import FEATURES, feature_value, oof_predict, safe_auroc

SHARED = ["clean", "blur_r2.0", "noise_s42", "jpeg_q1", "lowres_f0.35"]
MODELS = {
    "qwen25_7b":  "data/records",
    "llama31_8b": "data/records_models/llama31_8b",
    "mistral_7b": "data/records_models/mistral_7b",
}
HEADLINE = "qwen25_7b"


def load(model, conds):
    rows = []
    for c in conds:
        with open(f"{MODELS[model]}/{c}_records.jsonl") as f:
            for line in f:
                r = json.loads(line)
                r["_model"] = model
                rows.append(r)
    return rows


def ece_auroc(confs, labels):
    confs = [float(c) for c in confs]
    labels = [bool(v) for v in labels]
    return expected_calibration_error(confs, labels, N_BINS), safe_auroc(confs, labels)


def fmt(ece, au):
    return f"{ece:.3f} / " + (f"{au:.3f}" if au is not None else "  -  ")


def part1_raw_all_fields(summary):
    print("=== 1) raw logprob_geo on ALL fields (nulls included) ===")
    print(f"{'condition':14s}" + "".join(f"{m:>28s}" for m in MODELS))
    print(f"{'':14s}" + "".join(f"{'acc  conf   ECE':>28s}" for _ in MODELS))
    out = {}
    for c in SHARED:
        line = f"{c:14s}"
        for m in MODELS:
            rows = [r for r in load(m, [c]) if r.get("logprob_geo") is not None]
            confs = [r["logprob_geo"] for r in rows]
            labels = [r["is_correct"] for r in rows]
            ece, _ = ece_auroc(confs, labels)
            acc = float(np.mean(labels))
            out.setdefault(m, {})[c] = {"n": len(rows), "accuracy": acc,
                                        "avg_conf": float(np.mean(confs)), "ece": ece}
            line += f"{acc:>14.3f}{np.mean(confs):>7.3f}{ece:>7.3f}"
        print(line)
    print("(fields with no logprob signal, e.g. empty answers, are left out)\n")
    summary["raw_all_fields"] = out


def part2_and_3(summary):
    rows = load(HEADLINE, QWEN_CONDITIONS)                  # Qwen: all 13 conditions
    for m in MODELS:
        if m != HEADLINE:
            rows += load(m, SHARED)
    rows = [r for r in rows if r["raw_value"] is not None]  # answered fields only

    X = np.array([[feature_value(r, f) for f in FEATURES] for r in rows])
    y = np.array([1 if r["is_correct"] else 0 for r in rows])
    groups = np.array([r["doc_id"] for r in rows])
    cond = np.array([r["condition"] for r in rows])
    model = np.array([r["_model"] for r in rows])
    lp = [FEATURES.index("logprob_geo")]
    shared = np.isin(cond, SHARED)

    print("=== 2) answered fields: ECE / AUROC, each model trained on its own data ===")
    print("=== 3) transfer: fusion trained on Qwen only, applied to the other models ===")
    out = {}
    for m in MODELS:
        own = (model == m) & shared
        scores = {
            "raw_logprob": X[:, lp[0]],
            "platt": oof_predict(X[:, lp], y, groups, own, own),
            "fused": oof_predict(X, y, groups, own, own),
        }
        if m != HEADLINE:
            from_qwen = model == HEADLINE
            scores["fused_from_qwen"] = oof_predict(X, y, groups, from_qwen, own)

        print(f"\n{m}")
        print(f"   {'condition':14s}{'n':>5s}{'acc':>7s}" +
              "".join(f"{name:>20s}" for name in scores))
        out[m] = {}
        for c in SHARED + ["all"]:
            mask = own if c == "all" else own & (cond == c)
            res = {"n": int(mask.sum()), "accuracy": float(y[mask].mean())}
            line = f"   {c:14s}{res['n']:>5d}{res['accuracy']:>7.3f}"
            for name, s in scores.items():
                ece, au = ece_auroc(s[mask], y[mask])
                res[name] = {"ece": ece, "auroc": au}
                line += f"{fmt(ece, au):>20s}"
            out[m][c] = res
            print(line)
    summary["answered_fields"] = out


def main():
    summary = {"shared_conditions": SHARED, "features": FEATURES}
    part1_raw_all_fields(summary)
    part2_and_3(summary)
    with open("results/model_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nsaved results/model_comparison.json")


if __name__ == "__main__":
    main()