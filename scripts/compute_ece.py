
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # save to file, no window
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from docutrust.metrics.ece import expected_calibration_error

CONDITIONS = ["clean",
              "blur_r1.0", "blur_r1.5", "blur_r2.0",
              "noise_s40", "noise_s42", "noise_s43",
              "jpeg_q10", "jpeg_q5", "jpeg_q1",
              "lowres_f0.55", "lowres_f0.35", "lowres_f0.3"]
LABELS = ["clean",
          "blur 1.0", "blur 1.5", "blur 2.0",
          "noise 40", "noise 42", "noise 43",
          "jpeg q10", "jpeg q5", "jpeg q1",
          "lowres 0.55", "lowres 0.35", "lowres 0.3"]
SIGNALS = ["logprob_geo", "logprob_mean", "logprob_min", "logprob_first",
           "margin_mean", "margin_min", "ocr_conf_min", "grounded",
           "validation_passed", "doc_ocr_conf", "engine_agreement",
           "self_consistency"]
PLOT_SIGNALS = SIGNALS[:4]  # the aggregation methods = the experimental variable
N_BINS = 10
OUT_DIR = "results"

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"


def load(condition):
    path = f"data/records/{condition}_records.jsonl"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def auroc(confs, correct):
    """P(a random correct field has higher confidence than a random wrong one).
    1.0 = perfect separation, 0.5 = coin flip, None = nothing to compare."""
    pos = [c for c, ok in zip(confs, correct) if ok]
    neg = [c for c, ok in zip(confs, correct) if not ok]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def reliability_bins(confs, correct, n_bins=N_BINS):
    """Per bin: avg confidence, accuracy, count. Same binning as ece.py."""
    bins = [[] for _ in range(n_bins)]
    for c, ok in zip(confs, correct):
        bins[min(int(c * n_bins), n_bins - 1)].append((c, ok))
    return [{"bin": i,
             "avg_conf": sum(c for c, _ in b) / len(b),
             "accuracy": sum(ok for _, ok in b) / len(b),
             "count": len(b)} for i, b in enumerate(bins) if b]


def analyse(records):
    labels = [r["label"] for r in records]
    out = {"n_fields": len(records),
           "accuracy": labels.count("CORRECT") / len(records),
           "wrong": labels.count("WRONG"),
           "miss": labels.count("MISS"),
           "signals": {}}
    for sig in SIGNALS:
        rows = [r for r in records if r.get(sig) is not None]
        if not rows:
            continue
        confs = [r[sig] for r in rows]
        correct = [r["is_correct"] for r in rows]
        out["signals"][sig] = {
            "n": len(rows),
            "avg_conf": sum(confs) / len(confs),
            "ece": expected_calibration_error(confs, correct, N_BINS),
            "auroc": auroc(confs, correct),
            "bins": reliability_bins(confs, correct),
        }
    return out


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_trends(summary, conds):
    x = list(range(len(conds)))
    xl = [LABELS[CONDITIONS.index(c)] for c in conds]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # (a) accuracy
    ax = axes[0]
    acc = [summary[c]["accuracy"] for c in conds]
    ax.bar(x, acc, width=0.55, color=COLORS[0])
    for i, a in enumerate(acc):
        ax.text(i, a + 0.02, f"{a:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_title("Extraction accuracy", color=INK, loc="left", fontsize=11)
    ax.set_ylim(0, 1.12)

    # (b) ECE and (c) AUROC, one line per aggregation method
    for ax, metric, title, loc in (
            (axes[1], "ece", "ECE  (lower = better calibrated)", "upper left"),
            (axes[2], "auroc", "AUROC  (higher = catches errors)", "lower left")):
        for sig, col, mk in zip(PLOT_SIGNALS, COLORS, MARKERS):
            ys = [summary[c]["signals"].get(sig, {}).get(metric) for c in conds]
            pts = [(i, y) for i, y in zip(x, ys) if y is not None]
            if not pts:
                continue
            ax.plot(*zip(*pts), color=col, marker=mk, markersize=7,
                    linewidth=2, label=sig.replace("logprob_", ""))
        ax.set_title(title, color=INK, loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=9, loc=loc)
    axes[1].set_ylim(bottom=0)
    axes[2].axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
    axes[2].text(x[-1] + 0.25, 0.47, "coin flip", fontsize=8,
                 color=MUTED, ha="right", va="top")
    axes[2].set_ylim(0, 1.05)

    for ax in axes:
        style(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(xl)
        ax.set_xlim(-0.5, len(x) - 0.2)
    fig.suptitle("Qwen2.5-7B on synthetic GST invoices: confidence under blur",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ece_vs_blur.png"), dpi=150)
    plt.close(fig)


def plot_reliability(summary, conds, sig="logprob_geo"):
    fig, axes = plt.subplots(1, len(conds), figsize=(4 * len(conds), 4.2),
                             sharey=True)
    axes = axes if len(conds) > 1 else [axes]
    for ax, c in zip(axes, conds):
        s = summary[c]["signals"].get(sig)
        ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1)
        if s:
            b = s["bins"]
            ax.bar([k["avg_conf"] for k in b], [k["accuracy"] for k in b],
                   width=0.08, color=COLORS[0], edgecolor="white", linewidth=2)
            for k in b:
                ax.text(k["avg_conf"], k["accuracy"] + 0.03, f"n={k['count']}",
                        ha="center", fontsize=8, color=MUTED)
            ax.set_title(f"{LABELS[CONDITIONS.index(c)]}   ECE = {s['ece']:.3f}",
                         color=INK, loc="left", fontsize=10)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.12)
        ax.set_xlabel("confidence", color=MUTED, fontsize=9)
        style(ax)
    axes[0].set_ylabel("accuracy", color=MUTED, fontsize=9)
    fig.suptitle(f"Reliability diagrams ({sig}). Dashed line = perfect calibration",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "reliability_by_condition.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = {}
    for c in CONDITIONS:
        recs = load(c)
        if recs is None:
            print(f"(skipping {c}: no records file yet)")
            continue
        summary[c] = analyse(recs)
    conds = list(summary)
    if not conds:
        sys.exit("no records found - run build_records.py first")

    rows = []
    for c in conds:
        s = summary[c]
        print(f"\n{c}: {s['n_fields']} fields, accuracy {s['accuracy']:.3f} "
              f"(wrong {s['wrong']}, miss {s['miss']})")
        print(f"  {'signal':<15}{'avg_conf':>9}{'ECE':>8}{'AUROC':>8}")
        for sig, v in s["signals"].items():
            au = "   -" if v["auroc"] is None else f"{v['auroc']:.3f}"
            print(f"  {sig:<15}{v['avg_conf']:>9.3f}{v['ece']:>8.3f}{au:>8}")
            rows.append([c, s["accuracy"], sig, v["avg_conf"], v["ece"], v["auroc"]])

    with open(os.path.join(OUT_DIR, "ece_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT_DIR, "ece_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "accuracy", "signal", "avg_conf", "ece", "auroc"])
        w.writerows(rows)

    plot_trends(summary, conds)
    plot_reliability(summary, conds)
    print("\nsaved results/: ece_summary.json, ece_table.csv, "
          "ece_vs_blur.png, reliability_by_condition.png")


if __name__ == "__main__":
    main()