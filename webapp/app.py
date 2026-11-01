"""
DocuTrust — Flask demo surface for the research pipeline.

This app has NO research logic of its own. It only:
  - reads files the pipeline already produced (results/, data/, artifacts/), and
  - calls functions from the docutrust package (src/docutrust/).

Pages
  /            Results   — the problem (raw logprob ECE / AUROC per condition) and the fix
                           (fusion: in-distribution, transfer to unseen degradations, 3 LLMs)
  /explorer    Explorer  — one document under one condition: image, OCR, LLM output,
                           per-field signals and fused confidence
  /analyze     Live      — upload a receipt, run OCR → Qwen (Ollama) → signals → fused confidence

Run from the repo root:
  python webapp/app.py        →  http://127.0.0.1:5000
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from flask import Flask, abort, flash, redirect, render_template, request, send_file, url_for

# --- make `import docutrust` work without depending on the current folder -----------
ROOT = Path(__file__).resolve().parent.parent          # .../Docutrust
sys.path.insert(0, str(ROOT / "src"))

from docutrust.degrade.blur import BLUR_LEVELS, blur, condition_name  # noqa: E402
from docutrust.fusion.score import fused_confidence                   # noqa: E402
from docutrust.normalize import label as label_field                  # noqa: E402
from docutrust.signals.logprob import FIELDS, field_signals, parse_values  # noqa: E402
from docutrust.signals.ocr import doc_ocr_conf, ocr_signals            # noqa: E402
from docutrust.signals.validation import validation_signals            # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"

# Every signal in results/ece_summary.json (same order as compute_ece.py) — Results page
SIGNALS = [
    ("logprob_geo", "Logprob · geo mean"),
    ("logprob_mean", "Logprob · mean"),
    ("logprob_min", "Logprob · min token"),
    ("logprob_first", "Logprob · first token"),
    ("margin_mean", "Margin · mean"),
    ("margin_min", "Margin · min"),
    ("ocr_conf_min", "OCR · min conf"),
    ("grounded", "OCR · grounded"),
    ("validation_passed", "Rules · validation"),
    ("doc_ocr_conf", "OCR · document"),
    ("engine_agreement", "OCR · 2 engines"),
    ("self_consistency", "LLM · self-consistency"),
]
# Columns in the per-field tables (Explorer, Live): fused first, raw logprob to compare
TABLE_SIGNALS = [
    ("fused", "Fused · confidence"),
    ("logprob_geo", "Logprob · geo mean"),
    ("ocr_conf_min", "OCR · min conf"),
    ("engine_agreement", "OCR · 2 engines"),
    ("validation_passed", "Rules · validation"),
    ("doc_ocr_conf", "OCR · document"),
]
HEADLINE = "logprob_geo"           # the raw confidence the project criticises
FUSED = "fused"                    # the confidence the project proposes
CONFIDENT = 0.90                   # raw logprob ≥ this but wrong = "confident but wrong"
FLAG = 0.50                        # fused below this = "check this field"
CONDITION_ORDER = ["clean",
                   "blur_r1.0", "blur_r1.5", "blur_r2.0",
                   "noise_s40", "noise_s42", "noise_s43",
                   "jpeg_q10", "jpeg_q5", "jpeg_q1",
                   "lowres_f0.55", "lowres_f0.35", "lowres_f0.3",
                   "real"]
FUSION_SCORES = [("raw_logprob", "Raw logprob"), ("platt", "Platt (logprob only)"),
                 ("fused", "Fused (all signals)")]
MODEL_NAMES = {"qwen25_7b": "Qwen 2.5 7B", "llama31_8b": "Llama 3.1 8B", "mistral_7b": "Mistral 7B"}

# Live page: Ollama location — local by default, Kaggle tunnel via env var
OLLAMA_URL = os.environ.get("DOCUTRUST_OLLAMA_URL", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("DOCUTRUST_MODEL", "qwen2.5:7b-instruct")
LIVE_FUSION = "fusion_model_live"  # 7 signals: EasyOCR is too heavy to run per upload
MAX_SIDE = 2000                    # phone photos are huge; same limit as scripts/prep_real.py

app = Flask(__name__)
app.secret_key = "docutrust-demo"
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024


# =====================================================================================
# helpers: reading pipeline outputs
# =====================================================================================
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def conditions():
    """Conditions that have a records file, in a fixed order."""
    found = {p.name[: -len("_records.jsonl")] for p in (DATA / "records").glob("*_records.jsonl")}
    found.discard("field")  # field_records.jsonl is an older file, not a condition
    return [c for c in CONDITION_ORDER if c in found] + sorted(found - set(CONDITION_ORDER))


def pretty(cond):
    if cond == "real":
        return "real receipt"
    for prefix, name in (("blur_r", "blur "), ("noise_s", "noise "),
                         ("jpeg_q", "jpeg q"), ("lowres_f", "lowres ")):
        if cond.startswith(prefix):
            return name + cond[len(prefix):]
    return cond


def image_path(cond, doc_id):
    if cond == "clean":
        return DATA / "synthetic" / f"{doc_id}.png"
    if cond == "real":
        return DATA / "real" / f"{doc_id}.png"
    return DATA / "degraded" / cond / f"{doc_id}.png"


def ocr_file(cond):
    return DATA / "ocr" / f"{cond}_ocr.jsonl"


def find_doc(path, doc_id):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if f'"{doc_id}"' in line:
                rec = json.loads(line)
                if rec.get("doc_id") == doc_id:
                    return rec
    return None


def oof_scores(cond):
    """Out-of-fold fused scores saved by scripts/fuse.py: {(doc_id, field): score}.
    Each one comes from a model that never saw that document."""
    path = RESULTS / "fusion_oof.jsonl"
    if not path.exists():
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if f'"{cond}"' not in line:
                continue
            d = json.loads(line)
            if d["condition"] == cond:
                out[(d["doc_id"], d["field_name"])] = d["fused"]
    return out


def is_confident_wrong(rec):
    c = rec.get(HEADLINE)
    return rec.get("label") != "CORRECT" and c is not None and c >= CONFIDENT


def is_flagged(rec):
    f = rec.get(FUSED)
    return f is not None and f < FLAG


@app.template_filter("conf")
def fmt_conf(v, digits=3):
    if v is None:
        return "—"
    return f"{float(v):.{digits}f}"


@app.context_processor
def inject():
    return {"SIGNALS": SIGNALS, "SIGNAL_NAMES": dict(SIGNALS + TABLE_SIGNALS),
            "TABLE_SIGNALS": TABLE_SIGNALS, "pretty": pretty, "HEADLINE": HEADLINE,
            "FUSED": FUSED, "CONFIDENT": CONFIDENT, "FLAG": FLAG}


# =====================================================================================
# Page 1 — Results
# =====================================================================================
@app.get("/")
def results():
    summary = read_json(RESULTS / "ece_summary.json")
    if not summary:
        return render_template("results.html", summary=None)
    conds = [c for c in CONDITION_ORDER if c in summary] + [c for c in summary if c not in CONDITION_ORDER]

    sig = request.args.get("signal", HEADLINE)
    if sig not in dict(SIGNALS):
        sig = HEADLINE

    # the problem, told on blur: clean vs heavy blur
    worst = "blur_r2.0" if "blur_r2.0" in summary else conds[-1]
    first, last = summary[conds[0]], summary[worst]
    headline = {
        "acc_from": first["accuracy"], "acc_to": last["accuracy"],
        "conf_from": first["signals"][HEADLINE]["avg_conf"], "conf_to": last["signals"][HEADLINE]["avg_conf"],
        "ece_from": first["signals"][HEADLINE]["ece"], "ece_to": last["signals"][HEADLINE]["ece"],
        "auroc_to": last["signals"][HEADLINE]["auroc"], "from": pretty(conds[0]), "to": pretty(worst),
    }
    fusion = read_json(RESULTS / "fusion_summary.json")
    models = read_json(RESULTS / "model_comparison.json")
    return render_template("results.html", summary=summary, conds=conds, sig=sig, headline=headline,
                           has_trend_png=(RESULTS / "ece_vs_blur.png").exists(),
                           fusion=fusion, models=models, FUSION_SCORES=FUSION_SCORES,
                           MODEL_NAMES=MODEL_NAMES, CONDITION_ORDER=CONDITION_ORDER)


@app.get("/results/<name>")
def results_file(name):
    if name not in {"ece_vs_blur.png", "reliability_by_condition.png", "reliability.png", "ece_table.csv"}:
        abort(404)
    p = RESULTS / name
    if not p.exists():
        abort(404)
    return send_file(p)


# =====================================================================================
# Page 2 — Explorer
# =====================================================================================
@app.get("/explorer")
def explorer():
    conds = conditions()
    if not conds:
        return render_template("explorer.html", conds=[], doc=None)
    cond = request.args.get("cond") or ("blur_r2.0" if "blur_r2.0" in conds else conds[-1])
    if cond not in conds:
        abort(404)

    records = read_jsonl(DATA / "records" / f"{cond}_records.jsonl")
    # honest scores: out-of-fold where they exist (synthetic data); otherwise the final
    # model (docutrust.fusion.score), e.g. the real receipt, which no model was trained on
    oof = oof_scores(cond)
    for r in records:
        key = (r["doc_id"], r["field_name"])
        r[FUSED] = oof[key] if key in oof else fused_confidence(r)
    fused_source = ("out-of-fold: each score comes from a model that never saw this document"
                    if oof else "final model: this data was not used to train it")
    doc_ids = sorted({r["doc_id"] for r in records})
    doc = request.args.get("doc") or doc_ids[0]
    if doc not in doc_ids:
        doc = doc_ids[0]

    rows = [r for r in records if r["doc_id"] == doc]
    rows.sort(key=lambda r: FIELDS.index(r["field_name"]) if r["field_name"] in FIELDS else 99)

    # the same fields under clean, for side-by-side comparison
    clean_rows = {}
    if cond not in ("clean", "real") and "clean" in conds:
        clean_rows = {r["field_name"]: r for r in read_jsonl(DATA / "records" / "clean_records.jsonl")
                      if r["doc_id"] == doc}

    ocr = find_doc(ocr_file(cond), doc)
    llm = find_doc(DATA / "llm" / f"{cond}_llm.jsonl", doc)

    # condition-level stats + the most confident mistakes (best demo material)
    n = len(records)
    n_correct = sum(r["label"] == "CORRECT" for r in records)
    wrongs = [r for r in records if r["label"] != "CORRECT"]
    n_cw = sum(is_confident_wrong(r) for r in records)
    wrong_values = [r for r in records if r["label"] == "WRONG" and r.get("raw_value") is not None]
    cw_values = [r for r in wrong_values if is_confident_wrong(r)]
    caught = sum(is_flagged(r) for r in cw_values)    # confident-but-wrong values fusion flags
    worst = sorted(wrong_values, key=lambda r: -(r.get(HEADLINE) or 0))[:12]
    miss_cw = sum(r["label"] == "MISS" and is_confident_wrong(r) for r in records)

    # per-doc summary for the doc picker
    per_doc = {}
    for r in records:
        d = per_doc.setdefault(r["doc_id"], [0, 0])
        d[0] += r["label"] == "CORRECT"
        d[1] += 1

    return render_template(
        "explorer.html", conds=conds, cond=cond, doc=doc, doc_ids=doc_ids, per_doc=per_doc,
        rows=rows, clean_rows=clean_rows, ocr=ocr, llm=llm,
        has_clean_img=image_path("clean", doc).exists(), has_img=image_path(cond, doc).exists(),
        stats={"n": n, "correct": n_correct, "wrong": len(wrongs), "cw": n_cw, "miss_cw": miss_cw,
               "cw_values": len(cw_values), "caught": caught},
        worst=worst, fused_source=fused_source,
    )


@app.get("/image/<cond>/<doc_id>.png")
def image(cond, doc_id):
    if cond not in conditions() or not doc_id.replace("_", "").isalnum():
        abort(404)
    p = image_path(cond, doc_id)
    if not p.exists():
        abort(404)
    return send_file(p, mimetype="image/png")


# =====================================================================================
# Page 3 — Live analyze (needs Ollama: local, or Kaggle via tunnel)
# =====================================================================================
def ollama_status():
    import requests
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=4)
        r.raise_for_status()
        names = [m.get("name", "") for m in r.json().get("models", [])]
        if MODEL in names or any(n.split(":")[0] == MODEL for n in names):
            return {"ok": True, "detail": f"{MODEL} ready at {OLLAMA_URL}"}
        return {"ok": False, "detail": f"Ollama reachable, but {MODEL} not pulled (has: {', '.join(names) or 'none'})"}
    except Exception:  # noqa: BLE001
        return {"ok": False, "detail": f"Ollama not reachable at {OLLAMA_URL}"}


def run_live(img, radius, gold):
    """One pass through the real pipeline, using the package's own functions."""
    import pytesseract
    from PIL import ImageOps
    from docutrust.llm import ollama_runner
    from docutrust.ocr.tesseract import ocr_image

    # point the runner at local Ollama or the Kaggle tunnel, without editing the package
    ollama_runner.OLLAMA_URL = OLLAMA_URL
    ollama_runner.MODEL = MODEL

    cond = "clean" if radius == 0 else condition_name(radius)
    img = ImageOps.exif_transpose(img).convert("RGB")   # phone photos store rotation in metadata
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    if radius:
        img = blur(img, radius)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name)
        tmp_path = tmp.name
    try:
        t0 = time.time()
        ocr = ocr_image(tmp_path)                    # docutrust.ocr.tesseract
        t_ocr = time.time() - t0
        t0 = time.time()
        out = ollama_runner.extract(ocr["text"])     # docutrust.llm.ollama_runner (incl. byte check)
        t_llm = time.time() - t0
    finally:
        os.unlink(tmp_path)

    values = parse_values(out["response"])           # docutrust.signals.logprob
    checks = validation_signals(values, ocr["text"]) # docutrust.signals.validation
    doc_conf = doc_ocr_conf(ocr["words"])            # docutrust.signals.ocr
    rows = []
    for f in FIELDS:
        sig = field_signals(out["response"], out["logprobs"], f) or {}
        row = {"field_name": f, "raw_value": values.get(f), **sig}
        row.update(ocr_signals(values.get(f), ocr["words"]))
        row["validation_passed"] = checks[f]
        row["doc_ocr_conf"] = doc_conf
        row["engine_agreement"] = None               # EasyOCR is not run live
        row[FUSED] = fused_confidence(row, LIVE_FUSION)
        if gold is not None:
            row["gold_value"] = gold.get(f)
            row["label"] = label_field(values.get(f), gold.get(f), f)   # docutrust.normalize
        rows.append(row)

    import base64, io
    buf = io.BytesIO()
    preview = img.copy()
    preview.thumbnail((700, 1000))
    preview.save(buf, format="PNG")
    return {
        "cond": cond, "rows": rows, "ocr_text": ocr["text"], "ocr_words": len(ocr["words"]),
        "response": out["response"], "n_tokens": len(out["logprobs"]),
        "tok_s": out["eval_count"] / (out["eval_duration"] / 1e9) if out.get("eval_duration") else None,
        "t_ocr": t_ocr, "t_llm": t_llm, "img_b64": base64.b64encode(buf.getvalue()).decode(),
        "tesseract": str(pytesseract.get_tesseract_version()),
        "n_flagged": sum(is_flagged(r) for r in rows),
    }


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    status = ollama_status()
    if request.method == "GET":
        return render_template("analyze.html", status=status, levels=BLUR_LEVELS, result=None,
                               model=MODEL, fields=FIELDS)

    from PIL import Image
    file = request.files.get("image")
    if not file or not file.filename:
        flash("Choose a receipt image first.")
        return redirect(url_for("analyze"))
    try:
        radius = float(request.form.get("radius", 0) or 0)
    except ValueError:
        radius = 0.0

    gold = None
    gold_raw = (request.form.get("gold") or "").strip()
    if gold_raw:
        try:
            gold = json.loads(gold_raw)
            assert isinstance(gold, dict)
        except Exception:  # noqa: BLE001
            flash('Gold must be a JSON object, e.g. {"total_amount": "2886.96"}')
            return redirect(url_for("analyze"))

    try:
        img = Image.open(file.stream)
        img.load()
        result = run_live(img, radius, gold)
    except Exception as e:  # noqa: BLE001
        flash(f"Pipeline failed: {e}  ({status['detail']})")
        return redirect(url_for("analyze"))

    return render_template("analyze.html", status=status, levels=BLUR_LEVELS, result=result,
                           model=MODEL, fields=FIELDS, gold=gold, radius=radius)


@app.get("/gold/<doc_id>.json")
def gold_json(doc_id):
    """Convenience for the live demo: the answer key of a synthetic doc."""
    p = DATA / "synthetic" / f"{doc_id}.json"
    if not doc_id.replace("_", "").isalnum() or not p.exists():
        abort(404)
    return send_file(p, mimetype="application/json")


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))