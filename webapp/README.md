# DocuTrust — web demo

A thin Flask layer over the research pipeline. It has no research logic of its own:
it reads what the pipeline already produced (`results/`, `data/`) and calls functions
from `src/docutrust/`.

| Page | What it shows | Needs Kaggle? |
|---|---|---|
| **Results** `/` | Accuracy, ECE, AUROC per blur level and signal; reliability diagrams | No |
| **Explorer** `/explorer` | One document under one condition: clean vs blurred image, Tesseract text, Qwen output, per-field signals and labels; most confident mistakes | No |
| **Live** `/analyze` | Upload a receipt → blur → OCR → Qwen → signals → labels, live | Yes (Ollama) |

## Run (Mac)

From the `Docutrust` folder:

```bash
source .venv/bin/activate
pip install -r webapp/requirements.txt
python webapp/app.py
```

Open http://127.0.0.1:5000. Results and Explorer work immediately.

## Live page with Kaggle

1. Kaggle notebook: run cells 1–8 as usual (install, `ollama serve`, pull `qwen2.5:7b-instruct`).
   Turn **Internet ON** in notebook settings.
2. Add a new cell after cell 8, paste `webapp/kaggle_tunnel.py`, run it. It prints a
   `https://….trycloudflare.com` URL.
3. Mac:
   ```bash
   export DOCUTRUST_OLLAMA_URL=https://<printed-url>.trycloudflare.com
   python webapp/app.py
   ```
   The status pill on the Live page turns green.
4. Demo: upload `data/synthetic/doc_001.png`, blur 2.0, paste the gold from `/gold/doc_001.json`.

Keep the Kaggle session alive during the demo; a restart gives a new URL. Stop the tunnel
afterwards (`tunnel.terminate()`), since the URL is public while it runs.

Optional: `DOCUTRUST_MODEL` (default `qwen2.5:7b-instruct`), `PORT` (default 5000).

## Which project code each page uses

- Live: `degrade.blur.blur`, `ocr.tesseract.ocr_image`, `llm.ollama_runner.extract`
  (including the byte-reconstruction check), `signals.logprob.field_signals` / `parse_values`,
  `normalize.label`. The app only sets `ollama_runner.OLLAMA_URL` / `MODEL` at runtime; the
  package is not modified.
- Results: `results/ece_summary.json` and `results/ece_vs_blur.png` from `scripts/compute_ece.py`.
- Explorer: `data/records`, `data/llm`, `data/ocr`, `data/synthetic`, `data/degraded`.
