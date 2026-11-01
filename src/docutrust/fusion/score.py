"""Fused confidence for one extracted field, using a saved fusion model.

No training happens here: scripts/train_fusion.py fits the models and saves
them to artifacts/. This module only loads a model and scores fields, so the
Flask app (and anything else) can use the fusion without research code.

  fused_confidence(signals)                           full model, 8 signals
  fused_confidence(signals, model="fusion_model_live") no EasyOCR, 7 signals

`signals` is a dict like a records row: it needs "raw_value" and the signal
keys listed in the model's .json file. Missing signals count as 0.0 ("no
evidence"), exactly as in training.
"""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

# src/docutrust/fusion/score.py -> repo root is 3 folders up from this folder
ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "artifacts"


@lru_cache(maxsize=None)
def load_model(name="fusion_model"):
    """Load a saved model once; later calls reuse it."""
    model = joblib.load(ARTIFACTS / f"{name}.joblib")
    with open(ARTIFACTS / f"{name}.json") as f:
        meta = json.load(f)
    return model, meta["features"]


def _value(signals, name):
    v = signals.get(name)
    return 0.0 if v is None else float(v)


def fused_confidence(signals, model="fusion_model"):
    """Probability (0-1) that the extracted value is correct. None for null answers,
    because the fusion model covers answered fields only."""
    if signals.get("raw_value") is None:
        return None
    m, features = load_model(model)
    x = np.array([[_value(signals, f) for f in features]])
    return float(m.predict_proba(x)[0, 1])