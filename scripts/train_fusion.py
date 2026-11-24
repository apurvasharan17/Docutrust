
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn

sys.path.insert(0, "scripts")
from fuse import CONDITIONS, FEATURES, feature_value, new_model

OUT_DIR = "artifacts"
VARIANTS = {
    "fusion_model": FEATURES,
    "fusion_model_live": [f for f in FEATURES if f != "engine_agreement"],
}


def load_answered():
    rows = []
    for c in CONDITIONS:
        with open(f"data/records/{c}_records.jsonl") as f:
            rows += [json.loads(line) for line in f]
    return [r for r in rows if r["raw_value"] is not None]   # answered fields only


def main():
    rows = load_answered()
    y = np.array([1 if r["is_correct"] else 0 for r in rows])
    os.makedirs(OUT_DIR, exist_ok=True)

    for name, feats in VARIANTS.items():
        X = np.array([[feature_value(r, f) for f in feats] for r in rows])
        model = new_model().fit(X, y)
        joblib.dump(model, f"{OUT_DIR}/{name}.joblib")
        meta = {
            "features": feats,                   # the app must pass signals in this order
            "trained_on": "Qwen 2.5 7B, answered fields, 13 synthetic conditions",
            "conditions": CONDITIONS,
            "n_fields": len(rows),
            "accuracy_in_training_data": float(y.mean()),
            "sklearn_version": sklearn.__version__,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "weights": dict(zip(feats, model[-1].coef_[0].tolist())),
        }
        with open(f"{OUT_DIR}/{name}.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"{name}: {len(feats)} signals, {len(rows)} fields "
              f"-> {OUT_DIR}/{name}.joblib")


if __name__ == "__main__":
    main()