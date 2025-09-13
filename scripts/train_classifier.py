"""
Train a multiclass router classifier over embeddings, calibrate, and tune thresholds.

Saves:
- Model: models/router.joblib
- Thresholds: config/router_thresholds.json

Usage:
  PYTHONPATH=. python scripts/train_classifier.py --data data/synth_dataset.csv --embed-model sentence-transformers/all-MiniLM-L6-v2 --calibrate
"""
from __future__ import annotations
import argparse, json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn import metrics as M
import joblib

from app.featurizer import EmbeddingFeaturizer


@dataclass
class TuneResult:
    tau: float
    delta: float
    macro_f1: float
    accuracy: float


def tune_thresholds(labels, probas, class_names):
    # labels: true labels (str), probas: ndarray (n_samples, n_classes)
    y_true = np.array(labels)
    n = len(class_names)
    best = None
    tau_grid = np.linspace(0.5, 0.9, 9)
    delta_grid = np.linspace(0.0, 0.2, 5)
    for tau in tau_grid:
        for delta in delta_grid:
            y_pred = []
            for p in probas:
                top = int(np.argmax(p))
                sorted_idx = np.argsort(-p)
                top2 = sorted_idx[1] if len(sorted_idx) > 1 else top
                margin = p[top] - p[top2] if top2 != top else p[top]
                if p[top] >= tau and margin >= delta:
                    y_pred.append(class_names[top])
                else:
                    y_pred.append("OUT_OF_SCOPE")
            macro_f1 = M.f1_score(y_true, y_pred, labels=sorted(set(y_true) | set(y_pred)), average='macro')
            acc = M.accuracy_score(y_true, y_pred)
            tr = TuneResult(float(tau), float(delta), float(macro_f1), float(acc))
            if best is None or tr.macro_f1 > best.macro_f1 or (tr.macro_f1 == best.macro_f1 and tr.accuracy > best.accuracy):
                best = tr
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_dataset.csv")
    ap.add_argument("--embed-model", default=None, help="SentenceTransformer model name. Defaults to EMBED_MODEL or MiniLM-L6.")
    ap.add_argument("--out-model", default="models/router.joblib")
    ap.add_argument("--out-config", default="config/router_thresholds.json")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X = df["text"].astype(str).tolist()
    y = df["label"].astype(str).tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)

    model_name = args.embed_model if args.embed_model else (os.getenv("EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2")
    featurizer = EmbeddingFeaturizer(model_name=model_name, normalize=True)
    base = LogisticRegression(max_iter=2000, multi_class='multinomial', n_jobs=None)

    pipe = Pipeline([
        ("emb", featurizer),
        ("clf", base),
    ])
    if args.calibrate:
        # Wrap in calibration on top of embeddings + base LR
        pipe = Pipeline([
            ("emb", featurizer),
            ("clf", CalibratedClassifierCV(estimator=base, cv=3, method='sigmoid')),
        ])

    print("Fitting classifier…")
    pipe.fit(X_train, y_train)
    probas = pipe.predict_proba(X_test)
    preds = pipe.predict(X_test)
    labels = sorted(set(y_test) | set(preds))
    print("Accuracy:", M.accuracy_score(y_test, preds))
    print("Classification report:\n", M.classification_report(y_test, preds, labels=labels))

    # Tune thresholds on test predictions to simulate deployment behavior
    class_names = list(pipe.classes_)
    best = tune_thresholds(y_test, probas, class_names)
    print(f"Best thresholds -> tau={best.tau:.2f}, delta={best.delta:.2f}, macro_f1={best.macro_f1:.3f}, acc={best.accuracy:.3f}")

    # Save artifacts
    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe}, args.out_model)
    Path(args.out_config).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_config, "w", encoding="utf-8") as f:
        json.dump({"tau": best.tau, "delta": best.delta}, f)
    print(f"Saved model -> {args.out_model}")
    print(f"Saved thresholds -> {args.out_config}")


if __name__ == "__main__":
    import os
    main()
