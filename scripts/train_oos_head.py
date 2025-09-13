"""
Train a dedicated OOS binary classifier (positive = OUT_OF_SCOPE), calibrated.

Saves model to models/oos_head.joblib and prints ROC AUC and AUPRC.

Usage:
  PYTHONPATH=. python scripts/train_oos_head.py --data data/synth_dataset.csv --embed-model sentence-transformers/all-mpnet-base-v2 --calibrate
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn import metrics as M
import joblib

from app.featurizer import EmbeddingFeaturizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_dataset.csv")
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--out", default="models/oos_head.joblib")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X = df["text"].astype(str).tolist()
    y = [1 if lab == "OUT_OF_SCOPE" else 0 for lab in df["label"].astype(str).tolist()]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)

    import os
    model_name = args.embed_model if args.embed_model else (os.getenv("EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2")
    featurizer = EmbeddingFeaturizer(model_name=model_name, normalize=True)
    base = LogisticRegression(max_iter=2000, class_weight='balanced')

    if args.calibrate:
        pipe = Pipeline([
            ("emb", featurizer),
            ("clf", CalibratedClassifierCV(estimator=base, cv=3, method='sigmoid')),
        ])
    else:
        pipe = Pipeline([
            ("emb", featurizer),
            ("clf", base),
        ])

    print("Fitting OOS head…")
    pipe.fit(X_train, y_train)

    p = pipe.predict_proba(X_test)[:, 1]
    roc = M.roc_auc_score(y_test, p)
    prec, rec, _ = M.precision_recall_curve(y_test, p)
    ap_score = M.average_precision_score(y_test, p)
    print(f"OOS ROC-AUC: {roc:.4f}  |  OOS AUPRC: {ap_score:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe}, args.out)
    print(f"Saved OOS head -> {args.out}")


if __name__ == "__main__":
    main()

