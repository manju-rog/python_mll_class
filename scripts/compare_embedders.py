"""
Compare multiple embed models under min OOS recall constraint using the same pipeline.

Usage:
  PYTHONPATH=. python scripts/compare_embedders.py --data data/synth_dataset.csv --models \
    sentence-transformers/all-mpnet-base-v2 intfloat/e5-base-v2 sentence-transformers/all-MiniLM-L12-v2
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn import metrics as M

from app.featurizer import EmbeddingFeaturizer


def tune_thresholds(labels, probas, class_names, min_oos_recall: float = 0.85):
    y_true = np.array(labels)
    best = None
    best_score = (-1.0, -1.0)
    tau_grid = np.linspace(0.5, 0.95, 19)
    delta_grid = np.linspace(0.0, 0.3, 16)
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
            labels_used = sorted(set(y_true) | set(y_pred))
            macro_f1 = M.f1_score(y_true, y_pred, labels=labels_used, average='macro')
            acc = M.accuracy_score(y_true, y_pred)
            # OOS recall constraint
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "OUT_OF_SCOPE" and yp == "OUT_OF_SCOPE")
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "OUT_OF_SCOPE" and yp != "OUT_OF_SCOPE")
            oos_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if oos_rec < min_oos_recall:
                continue
            score = (macro_f1, acc)
            if score > best_score:
                best_score = score
                best = {"tau": float(tau), "delta": float(delta), "macro_f1": float(macro_f1), "accuracy": float(acc), "oos_recall": float(oos_rec)}
    return best


def run_for_model(model_name: str, X_train, X_test, y_train, y_test):
    featurizer = EmbeddingFeaturizer(model_name=model_name, normalize=True)
    base = LogisticRegression(max_iter=2000, multi_class='multinomial')
    pipe = Pipeline([
        ("emb", featurizer),
        ("clf", CalibratedClassifierCV(estimator=base, cv=3, method='sigmoid')),
    ])
    pipe.fit(X_train, y_train)
    probas = pipe.predict_proba(X_test)
    preds = pipe.predict(X_test)
    labels_used = sorted(set(y_test) | set(preds))
    acc = M.accuracy_score(y_test, preds)
    rep = M.classification_report(y_test, preds, labels=labels_used, digits=3)
    best = tune_thresholds(y_test, probas, list(pipe.classes_))
    return acc, rep, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_dataset.csv")
    ap.add_argument("--models", nargs='+', default=[
        'sentence-transformers/all-mpnet-base-v2',
        'intfloat/e5-base-v2',
        'sentence-transformers/all-MiniLM-L12-v2'
    ])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X = df["text"].astype(str).tolist()
    y = df["label"].astype(str).tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)

    print("Comparing models with OOS recall >= 0.85 constraint:\n")
    for m in args.models:
        print(f"=== {m} ===")
        acc, rep, best = run_for_model(m, X_train, X_test, y_train, y_test)
        print(f"Accuracy (raw preds): {acc:.4f}")
        print("Best thresholds:", best)
        print(rep)


if __name__ == "__main__":
    main()

