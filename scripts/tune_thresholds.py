"""
Tune tau and delta thresholds using an existing trained model and a labeled dataset.

Writes to config/router_thresholds.json

Usage:
  PYTHONPATH=. python scripts/tune_thresholds.py --data data/synth_dataset.csv --out config/router_thresholds.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn import metrics as M


def _oos_recall(y_true, y_pred) -> float:
    # Recall for the OUT_OF_SCOPE class: TP_OOS / (TP_OOS + FN_OOS)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "OUT_OF_SCOPE" and yp == "OUT_OF_SCOPE")
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "OUT_OF_SCOPE" and yp != "OUT_OF_SCOPE")
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def tune_thresholds(labels, probas, class_names, min_oos_recall: float = 0.85):
    y_true = np.array(labels)
    best = None
    best_score = (-1.0, -1.0)  # (macro_f1, accuracy)
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
            # Enforce OOS recall constraint
            oos_rec = _oos_recall(y_true, y_pred)
            if oos_rec < min_oos_recall:
                continue
            score = (macro_f1, acc)
            if score > best_score:
                best_score = score
                best = {"tau": float(tau), "delta": float(delta), "macro_f1": float(macro_f1), "accuracy": float(acc)}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_dataset.csv")
    ap.add_argument("--model", default="models/router.joblib")
    ap.add_argument("--out", default="config/router_thresholds.json")
    ap.add_argument("--min-oos-recall", type=float, default=0.85)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X = df["text"].astype(str).tolist()
    y = df["label"].astype(str).tolist()

    bundle = joblib.load(args.model)
    pipe = bundle["pipeline"]
    probas = pipe.predict_proba(X)
    class_names = list(pipe.classes_)
    best = tune_thresholds(y, probas, class_names, min_oos_recall=args.min_oos_recall)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        # Also set OOS gating aligned with chosen thresholds for simplicity
        json.dump({"tau": best["tau"], "delta": best["delta"], "oos_tau": max(0.7, best["tau"]), "oos_delta": max(0.15, best["delta"])}, f)
    print("Best thresholds:", best)
    print("Saved ->", args.out)


if __name__ == "__main__":
    main()
