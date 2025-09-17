"""
Plot OOS precision-recall curve using the trained OOS head.

Usage:
  PYTHONPATH=. python scripts/oos_pr_curve.py --data data/synth_dataset.csv --out data/oos_pr.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn import metrics as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_dataset.csv")
    ap.add_argument("--model", default="models/oos_head.joblib")
    ap.add_argument("--out", default="data/oos_pr.png")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X = df["text"].astype(str).tolist()
    y = np.array([1 if lab == "OUT_OF_SCOPE" else 0 for lab in df["label"].astype(str).tolist()])

    bundle = joblib.load(args.model)
    pipe = bundle["pipeline"]
    p = pipe.predict_proba(X)[:, 1]

    prec, rec, thr = M.precision_recall_curve(y, p)
    ap_score = M.average_precision_score(y, p)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5))
    plt.plot(rec, prec, label=f"OOS PR (AP={ap_score:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("OOS Precision-Recall Curve")
    plt.legend()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f"Saved OOS PR curve -> {args.out}")


if __name__ == "__main__":
    main()

