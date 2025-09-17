"""
Pick OOS thresholds from the OOS head precision-recall curve.

Saves oos_tau (and optionally oos_delta passthrough) into config/router_thresholds.json,
preserving existing tau/delta from the same file.

Usage examples:
  # Knee (max F1 on PR curve)
  PYTHONPATH=. python scripts/oos_threshold_picker.py --data data/synth_dataset.csv \
    --model models/oos_head.joblib --out config/router_thresholds.json

  # Target precision >= 0.90, pick highest recall point
  PYTHONPATH=. python scripts/oos_threshold_picker.py --data data/synth_dataset.csv \
    --model models/oos_head.joblib --out config/router_thresholds.json --target-precision 0.90
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn import metrics as M


def pick_threshold_precision_target(y_true, p, target_precision: float) -> float:
    prec, rec, thr = M.precision_recall_curve(y_true, p)
    best_thr = None
    best_rec = -1.0
    # precision-recall curve returns thresholds for all but first point
    # Map thresholds to indices (skip first PR point which has no threshold)
    for i, t in enumerate(thr):
        # threshold t corresponds to prec[i+1], rec[i+1]
        pr = prec[i + 1]
        rc = rec[i + 1]
        if pr >= target_precision and rc > best_rec:
            best_rec = rc
            best_thr = t
    if best_thr is None:
        # Fall back to knee (max F1)
        return pick_threshold_knee(y_true, p)
    return float(best_thr)


def pick_threshold_knee(y_true, p) -> float:
    prec, rec, thr = M.precision_recall_curve(y_true, p)
    # Choose threshold maximizing F1 = 2PR/(P+R).
    # Use points with thresholds (skip first PR point)
    f1s = []
    for i, t in enumerate(thr):
        pr = prec[i + 1]
        rc = rec[i + 1]
        if (pr + rc) == 0:
            f1 = 0.0
        else:
            f1 = 2 * pr * rc / (pr + rc)
        f1s.append((f1, t))
    f1s.sort(key=lambda x: x[0], reverse=True)
    return float(f1s[0][1]) if f1s else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_dataset.csv")
    ap.add_argument("--model", default="models/oos_head.joblib")
    ap.add_argument("--out", default="config/router_thresholds.json")
    ap.add_argument("--target-precision", type=float, default=None)
    ap.add_argument("--oos-delta", type=float, default=None, help="Optional margin vs in-domain probs; if omitted, leaves as-is")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X = df["text"].astype(str).tolist()
    y = np.array([1 if lab == "OUT_OF_SCOPE" else 0 for lab in df["label"].astype(str).tolist()])

    bundle = joblib.load(args.model)
    pipe = bundle["pipeline"]
    p = pipe.predict_proba(X)[:, 1]

    if args.target_precision is not None:
        oos_tau = pick_threshold_precision_target(y, p, args.target_precision)
        mode = f"target_precision>={args.target_precision}"
    else:
        oos_tau = pick_threshold_knee(y, p)
        mode = "knee_max_f1"

    cfg_path = Path(args.out)
    cfg = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    cfg["oos_tau"] = float(oos_tau)
    if args.oos_delta is not None:
        cfg["oos_delta"] = float(args.oos_delta)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    print(f"Saved OOS thresholds -> {cfg_path}  ({mode}),  oos_tau={oos_tau:.3f}")


if __name__ == "__main__":
    main()

