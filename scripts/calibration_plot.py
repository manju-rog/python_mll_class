"""
Reliability diagram (calibration plot) for the router.
Bins decisions by returned confidence and plots mean predicted vs. actual accuracy.

Usage:
  PYTHONPATH=. python scripts/calibration_plot.py --data data/synth_dataset.csv --apps apps.yaml --out data/calibration.png
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from app.router import calibrated_decision


def collect(csv_path: Path, apps_path: Path):
    df = pd.read_csv(csv_path)
    confs, correct = [], []
    for _, row in df.iterrows():
        msg = str(row["text"])
        label = str(row["label"])  # true label: app_id or OUT_OF_SCOPE
        dec = calibrated_decision(msg, apps_path)
        pred = dec["app_id"] if dec["type"] == "ROUTED" else "OUT_OF_SCOPE"
        c = float(dec.get("confidence", 0.0)) if dec["type"] == "ROUTED" else 1.0 - 1e-6
        confs.append(c)
        correct.append(1.0 if pred == label else 0.0)
    return np.array(confs), np.array(correct)


def reliability(confs: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confs, bins) - 1
    xs, ys, counts = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        xs.append(confs[mask].mean())
        ys.append(correct[mask].mean())
        counts.append(int(mask.sum()))
    return np.array(xs), np.array(ys), np.array(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_dataset.csv")
    ap.add_argument("--apps", default="apps.yaml")
    ap.add_argument("--out", default="data/calibration.png")
    ap.add_argument("--bins", type=int, default=12)
    args = ap.parse_args()

    confs, correct = collect(Path(args.data), Path(args.apps))
    xs, ys, counts = reliability(confs, correct, n_bins=args.bins)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k--", label="ideal")
    sizes = 20 + 80 * (counts / counts.max())
    plt.scatter(xs, ys, s=sizes, c="tab:blue", alpha=0.8, label="bins")
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Router Reliability Diagram")
    plt.legend()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f"saved calibration plot to {args.out}")


if __name__ == "__main__":
    main()

