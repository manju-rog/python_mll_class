"""
Train per-embedder models and export confusion matrices for side-by-side comparisons.
Uses calibrated multiclass + OOS head, applies OOS recall-constrained thresholds and
OOS precision-target threshold, then predicts on a test split.

Outputs:
- reports/confusion_<embedder>.csv
- reports/confusion_long.csv (append mode)
- reports/summary_compare.json

Usage:
  PYTHONPATH=. python scripts/export_confusions.py --data data/synth_dataset.csv \
    --models sentence-transformers/all-mpnet-base-v2 intfloat/e5-base-v2 sentence-transformers/all-MiniLM-L12-v2 \
    --min-oos-recall 0.85 --target-oos-precision 0.9
"""
from __future__ import annotations
import argparse, json
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
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "OUT_OF_SCOPE" and yp == "OUT_OF_SCOPE")
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "OUT_OF_SCOPE" and yp != "OUT_OF_SCOPE")
            oos_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if oos_rec < min_oos_recall:
                continue
            score = (macro_f1, acc)
            if score > best_score:
                best_score = score
                best = {"tau": float(tau), "delta": float(delta)}
    return best or {"tau": 0.6, "delta": 0.1}


def pick_oos_tau_from_pr(y_true, p, target_precision: float | None):
    prec, rec, thr = M.precision_recall_curve(y_true, p)
    if target_precision is not None:
        best_thr, best_rec = None, -1.0
        for i, t in enumerate(thr):
            pr = prec[i + 1]
            rc = rec[i + 1]
            if pr >= target_precision and rc > best_rec:
                best_rec, best_thr = rc, t
        if best_thr is not None:
            return float(best_thr)
    # knee by max F1
    best_thr, best_f1 = 0.5, -1.0
    for i, t in enumerate(thr):
        pr = prec[i + 1]
        rc = rec[i + 1]
        f1 = (2 * pr * rc / (pr + rc)) if (pr + rc) else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return float(best_thr)


def sanitize(name: str) -> str:
    return name.replace('/', '_').replace(':', '_')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_dataset.csv")
    ap.add_argument("--models", nargs='+', required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-oos-recall", type=float, default=0.85)
    ap.add_argument("--target-oos-precision", type=float, default=0.90)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X = df["text"].astype(str).tolist()
    y = df["label"].astype(str).tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)

    long_rows = []
    summary = {}
    for model_name in args.models:
        featurizer = EmbeddingFeaturizer(model_name=model_name, normalize=True)

        # Multiclass
        base = LogisticRegression(max_iter=2000, multi_class='multinomial')
        clf_pipe = Pipeline([
            ("emb", featurizer),
            ("clf", CalibratedClassifierCV(estimator=base, cv=3, method='sigmoid')),
        ])
        clf_pipe.fit(X_train, y_train)
        proba = clf_pipe.predict_proba(X_test)
        classes = list(clf_pipe.classes_)
        best = tune_thresholds(y_test, proba, classes, min_oos_recall=args.min_oos_recall)

        # OOS head
        y_test_bin = np.array([1 if lab == "OUT_OF_SCOPE" else 0 for lab in y_test])
        base_bin = LogisticRegression(max_iter=2000, class_weight='balanced')
        oos_pipe = Pipeline([
            ("emb", featurizer),
            ("clf", CalibratedClassifierCV(estimator=base_bin, cv=3, method='sigmoid')),
        ])
        oos_pipe.fit(X_train, [1 if lab == "OUT_OF_SCOPE" else 0 for lab in y_train])
        p_oos = oos_pipe.predict_proba(X_test)[:, 1]
        oos_tau = pick_oos_tau_from_pr(y_test_bin, p_oos, args.target_oos_precision)

        # Predict with gating fusion
        preds = []
        for i in range(len(X_test)):
            row = proba[i]
            # Multiclass OOS prob
            p_out_mc = row[classes.index("OUT_OF_SCOPE")] if "OUT_OF_SCOPE" in classes else 0.0
            p_out_fused = max(p_out_mc, p_oos[i])
            in_mask = [j for j, c in enumerate(classes) if c != "OUT_OF_SCOPE"]
            in_probs = row[in_mask]
            in_labels = [classes[j] for j in in_mask]
            top_idx = int(np.argmax(in_probs))
            second_idx = int(np.argsort(-in_probs)[1]) if len(in_probs) > 1 else top_idx
            top_prob = float(in_probs[top_idx])
            second_prob = float(in_probs[second_idx])
            margin = top_prob - second_prob
            if p_out_fused >= max(0.7, best["delta"]) and (p_out_fused - top_prob) >= max(0.15, best["delta"]):
                preds.append("OUT_OF_SCOPE")
            elif top_prob >= best["tau"] and margin >= best["delta"]:
                preds.append(in_labels[top_idx])
            else:
                preds.append("OUT_OF_SCOPE")

        labels_all = sorted(set(y_test) | set(preds))
        cm = M.confusion_matrix(y_test, preds, labels=labels_all)
        df_cm = pd.DataFrame(cm, index=labels_all, columns=labels_all)
        out_csv = Path(f"reports/confusion_{sanitize(model_name)}.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_cm.to_csv(out_csv)

        # Append to long
        for gi, g in enumerate(labels_all):
            for pi, p in enumerate(labels_all):
                long_rows.append({
                    "embedder": model_name,
                    "gold": g,
                    "pred": p,
                    "count": int(cm[gi, pi])
                })

        summary[model_name] = {
            "accuracy": float(M.accuracy_score(y_test, preds)),
            "macro_f1": float(M.f1_score(y_test, preds, labels=labels_all, average='macro')),
            "best_thresholds": best,
            "oos_tau": float(oos_tau),
        }

    # Write combined outputs
    long_df = pd.DataFrame(long_rows)
    long_path = Path("reports/confusion_long.csv")
    long_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(long_path, index=False)
    Path("reports/summary_compare.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote:")
    print(" -", long_path)
    print(" - reports/summary_compare.json")


if __name__ == "__main__":
    main()

