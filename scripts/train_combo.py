"""
Train multiclass router + OOS head together for a given embedder, tune thresholds
(tau, delta) with min OOS recall constraint, set OOS threshold via PR (target precision
or knee), and write a single JSON report.

Artifacts:
- models/router.joblib, models/oos_head.joblib
- config/router_thresholds.json
- reports/train_combo_<embedder>.json

Usage:
  export EMBED_MODEL=sentence-transformers/all-mpnet-base-v2
  PYTHONPATH=. python scripts/train_combo.py --data data/synth_dataset.csv \
    --min-oos-recall 0.85 --target-oos-precision 0.9
"""
from __future__ import annotations
import argparse, json, os
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
            return float(best_thr), float(prec[np.where(thr == best_thr)[0][0] + 1]), float(best_rec)
    # knee by max F1
    best_thr, best_f1, best_pr, best_rc = 0.5, -1.0, 0.0, 0.0
    for i, t in enumerate(thr):
        pr = prec[i + 1]
        rc = rec[i + 1]
        f1 = (2 * pr * rc / (pr + rc)) if (pr + rc) else 0.0
        if f1 > best_f1:
            best_f1, best_thr, best_pr, best_rc = f1, t, pr, rc
    return float(best_thr), float(best_pr), float(best_rc)


def sanitize(name: str) -> str:
    return name.replace('/', '_').replace(':', '_')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_dataset.csv")
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-oos-recall", type=float, default=0.85)
    ap.add_argument("--target-oos-precision", type=float, default=0.90)
    ap.add_argument("--out-model", default="models/router.joblib")
    ap.add_argument("--out-oos", default="models/oos_head.joblib")
    ap.add_argument("--out-config", default="config/router_thresholds.json")
    ap.add_argument("--out-report", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X = df["text"].astype(str).tolist()
    y = df["label"].astype(str).tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)

    model_name = args.embed_model if args.embed_model else (os.getenv("EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2")
    featurizer = EmbeddingFeaturizer(model_name=model_name, normalize=True)

    # Multiclass pipeline (calibrated LR)
    base = LogisticRegression(max_iter=2000, multi_class='multinomial')
    clf_pipe = Pipeline([
        ("emb", featurizer),
        ("clf", CalibratedClassifierCV(estimator=base, cv=3, method='sigmoid')),
    ])
    clf_pipe.fit(X_train, y_train)
    proba = clf_pipe.predict_proba(X_test)
    preds = clf_pipe.predict(X_test)
    labels_used = sorted(set(y_test) | set(preds))
    acc = M.accuracy_score(y_test, preds)
    rep = M.classification_report(y_test, preds, labels=labels_used, digits=3)
    best = tune_thresholds(y_test, proba, list(clf_pipe.classes_), min_oos_recall=args.min_oos_recall)

    # OOS head (binary)
    y_bin = [1 if lab == "OUT_OF_SCOPE" else 0 for lab in y]
    y_train_bin = [1 if lab == "OUT_OF_SCOPE" else 0 for lab in y_train]
    y_test_bin = [1 if lab == "OUT_OF_SCOPE" else 0 for lab in y_test]
    base_bin = LogisticRegression(max_iter=2000, class_weight='balanced')
    oos_pipe = Pipeline([
        ("emb", featurizer),
        ("clf", CalibratedClassifierCV(estimator=base_bin, cv=3, method='sigmoid')),
    ])
    oos_pipe.fit(X_train, y_train_bin)
    p_oos = oos_pipe.predict_proba(X_test)[:, 1]
    roc = M.roc_auc_score(y_test_bin, p_oos)
    ap_score = M.average_precision_score(y_test_bin, p_oos)
    oos_tau, pr_at, rc_at = pick_oos_tau_from_pr(np.array(y_test_bin), p_oos, target_precision=args.target_oos_precision)

    # Save artifacts
    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": clf_pipe}, args.out_model)
    Path(args.out_oos).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": oos_pipe}, args.out_oos)

    cfg_path = Path(args.out_config)
    cfg = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    cfg.update({
        "tau": best["tau"],
        "delta": best["delta"],
        "oos_tau": float(oos_tau),
        "oos_delta": float(max(0.15, best["delta"]))
    })
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    # Report
    report = {
        "embed_model": model_name,
        "dataset_size": len(df),
        "test_size": len(y_test),
        "multiclass": {
            "raw_accuracy": float(acc),
            "classification_report": rep,
            "best_thresholds": best,
        },
        "oos_head": {
            "roc_auc": float(roc),
            "auprc": float(ap_score),
            "oos_tau": float(oos_tau),
            "precision_at_tau": float(pr_at),
            "recall_at_tau": float(rc_at),
        }
    }
    out_rep = args.out_report or f"reports/train_combo_{sanitize(model_name)}.json"
    Path(out_rep).parent.mkdir(parents=True, exist_ok=True)
    Path(out_rep).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

