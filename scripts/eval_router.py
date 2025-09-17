# Offline evaluation of the router using dataset csv. LLM arbiter is optional.
import pandas as pd, argparse, json, os, sys
from pathlib import Path
from app.router import calibrated_decision

def evaluate(csv_path: Path, apps_path: Path):
    df = pd.read_csv(csv_path)
    y_true, y_pred = [], []
    for _, row in df.iterrows():
        msg = str(row["text"])
        label = str(row["label"])
        dec = calibrated_decision(msg, apps_path)
        if dec["type"] == "ROUTED":
            pred = dec["app_id"]
        else:
            pred = "OUT_OF_SCOPE"
        y_true.append(label)
        y_pred.append(pred)

    # metrics
    import sklearn.metrics as m
    labels = sorted(set(y_true + y_pred))
    print("Labels:", labels)
    print("Accuracy:", m.accuracy_score(y_true, y_pred))
    print("Confusion matrix:\n", m.confusion_matrix(y_true, y_pred, labels=labels))
    print("Classification report:\n", m.classification_report(y_true, y_pred, labels=labels))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_dataset.csv")
    ap.add_argument("--apps", default="apps.yaml")
    args = ap.parse_args()
    evaluate(Path(args.data), Path(args.apps))
