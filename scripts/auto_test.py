"""
Auto-test the router on a dataset and print a leaderboard + save errors.

Usage examples:
  # Local function (fastest, uses GEMINI_API_KEY if set)
  PYTHONPATH=. python scripts/auto_test.py --data data/synth_dataset.csv --apps apps.yaml --limit 250

  # Against running API server
  python scripts/auto_test.py --server http://127.0.0.1:8000 --data data/synth_dataset.csv --limit 250
"""
from __future__ import annotations
import argparse, collections, csv, json, os, random
from pathlib import Path
import pandas as pd


def run_local(msg: str, apps_path: Path):
    from app.router import calibrated_decision
    return calibrated_decision(msg, apps_path)


def run_server(msg: str, server: str):
    import requests
    r = requests.post(f"{server.rstrip('/')}/chat", json={"message": msg}, timeout=30)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_dataset.csv")
    ap.add_argument("--apps", default="apps.yaml")
    ap.add_argument("--server", default=None, help="Base URL like http://127.0.0.1:8000")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--out", default="data/auto_test_errors.csv")
    ap.add_argument("--no-llm", action="store_true", help="Disable LLM arbiter for this run")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if args.no_llm:
        # Disable LLM arbiter for this process to avoid rate limits
        os.environ.pop("GEMINI_API_KEY", None)
    if args.limit and len(df) > args.limit:
        df = df.sample(args.limit, random_state=42).reset_index(drop=True)

    run = (lambda m: run_server(m, args.server)) if args.server else (lambda m: run_local(m, Path(args.apps)))

    route_counts = collections.Counter()
    errors = []
    confusion = collections.Counter()

    for _, row in df.iterrows():
        msg = str(row["text"])
        gold = str(row["label"])
        dec = run(msg)
        if dec.get("type") == "ROUTED":
            pred = dec.get("app_id")
            route_counts[pred] += 1
        else:
            pred = "OUT_OF_SCOPE"
            route_counts[pred] += 1
        if pred != gold:
            confusion[(gold, pred)] += 1
            errors.append({
                "text": msg,
                "gold": gold,
                "pred": pred,
                "type": dec.get("type"),
                "confidence": dec.get("confidence"),
                "message": dec.get("message"),
            })

    # Leaderboard
    print("Top predicted routes (count):")
    for route, cnt in route_counts.most_common():
        print(f"  {route:16s} {cnt}")

    print("\nTop confusions (gold -> pred, count):")
    for (g, p), cnt in confusion.most_common(10):
        print(f"  {g:16s} -> {p:16s} {cnt}")

    # Save errors CSV
    if errors:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(errors[0].keys()))
            w.writeheader()
            w.writerows(errors)
        print(f"\nSaved {len(errors)} errors to {args.out}")
    else:
        print("\nNo errors in the sampled set.")


if __name__ == "__main__":
    main()
