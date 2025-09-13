# Chatbot Router Demo — Runbook

This runbook summarizes what’s implemented, what works today, and how to start the backend and the minimal frontend.

## What’s Implemented
- Hybrid router: rules (precision) + embeddings (coverage) + optional Gemini arbiter (tie-break/structure). See `app/router.py` and `apps.yaml`.
- Caching for speed: in‑process caches for rules and embeddings in `app/router.py`.
- Dataset generator: large, adversarial synthetic data (typos, Hinglish, emojis, HTML junk, neg-keywords, OOS decoys like “leave-one-out”, “absorption costing”, “cost center codes”, etc.). `scripts/generate_synth_data.py`.
- Classifier over embeddings: calibrated multinomial logistic regression with saved model. `scripts/train_classifier.py` → `models/router.joblib`.
- Dedicated OOS head: calibrated binary classifier fused with multiclass OOS prob. `scripts/train_oos_head.py` → `models/oos_head.joblib`.
- Thresholding & calibration:
  - General thresholds (tau/margin) tuner with min OOS recall constraint. `scripts/tune_thresholds.py` → `config/router_thresholds.json`.
  - OOS PR-based threshold picker (knee or target precision). `scripts/oos_threshold_picker.py`.
  - Router reads thresholds from `config/router_thresholds.json`.
- Evaluation & analysis:
  - Offline eval with confusion matrix and class report. `scripts/eval_router.py`.
  - Auto-test leaderboard + misroutes CSV. `scripts/auto_test.py`.
  - Reliability (calibration) plot. `scripts/calibration_plot.py`.
  - Compare embedders under OOS recall constraint. `scripts/compare_embedders.py`.
  - Export confusion matrices per embedder + combined long CSV. `scripts/export_confusions.py`.
- Phase 2 stub: Action simulation for “mark absent” with basic entity/date parsing (ISO normalized). `app/actions.py`, wired in `app/main.py`.
- Minimal React chat component: `web/src/Chat.tsx`.

## Recommended Quick Start
1) Environment
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) Data
```bash
# Default (~thousands)
python scripts/generate_synth_data.py --out data/synth_dataset.csv

# Scale up (e.g., 20k+ rows):
python scripts/generate_synth_data.py --out data/synth_dataset.csv --min_rows 20000 --mult 2 --aug_min 3 --aug_max 6
```

3) Train (combined: multiclass + OOS head, with thresholds)
```bash
export EMBED_MODEL=sentence-transformers/all-mpnet-base-v2   # or intfloat/e5-base-v2, or sentence-transformers/all-MiniLM-L12-v2
PYTHONPATH=. python scripts/train_combo.py --data data/synth_dataset.csv \
  --min-oos-recall 0.85 --target-oos-precision 0.90
```
Artifacts:
- `models/router.joblib` (multiclass)
- `models/oos_head.joblib` (OOS)
- `config/router_thresholds.json` (tau, delta, oos_tau, oos_delta)
- `reports/train_combo_<embedder>.json`

4) Run the backend (with warm startup)
```bash
unset GEMINI_API_KEY   # keep disabled for local eval; set it to enable LLM arbiter
export EMBED_MODEL=sentence-transformers/all-mpnet-base-v2
PYTHONPATH=. uvicorn app.main:app --reload
# POST http://127.0.0.1:8000/chat with: {"message":"who was absent last week"}
```

Notes:
- On startup we preload catalog, rules and embeddings and reuse them per request.
- Confidence margin guard: set `ROUTER_MARGIN_GUARD` (default 0.08) to avoid routing when top-2 are close.

5) Quick checks
```bash
curl -s -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"who was absent last week"}'
curl -s -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"mark Manju absent today"}'
curl -s -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"leave-one-out cross validation"}'
```

## Frontend (Minimal)
- Use the provided React component `web/src/Chat.tsx` inside any Vite/Next React app.
- The component posts to `/chat` on the same origin; proxy your dev server to `http://127.0.0.1:8000` or serve both behind one domain.

Example (Vite) steps:
```bash
npm create vite@latest my-chat -- --template react-ts
cd my-chat && npm install
# Copy web/src/Chat.tsx into src/Chat.tsx and render it from App.tsx
# Add a proxy in vite.config.ts to forward /chat to http://127.0.0.1:8000
npm run dev
```

vite.config.ts proxy snippet:
```ts
server: { proxy: { '/chat': 'http://127.0.0.1:8000' } }
```

## Evaluation & Analysis
- Offline evaluation:
```bash
PYTHONPATH=. python scripts/eval_router.py --data data/synth_dataset.csv --apps apps.yaml
```

- Auto-test (leaderboard + errors CSV):
```bash
PYTHONPATH=. python scripts/auto_test.py --data data/synth_dataset.csv --apps apps.yaml --limit 500 --no-llm
```

- Calibration (reliability) plot:
```bash
PYTHONPATH=. python scripts/calibration_plot.py --data data/synth_dataset.csv --apps apps.yaml --out data/calibration.png
```

- OOS PR curve + threshold picking:
```bash
PYTHONPATH=. python scripts/oos_pr_curve.py --data data/synth_dataset.csv --out data/oos_pr.png
PYTHONPATH=. python scripts/oos_threshold_picker.py --data data/synth_dataset.csv --model models/oos_head.joblib --out config/router_thresholds.json --target-precision 0.90
```

- Compare embedders under min OOS recall:
```bash
PYTHONPATH=. python scripts/compare_embedders.py --data data/synth_dataset.csv \
  --models sentence-transformers/all-mpnet-base-v2 intfloat/e5-base-v2 sentence-transformers/all-MiniLM-L12-v2
```

- Export confusions per embedder:
```bash
PYTHONPATH=. python scripts/export_confusions.py --data data/synth_dataset.csv \
  --models sentence-transformers/all-mpnet-base-v2 intfloat/e5-base-v2 sentence-transformers/all-MiniLM-L12-v2
```

## Configuration & Tips
- Apps catalog: `apps.yaml` (add `keywords` and `neg_keywords` for precision; sample intents/examples improve embeddings).
- Thresholds: `config/router_thresholds.json` controls router gating; regenerated by training/tuning.
- Env vars:
  - `EMBED_MODEL`: choose embedder (e.g., `sentence-transformers/all-mpnet-base-v2`).
  - `GEMINI_API_KEY`: optional arbiter for tie-break/entities.
- Logs: `uvicorn` stdout in your terminal or `uvicorn.log` if started via nohup.
- Performance: first run downloads models; subsequent runs are cached. CPU works; GPU (if available) speeds up embedding.
- Known warning: urllib3 NotOpenSSL on macOS is harmless for this demo.
