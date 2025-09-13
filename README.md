# Chatbot Router Demo (Local, No Docker)

A ruthless-accuracy **hybrid router** for a business chatbot that cleanly routes to:
- Absence Management
- TDO Drafting
- Cost Estimation/Predictions

## 1) Setup (local)
```bash
cd chatbot-router-demo
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
# Optional (for LLM arbitration):
export GEMINI_API_KEY=your_key_here
```

## 2) Generate BIG synthetic dataset (thousands, no LLM)
```bash
python scripts/generate_synth_data.py --out data/synth_dataset.csv
```

## 3) Train + Calibrate (new, optional but recommended)
```bash
# Default embedder (fast): sentence-transformers/all-MiniLM-L6-v2
PYTHONPATH=. python scripts/train_classifier.py --data data/synth_dataset.csv --calibrate

# Optionally try a larger embedder for even higher accuracy (downloads ~1GB):
# export EMBED_MODEL=intfloat/e5-large-v2
# PYTHONPATH=. python scripts/train_classifier.py --data data/synth_dataset.csv --calibrate
```

## 4) Evaluate routing quality (offline)
```bash
python scripts/eval_router.py --data data/synth_dataset.csv --apps apps.yaml
```

## 5) Run the local API
```bash
uvicorn app.main:app --reload
# POST http://127.0.0.1:8000/chat  with JSON: {"message": "who was absent last week"}
```

## 6) Auto-test (leaderboard + errors)
```bash
# Local function path (fastest). Uses GEMINI_API_KEY if set; add --no-llm to avoid rate limits.
PYTHONPATH=. python scripts/auto_test.py --data data/synth_dataset.csv --apps apps.yaml --limit 250 --no-llm

# Against running server
python scripts/auto_test.py --server http://127.0.0.1:8000 --data data/synth_dataset.csv --limit 250
```
Outputs a routes leaderboard and saves misroutes to `data/auto_test_errors.csv`.

## 7) Calibration plot
```bash
PYTHONPATH=. python scripts/calibration_plot.py --data data/synth_dataset.csv --apps apps.yaml --out data/calibration.png
```

## 5) Frontend
The repo includes a minimal React `Chat.tsx` component (use in any Vite/Next app). For a quick check, hit the API with curl or Postman.

## How accuracy stays high
- Deterministic regex rules for precision on high-signal words
- Embedding similarity (configurable model via `EMBED_MODEL`) for broad coverage
- Trained classifier over embeddings with calibrated probabilities
- (Optional) Gemini 2.5 Flash arbiter for tie-break + structure
- Tuned thresholds + margin gating: only route when confident; otherwise friendly capability-bounded reply
- Ruthless synthetic data with typos, Hinglish code-switching, emojis, boilerplate noise, and HTML junk to harden OOS detection

## Notes
- Phase 2 (actions) can map intents to internal APIs with confirmation & authz checks.
- Everything is local-first. No Docker required.
 - Use `EMBED_MODEL` to switch to a stronger embedder (e.g., `intfloat/e5-large-v2`) and re-run training for peak accuracy.

## Phase 2: Action stub (demo)
If the router routes to `absence` and the user asks to mark someone absent (e.g., "mark Manju absent today"), the API returns an `action` block simulating the operation with basic entity extraction. See `app/actions.py` and `app/main.py`.

## Gemini Arbiter
Set `GEMINI_API_KEY` in your shell to enable LLM arbitration. The router gates calls to Gemini to conserve free-tier quota; use `--no-llm` in auto-tests to avoid rate limits.
