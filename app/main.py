from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path
from .router import (
    calibrated_decision,
    load_catalog,
    build_rules,
    get_embedder_and_index,
)
from .actions import extract_absence_entities, perform_mark_absent

app = FastAPI()

class ChatReq(BaseModel):
    message: str
    session_id: Optional[str] = None

@app.on_event("startup")
def init_state():
    # Preload catalog, rules, and embeddings at startup for latency and stability
    root = Path(__file__).parent.parent
    catalog = load_catalog(root / "apps.yaml")
    rules = build_rules(catalog)
    embedder, index, app_ids, embs = get_embedder_and_index(catalog)
    app.state.catalog = catalog
    app.state.rules = rules
    app.state.embedder = embedder
    app.state.index = index
    app.state.app_ids = app_ids
    app.state.embs = embs

@app.post("/chat")
def chat(req: ChatReq):
    decision = calibrated_decision(
        req.message,
        Path(__file__).parent.parent / "apps.yaml",
        pre={
            "catalog": getattr(app.state, "catalog", None),
            "rules": getattr(app.state, "rules", None),
            "embedder": getattr(app.state, "embedder", None),
            "app_ids": getattr(app.state, "app_ids", None),
            "embs": getattr(app.state, "embs", None),
        },
    )
    # Phase 2: simulate action for absence "mark_absent"
    if (
        decision.get("type") == "ROUTED"
        and decision.get("app_id") == "absence"
    ):
        intent = str(decision.get("intent") or "").lower()
        # Heuristic: either explicit intent OR surface pattern in the message
        if intent == "mark_absent" or (
            __import__("re").search(r"\b(mark|set|put)\b.*\b(absent|on leave)\b", req.message, __import__("re").I)
        ):
            ents: Dict[str, Any] = decision.get("entities") or {}
            args = extract_absence_entities(req.message)
            # Prefer LLM-provided entities when present
            employee = (ents.get("employee") or ents.get("person") or args.employee)
            date = (ents.get("date") or ents.get("date_range") or args.date)
            result = perform_mark_absent(type(args)(employee=employee, date=date))
            # Return an enriched response including action stub
            return {
                **decision,
                "intent": "mark_absent",
                "action": {
                    "name": "mark_absent",
                    "args": {"employee": employee, "date": date},
                    "result": result,
                },
                "message": result.get("message", decision.get("message")),
            }
    return decision
