from __future__ import annotations
import json, os, re, hashlib
from pydantic import BaseModel
from typing import Literal, Optional, Dict, Any, List, Tuple
from pathlib import Path

# Optional heavy deps imported lazily inside functions to keep CLI quick.

class RouteDecision(BaseModel):
    route: Literal["ROUTE","OUT_OF_SCOPE","CONFUSED"]
    app_id: Optional[str] = None
    intent: Optional[str] = None
    confidence: float
    entities: Dict[str, Any] = {}
    rationale: str

# Simple in-process caches to avoid reloading heavy models per call
_RULES_CACHE: dict[str, Dict[str, Tuple[re.Pattern, Optional[re.Pattern]]]] = {}
_EMBED_CACHE: dict[str, tuple] = {}
_CLS_CACHE: Optional[dict] = None
_OOS_CACHE: Optional[dict] = None
_THRESHOLDS: Optional[dict] = None

def _catalog_fingerprint(catalog: dict) -> str:
    # Fingerprint only relevant parts of the catalog for stable caching
    safe = {
        "apps": [
            {
                "id": a.get("id"),
                "description": a.get("description", ""),
                "keywords": a.get("keywords", []),
                "intents": [
                    {"name": it.get("name"), "examples": it.get("examples", [])}
                    for it in a.get("intents", [])
                ],
            }
            for a in catalog.get("apps", [])
        ]
    }
    return hashlib.md5(json.dumps(safe, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

def load_catalog(path: str | Path) -> dict:
    import yaml  # pyyaml is a transitive dep via sentence-transformers
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_rules(catalog: dict):
    fp = _catalog_fingerprint(catalog)
    if fp in _RULES_CACHE:
        return _RULES_CACHE[fp]
    rules: Dict[str, Tuple[re.Pattern, Optional[re.Pattern]]] = {}
    for app in catalog["apps"]:
        aid = app["id"]
        kws = app.get("keywords", [])
        neg_kws = app.get("neg_keywords", [])
        pat = r"\b(" + "|".join([re.escape(k) for k in kws]) + r")\b" if kws else r"$^"  # no match if none
        neg_pat = r"\b(" + "|".join([re.escape(k) for k in neg_kws]) + r")\b" if neg_kws else None
        try:
            pos_rx = re.compile(pat, re.I)
        except re.error:
            pos_rx = re.compile("|".join([re.escape(k) for k in kws]), re.I)
        if neg_pat:
            try:
                neg_rx = re.compile(neg_pat, re.I)
            except re.error:
                neg_rx = re.compile("|".join([re.escape(k) for k in neg_kws]), re.I)
        else:
            neg_rx = None
        rules[aid] = (pos_rx, neg_rx)
    _RULES_CACHE[fp] = rules
    return rules

def build_app_corpus(app: dict) -> str:
    parts: List[str] = [app.get("description","")]
    parts += app.get("keywords", [])
    for it in app.get("intents", []):
        parts += it.get("examples", [])
    return "\n".join(parts)

def score_rules(msg: str, rules: Dict[str, Tuple["re.Pattern", Optional["re.Pattern"]]]):
    out = {}
    for aid, (pos_rx, neg_rx) in rules.items():
        if pos_rx.search(msg) and not (neg_rx and neg_rx.search(msg)):
            out[aid] = 0.95
        else:
            out[aid] = 0.0
    return out

def get_embedder_and_index(catalog: dict):
    model_name = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    fp = model_name + ":" + _catalog_fingerprint(catalog)
    if fp in _EMBED_CACHE:
        return _EMBED_CACHE[fp]
    from sentence_transformers import SentenceTransformer
    import numpy as np, faiss
    embedder = SentenceTransformer(model_name)
    app_texts = [build_app_corpus(a) for a in catalog["apps"]]
    app_ids = [a["id"] for a in catalog["apps"]]
    embs = embedder.encode(app_texts, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype("float32"))
    _EMBED_CACHE[fp] = (embedder, index, app_ids, embs)
    return _EMBED_CACHE[fp]

def score_embed(msg: str, embedder, embs, app_ids):
    import numpy as np
    q = embedder.encode([msg], normalize_embeddings=True)
    sims = (q @ embs.T).tolist()[0]
    return {aid: float(s) for aid, s in zip(app_ids, sims)}


def _load_classifier() -> Optional[dict]:
    global _CLS_CACHE
    if _CLS_CACHE is not None:
        return _CLS_CACHE
    model_path = Path("models/router.joblib")
    if not model_path.exists():
        _CLS_CACHE = None
        return _CLS_CACHE
    import joblib
    try:
        _CLS_CACHE = joblib.load(model_path)
    except Exception:
        _CLS_CACHE = None
    return _CLS_CACHE


def _load_thresholds() -> dict:
    global _THRESHOLDS
    if _THRESHOLDS is not None:
        return _THRESHOLDS
    cfg_path = Path("config/router_thresholds.json")
    if cfg_path.exists():
        try:
            _THRESHOLDS = json.load(open(cfg_path, "r", encoding="utf-8"))
        except Exception:
            _THRESHOLDS = {}
    else:
        _THRESHOLDS = {}
    return _THRESHOLDS


def _load_oos_head() -> Optional[dict]:
    global _OOS_CACHE
    if _OOS_CACHE is not None:
        return _OOS_CACHE
    model_path = Path("models/oos_head.joblib")
    if not model_path.exists():
        _OOS_CACHE = None
        return _OOS_CACHE
    import joblib
    try:
        _OOS_CACHE = joblib.load(model_path)
    except Exception:
        _OOS_CACHE = None
    return _OOS_CACHE


def normalize_text(s: str) -> str:
    # Lower, strip HTML tags, collapse spaces. Keep emojis.
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def llm_arbitrate(msg: str, catalog: dict, rule_score: dict, embed_score: dict) -> RouteDecision:
    # Uses Gemini if available; otherwise deterministic fallback. Gate LLM usage for rate limits.
    def top2(scores: dict[str, float]):
        if not scores:
            return (None, 0.0, 0.0)
        items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_id, best = items[0]
        second = items[1][1] if len(items) > 1 else 0.0
        margin = best - second
        return best_id, float(best), float(margin)

    best_rule_aid, best_rule, rule_margin = top2(rule_score)
    best_embed_aid, best_embed, embed_margin = top2(embed_score)

    api_key = os.getenv("GEMINI_API_KEY")

    # Heuristic: if signals are decisive, skip LLM
    if (best_rule >= 0.90 and rule_margin >= 0.20):
        return RouteDecision(route="ROUTE", app_id=best_rule_aid, intent=None,
                             confidence=best_rule, entities={}, rationale="rule gate")
    if (best_embed >= 0.70 and embed_margin >= 0.10):
        return RouteDecision(route="ROUTE", app_id=best_embed_aid, intent=None,
                             confidence=best_embed, entities={}, rationale="embed gate")

    if not api_key:
        # Conservative fallback without LLM
        if best_rule >= 0.9:
            return RouteDecision(route="ROUTE", app_id=best_rule_aid, intent=None,
                                 confidence=best_rule, entities={}, rationale="rule-based fallback")
        if best_embed >= 0.6:
            return RouteDecision(route="ROUTE", app_id=best_embed_aid, intent=None,
                                 confidence=best_embed, entities={}, rationale="embed-based fallback")
        return RouteDecision(route="CONFUSED", app_id=None, intent=None, confidence=0.3,
                             entities={}, rationale="no API key; low scores")
    from google import generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    schema = {
      "type":"object",
      "properties": {
        "route": {"enum":["ROUTE","OUT_OF_SCOPE","CONFUSED"]},
        "app_id": {"type":["string","null"]},
        "intent": {"type":["string","null"]},
        "confidence": {"type":"number"},
        "entities": {"type":"object"},
        "rationale": {"type":"string"}
      },
      "required":["route","confidence","rationale"]
    }
    prompt = f"""You are an enterprise router. Choose only from these apps: {[a['id'] for a in catalog['apps']]}.
If the message is unrelated, output OUT_OF_SCOPE. If unsure, CONFUSED.
Return ONLY JSON that matches this schema:
{json.dumps(schema)}

User message:
```{msg}```

Catalog:
```{json.dumps(catalog)}```

Signals:
rule_score={json.dumps(rule_score)}
embed_score={json.dumps(embed_score)}
"""
    resp = model.generate_content(
        prompt,
        generation_config={"response_mime_type":"application/json"}
    )
    try:
        data = json.loads(resp.text)
        return RouteDecision(**data)
    except Exception:
        # Fail-safe
        return RouteDecision(route="CONFUSED", app_id=None, intent=None, confidence=0.2, entities={}, rationale="parse error")

def calibrated_decision(msg: str, catalog_path: str | Path):
    raw_msg = msg
    msg = normalize_text(msg)
    catalog = load_catalog(catalog_path)
    rules = build_rules(catalog)
    rule_score = score_rules(msg, rules)

    # Embed similarity to app corpora
    embedder, index, app_ids, embs = get_embedder_and_index(catalog)
    embed_score = score_embed(msg, embedder, embs, app_ids)

    # Optional classifier probability over app_ids + OUT_OF_SCOPE
    cls = _load_classifier()
    cls_scores: Dict[str, float] = {aid: 0.0 for aid in app_ids}
    p_out = 0.0
    if cls is not None:
        pipe = cls["pipeline"]
        proba = pipe.predict_proba([raw_msg])[0]
        classes = list(pipe.classes_)
        for label, p in zip(classes, proba):
            if label in app_ids:
                cls_scores[label] = float(p)
            elif label == "OUT_OF_SCOPE":
                p_out = float(p)

    # Optional OOS head (binary). Positive class = OUT_OF_SCOPE (index 1)
    p_oos_head = 0.0
    oos_head = _load_oos_head()
    if oos_head is not None:
        oos_pipe = oos_head["pipeline"]
        p = oos_pipe.predict_proba([raw_msg])[0]
        # handle binary/ovr shapes
        p_oos_head = float(p[1]) if len(p) > 1 else float(p[0])

    # LLM arbitration only when needed
    rd = llm_arbitrate(raw_msg, catalog, rule_score, embed_score)

    # Combine signals: favor rules, then classifier, then embed/LLM
    combined = {}
    for aid in app_ids:
        combined[aid] = max(
            rule_score.get(aid, 0.0),
            0.7 * cls_scores.get(aid, 0.0) + 0.3 * embed_score.get(aid, 0.0),
            0.6 * (rd.confidence if rd.app_id == aid and rd.route == "ROUTE" else 0.0) + 0.4 * embed_score.get(aid, 0.0),
        )

    # Argmax + margin gating
    sorted_apps = sorted(app_ids, key=lambda a: combined[a], reverse=True)
    best_app = sorted_apps[0]
    second_app = sorted_apps[1] if len(sorted_apps) > 1 else best_app
    best_score = combined[best_app]
    second_score = combined[second_app]
    margin = best_score - second_score

    th = _load_thresholds()
    tau = float(th.get("tau", os.getenv("ROUTER_TAU", 0.60)))
    delta = float(th.get("delta", os.getenv("ROUTER_DELTA", 0.10)))
    oos_tau = float(th.get("oos_tau", os.getenv("ROUTER_OOS_TAU", 0.70)))
    oos_delta = float(th.get("oos_delta", os.getenv("ROUTER_OOS_DELTA", 0.20)))

    # Fuse OOS signals (binary head + multiclass OOS prob)
    p_oos_fused = max(p_out, p_oos_head)

    # Strong OOS gate: if OUT_OF_SCOPE clearly dominates
    if p_oos_fused >= oos_tau and (p_oos_fused - max(cls_scores.values())) >= oos_delta:
        return {
            "type": "SAFE_REPLY",
            "message": "This seems outside my scope. I cover Absence, TDO Drafting, and Cost Estimation.",
        }

    if best_score >= tau and margin >= delta:
        return {
            "type": "ROUTED",
            "app_id": best_app,
            "intent": rd.intent or "auto_infer",
            "confidence": round(float(best_score), 3),
            "entities": rd.entities,
            "message": f"I can help with that via {best_app.replace('_',' ').title()} (demo mode).",
        }

    # OOS gate: low signals + classifier OOS probability high
    if max(rule_score.values()) < 0.2 and max(embed_score.values()) < 0.35:
        return {
            "type": "SAFE_REPLY",
            "message": "I might not be the right assistant for that. I handle Absence Management, TDO Drafting, and Cost Estimation. Try: 'who was absent last week', 'draft a TDO for the payroll service', or 'estimate infra costs for a new API'.",
        }
    if p_oos_fused >= 0.5 and best_score < tau:
        return {
            "type": "SAFE_REPLY",
            "message": "This seems outside my scope. I cover Absence, TDO Drafting, and Cost Estimation.",
        }

    return {
        "type": "SAFE_REPLY",
        "message": "I’m not fully sure what you mean. Do you want Absence details, a TDO draft, or a Cost estimate? A little more context will help me route you correctly.",
    }
