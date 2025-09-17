#!/usr/bin/env python3
"""
Comprehensive Analysis of the Chatbot Router System
This script demonstrates every step of the routing process with detailed explanations.
"""

import sys
sys.path.append('.')

import json
import numpy as np
from pathlib import Path
from app.router import (
    calibrated_decision, load_catalog, build_rules, score_rules,
    get_embedder_and_index, score_embed, normalize_text,
    llm_arbitrate, _load_classifier, _load_oos_head, _load_thresholds
)

def analyze_routing_step_by_step(message: str):
    """
    Analyze the routing process step by step with detailed explanations
    """
    print("=" * 80)
    print(f"COMPREHENSIVE ROUTING ANALYSIS FOR: '{message}'")
    print("=" * 80)
    
    # Step 1: Text Normalization
    print("\n1. TEXT NORMALIZATION")
    print("-" * 40)
    raw_msg = message
    normalized_msg = normalize_text(message)
    print(f"Original message: '{raw_msg}'")
    print(f"Normalized message: '{normalized_msg}'")
    print("Normalization removes HTML tags, converts to lowercase, collapses whitespace")
    
    # Step 2: Load Configuration
    print("\n2. LOAD CONFIGURATION")
    print("-" * 40)
    catalog_path = Path('apps.yaml')
    catalog = load_catalog(catalog_path)
    print(f"Loaded catalog with {len(catalog['apps'])} applications:")
    for app in catalog['apps']:
        print(f"  - {app['id']}: {app['display_name']}")
        print(f"    Keywords: {app.get('keywords', [])}")
        print(f"    Negative Keywords: {app.get('neg_keywords', [])}")
    
    # Step 3: Rule-based Scoring
    print("\n3. RULE-BASED SCORING")
    print("-" * 40)
    rules = build_rules(catalog)
    rule_scores = score_rules(normalized_msg, rules)
    print("Rule patterns compiled:")
    for app_id, (pos_pattern, neg_pattern) in rules.items():
        print(f"  {app_id}:")
        print(f"    Positive pattern: {pos_pattern.pattern}")
        if neg_pattern:
            print(f"    Negative pattern: {neg_pattern.pattern}")
        else:
            print(f"    Negative pattern: None")
    
    print("\nRule scores:")
    for app_id, score in rule_scores.items():
        print(f"  {app_id}: {score:.6f}")
        if score > 0:
            print(f"    ✓ Positive keyword match found!")
        else:
            print(f"    ✗ No keyword match")
    
    # Step 4: Embedding-based Scoring
    print("\n4. EMBEDDING-BASED SCORING")
    print("-" * 40)
    embedder, index, app_ids, embs = get_embedder_and_index(catalog)
    embed_scores = score_embed(normalized_msg, embedder, embs, app_ids)
    
    print(f"Using embedder: {embedder}")
    print(f"Embedding dimension: {embs.shape[1]}")
    print(f"Number of app embeddings: {embs.shape[0]}")
    
    # Show what each app's corpus looks like
    print("\nApp corpora used for embeddings:")
    for i, app in enumerate(catalog['apps']):
        corpus_parts = [app.get('description', '')]
        corpus_parts += app.get('keywords', [])
        for intent in app.get('intents', []):
            corpus_parts += intent.get('examples', [])
        corpus = '\n'.join(corpus_parts)
        print(f"  {app['id']}:")
        print(f"    Corpus: {corpus[:100]}..." if len(corpus) > 100 else f"    Corpus: {corpus}")
    
    print("\nEmbedding similarity scores:")
    for app_id, score in embed_scores.items():
        print(f"  {app_id}: {score:.6f}")
    
    # Step 5: Classifier Scoring
    print("\n5. CLASSIFIER SCORING")
    print("-" * 40)
    cls = _load_classifier()
    cls_scores = {aid: 0.0 for aid in app_ids}
    p_out = 0.0
    
    if cls is not None:
        pipe = cls["pipeline"]
        proba = pipe.predict_proba([raw_msg])[0]
        classes = list(pipe.classes_)
        
        print(f"Classifier loaded successfully")
        print(f"Classes: {classes}")
        print(f"Raw probabilities: {proba}")
        
        for label, p in zip(classes, proba):
            if label in app_ids:
                cls_scores[label] = float(p)
            elif label == "OUT_OF_SCOPE":
                p_out = float(p)
        
        print("Classifier scores:")
        for app_id, score in cls_scores.items():
            print(f"  {app_id}: {score:.6f}")
        print(f"  OUT_OF_SCOPE: {p_out:.6f}")
    else:
        print("No classifier model found (models/router.joblib)")
    
    # Step 6: OOS Head Scoring
    print("\n6. OUT-OF-SCOPE HEAD SCORING")
    print("-" * 40)
    p_oos_head = 0.0
    oos_head = _load_oos_head()
    
    if oos_head is not None:
        oos_pipe = oos_head["pipeline"]
        p = oos_pipe.predict_proba([raw_msg])[0]
        p_oos_head = float(p[1]) if len(p) > 1 else float(p[0])
        print(f"OOS head probability: {p_oos_head:.6f}")
    else:
        print("No OOS head model found (models/oos_head.joblib)")
    
    # Step 7: LLM Arbitration
    print("\n7. LLM ARBITRATION")
    print("-" * 40)
    rd = llm_arbitrate(raw_msg, catalog, rule_scores, embed_scores)
    print(f"LLM decision:")
    print(f"  Route: {rd.route}")
    print(f"  App ID: {rd.app_id}")
    print(f"  Intent: {rd.intent}")
    print(f"  Confidence: {rd.confidence:.6f}")
    print(f"  Rationale: {rd.rationale}")
    print(f"  Entities: {rd.entities}")
    
    # Step 8: Score Fusion
    print("\n8. SCORE FUSION AND MARGIN GATING")
    print("-" * 40)
    
    # Calculate combined scores
    combined = {}
    for aid in app_ids:
        rule_component = rule_scores.get(aid, 0.0)
        cls_embed_component = 0.7 * cls_scores.get(aid, 0.0) + 0.3 * embed_scores.get(aid, 0.0)
        llm_embed_component = 0.6 * (rd.confidence if rd.app_id == aid and rd.route == "ROUTE" else 0.0) + 0.4 * embed_scores.get(aid, 0.0)
        
        combined[aid] = max(rule_component, cls_embed_component, llm_embed_component)
        
        print(f"{aid}:")
        print(f"  Rule component: {rule_component:.6f}")
        print(f"  Classifier+Embed component (0.7*cls + 0.3*embed): {cls_embed_component:.6f}")
        print(f"    = 0.7 * {cls_scores.get(aid, 0.0):.6f} + 0.3 * {embed_scores.get(aid, 0.0):.6f}")
        print(f"  LLM+Embed component (0.6*llm + 0.4*embed): {llm_embed_component:.6f}")
        print(f"    = 0.6 * {rd.confidence if rd.app_id == aid and rd.route == 'ROUTE' else 0.0:.6f} + 0.4 * {embed_scores.get(aid, 0.0):.6f}")
        print(f"  Combined score (max of above): {combined[aid]:.6f}")
        print()
    
    # Find top 2 scores for margin calculation
    sorted_apps = sorted(app_ids, key=lambda a: combined[a], reverse=True)
    best_app = sorted_apps[0]
    second_app = sorted_apps[1] if len(sorted_apps) > 1 else best_app
    best_score = combined[best_app]
    second_score = combined[second_app]
    margin = best_score - second_score
    
    print(f"Top scores:")
    print(f"  1st place: {best_app} with score {best_score:.6f}")
    print(f"  2nd place: {second_app} with score {second_score:.6f}")
    print(f"  Margin: {margin:.6f}")
    
    # Load thresholds
    th = _load_thresholds()
    tau = float(th.get("tau", 0.60))
    delta = float(th.get("delta", 0.10))
    oos_tau = float(th.get("oos_tau", 0.70))
    oos_delta = float(th.get("oos_delta", 0.20))
    
    print(f"\nThresholds:")
    print(f"  tau (confidence threshold): {tau}")
    print(f"  delta (margin threshold): {delta}")
    print(f"  oos_tau (OOS confidence threshold): {oos_tau}")
    print(f"  oos_delta (OOS margin threshold): {oos_delta}")
    
    # OOS fusion
    p_oos_fused = max(p_out, p_oos_head)
    print(f"\nOOS Analysis:")
    print(f"  Classifier OOS probability: {p_out:.6f}")
    print(f"  OOS head probability: {p_oos_head:.6f}")
    print(f"  Fused OOS probability: {p_oos_fused:.6f}")
    
    # Decision logic
    print(f"\n9. FINAL DECISION LOGIC")
    print("-" * 40)
    
    # Check strong OOS gate
    oos_margin = p_oos_fused - max(cls_scores.values()) if cls_scores else p_oos_fused
    print(f"OOS vs best classifier margin: {oos_margin:.6f}")
    
    if p_oos_fused >= oos_tau and oos_margin >= oos_delta:
        print(f"✓ Strong OOS gate triggered: p_oos_fused ({p_oos_fused:.6f}) >= oos_tau ({oos_tau}) AND margin ({oos_margin:.6f}) >= oos_delta ({oos_delta})")
        decision_type = "SAFE_REPLY (Strong OOS)"
    elif best_score >= tau and margin >= delta:
        print(f"✓ Routing gate passed: best_score ({best_score:.6f}) >= tau ({tau}) AND margin ({margin:.6f}) >= delta ({delta})")
        decision_type = f"ROUTED to {best_app}"
    elif max(rule_scores.values()) < 0.2 and max(embed_scores.values()) < 0.35:
        print(f"✓ Weak signals gate: max_rule ({max(rule_scores.values()):.6f}) < 0.2 AND max_embed ({max(embed_scores.values()):.6f}) < 0.35")
        decision_type = "SAFE_REPLY (Weak signals)"
    elif p_oos_fused >= 0.5 and best_score < tau:
        print(f"✓ OOS probability gate: p_oos_fused ({p_oos_fused:.6f}) >= 0.5 AND best_score ({best_score:.6f}) < tau ({tau})")
        decision_type = "SAFE_REPLY (OOS probability)"
    else:
        print(f"✓ Default fallback: No clear routing decision")
        decision_type = "SAFE_REPLY (Confused)"
    
    print(f"\nFinal Decision: {decision_type}")
    
    # Step 10: Compare with actual system output
    print(f"\n10. ACTUAL SYSTEM OUTPUT")
    print("-" * 40)
    actual_decision = calibrated_decision(raw_msg, catalog_path)
    print(json.dumps(actual_decision, indent=2))
    
    return actual_decision

def main():
    # Test cases covering different scenarios
    test_cases = [
        "who was absent last week",  # Should route to absence
        "draft a TDO for invoice microservice",  # Should route to tdo_drafting  
        "estimate cost for new API",  # Should route to cost_estimation
        "what's the weather today",  # Should be OUT_OF_SCOPE
        "hello there",  # Should be confused/safe reply
        "mark Manju absent today",  # Should route to absence with action
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*100}")
        print(f"TEST CASE {i+1}: {test_case}")
        print(f"{'='*100}")
        
        try:
            result = analyze_routing_step_by_step(test_case)
        except Exception as e:
            print(f"Error analyzing '{test_case}': {e}")
            import traceback
            traceback.print_exc()
        
        if i < len(test_cases) - 1:
            input("\nPress Enter to continue to next test case...")

if __name__ == "__main__":
    main()