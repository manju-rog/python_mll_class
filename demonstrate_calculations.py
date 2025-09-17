#!/usr/bin/env python3
"""
Demonstrate the exact calculations from the routing system
"""

import sys
sys.path.append('.')

from app.router import calibrated_decision
from pathlib import Path
import json

def demonstrate_score_fusion():
    """
    Show the exact score fusion calculations mentioned in the analysis
    """
    
    print("=" * 80)
    print("DEMONSTRATING SCORE FUSION CALCULATIONS")
    print("=" * 80)
    
    # Example values from your analysis
    print("\n1. INPUT SCORES (from system components)")
    print("-" * 50)
    
    # Rule scores
    rule_scores = {
        'absence': 0.0,  # Typo breaks regex, all apps score 0.0
        'cost_estimation': 0.0,
        'tdo_drafting': 0.0
    }
    print("Rule scores (typo breaks regex):")
    for app, score in rule_scores.items():
        print(f"  {app}: {score:.6f}")
    
    # Embedding scores  
    embed_scores = {
        'absence': 0.281586,
        'cost_estimation': 0.129069,
        'tdo_drafting': 0.053471
    }
    print("\nEmbedding scores (MiniLM):")
    for app, score in embed_scores.items():
        print(f"  {app}: {score:.6f}")
    
    # Classifier scores
    cls_scores = {
        'absence': 0.999375,
        'cost_estimation': 0.000548,
        'tdo_drafting': 0.000005
    }
    oos_prob = 0.000073
    print("\nClassifier scores:")
    for app, score in cls_scores.items():
        print(f"  {app}: {score:.6f}")
    print(f"  OUT_OF_SCOPE: {oos_prob:.6f}")
    
    # OOS head
    oos_head_prob = 0.000098
    print(f"\nOOS head probability: {oos_head_prob:.6f}")
    
    # LLM (returns CONFUSED with no API key)
    llm_confidence = 0.0  # No routing decision from LLM
    print(f"\nLLM arbitration: CONFUSED (no API key), confidence: {llm_confidence}")
    
    print("\n2. SCORE FUSION CALCULATIONS")
    print("-" * 50)
    
    combined_scores = {}
    
    for app in ['absence', 'cost_estimation', 'tdo_drafting']:
        print(f"\n{app.upper()}:")
        
        # Component 1: Rule
        rule_component = rule_scores[app]
        print(f"  Rule component: {rule_component:.6f}")
        
        # Component 2: 0.7*classifier + 0.3*embedding
        cls_embed_component = 0.7 * cls_scores[app] + 0.3 * embed_scores[app]
        print(f"  Classifier+Embed: 0.7 × {cls_scores[app]:.6f} + 0.3 × {embed_scores[app]:.6f} = {cls_embed_component:.6f}")
        
        # Component 3: 0.6*llm + 0.4*embedding (LLM didn't route to this app)
        llm_embed_component = 0.6 * 0.0 + 0.4 * embed_scores[app]
        print(f"  LLM+Embed: 0.6 × 0.0 + 0.4 × {embed_scores[app]:.6f} = {llm_embed_component:.6f}")
        
        # Combined (max of the three)
        combined = max(rule_component, cls_embed_component, llm_embed_component)
        combined_scores[app] = combined
        print(f"  Combined (max): {combined:.6f}")
    
    print("\n3. FINAL RANKING AND DECISION")
    print("-" * 50)
    
    # Sort by combined score
    sorted_apps = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("Final combined scores:")
    for i, (app, score) in enumerate(sorted_apps, 1):
        print(f"  {i}. {app}: {score:.6f}")
    
    # Top 2 for margin calculation
    best_app, best_score = sorted_apps[0]
    second_app, second_score = sorted_apps[1]
    margin = best_score - second_score
    
    print(f"\nTop confidence: {best_score:.6f}")
    print(f"Runner-up: {second_score:.6f}")
    print(f"Margin: {margin:.6f}")
    
    # Thresholds (from your example)
    tau = 0.60  # threshold
    delta = 0.10  # margin requirement
    
    print(f"\nThresholds:")
    print(f"  tau (confidence threshold): {tau}")
    print(f"  margin requirement: {delta}")
    
    # Decision logic
    print(f"\nDecision Analysis:")
    if best_score >= tau:
        print(f"  ✓ Confidence check: {best_score:.6f} >= {tau}")
    else:
        print(f"  ✗ Confidence check: {best_score:.6f} < {tau}")
        
    if margin >= delta:
        print(f"  ✓ Margin check: {margin:.6f} >= {delta}")
    else:
        print(f"  ✗ Margin check: {margin:.6f} < {delta}")
    
    if best_score >= tau and margin >= delta:
        print(f"\n🎯 DECISION: ROUTE to {best_app}")
        print(f"   Confidence: {best_score:.6f}")
        print(f"   Margin vs runner-up: {margin:.6f}")
    else:
        print(f"\n🛡️  DECISION: SAFE_REPLY (insufficient confidence/margin)")

def test_actual_system():
    """
    Test the actual system to compare with manual calculations
    """
    print("\n\n" + "=" * 80)
    print("ACTUAL SYSTEM TEST")
    print("=" * 80)
    
    test_message = "who was absent last week"
    print(f"\nTesting message: '{test_message}'")
    
    result = calibrated_decision(test_message, Path('apps.yaml'))
    print("\nActual system output:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    demonstrate_score_fusion()
    test_actual_system()