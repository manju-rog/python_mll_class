# Score Fusion Calculation Breakdown

## Your Example Analysis: "who was absent last week"

### Raw Scores from System Components

#### 1. Rule Scores (score_rules)
```
absence:         0.950000  ✓ (keyword "absent" matched)
tdo_drafting:    0.000000  ✗ (no TDO keywords)
cost_estimation: 0.000000  ✗ (no cost keywords)
```

#### 2. Embedding Scores (score_embed with MiniLM)
```
absence:         0.281586  (semantic similarity to absence corpus)
tdo_drafting:    0.053471  (low similarity to TDO corpus)
cost_estimation: 0.129069  (some similarity due to "estimation" context)
```

#### 3. Classifier Scores (models/router.joblib via _load_classifier)
```
absence:         0.999375  (very confident - trained on similar patterns)
tdo_drafting:    0.000005  (almost zero probability)
cost_estimation: 0.000548  (very low probability)
OUT_OF_SCOPE:    0.000073  (very low - clearly in scope)
```

#### 4. OOS Head Score (models/oos_head.joblib)
```
OUT_OF_SCOPE probability: 0.000098  (tiny - clearly not out of scope)
```

#### 5. LLM Arbitration (llm_arbitrate)
```
Route: CONFUSED
Reason: "no API key, low rule score"
Confidence: 0.3 (would be higher with API key)
```

### Score Fusion Calculations (calibrated_decision)

For each app, compute 3 components and take maximum:

#### Component Formulas:
1. **Rule component**: `rule_score[app]`
2. **Classifier+Embed**: `0.7 × classifier_prob[app] + 0.3 × embed_score[app]`  
3. **LLM+Embed**: `0.6 × llm_confidence[app] + 0.4 × embed_score[app]`

#### Detailed Calculations:

**ABSENCE:**
- Rule: `0.950000`
- Cls+Embed: `0.7 × 0.999375 + 0.3 × 0.281586 = 0.699563 + 0.084476 = 0.784039`
- LLM+Embed: `0.6 × 0.0 + 0.4 × 0.281586 = 0.0 + 0.112634 = 0.112634`
- **Combined**: `max(0.950000, 0.784039, 0.112634) = 0.950000`

**COST_ESTIMATION:**  
- Rule: `0.000000`
- Cls+Embed: `0.7 × 0.000548 + 0.3 × 0.129069 = 0.000384 + 0.038721 = 0.039105`
- LLM+Embed: `0.6 × 0.0 + 0.4 × 0.129069 = 0.0 + 0.051628 = 0.051628`
- **Combined**: `max(0.000000, 0.039105, 0.051628) = 0.051628`

**TDO_DRAFTING:**
- Rule: `0.000000`  
- Cls+Embed: `0.7 × 0.000005 + 0.3 × 0.053471 = 0.0000035 + 0.016041 = 0.016044`
- LLM+Embed: `0.6 × 0.0 + 0.4 × 0.053471 = 0.0 + 0.021388 = 0.021388`
- **Combined**: `max(0.000000, 0.016044, 0.021388) = 0.021388`

### Final Ranking and Margin Analysis

**Top Scores:**
1. **Absence**: 0.950000 (clear winner)
2. **Cost Estimation**: 0.051628 (distant second)  
3. **TDO Drafting**: 0.021388 (third)

**Margin Calculation:**
- **Winner margin**: 0.950000 - 0.051628 = **0.898372**
- **Required margin (delta)**: 0.0 (from config)
- **Confidence threshold (tau)**: 0.5 (from config)

**Decision Logic:**
- ✅ `best_score (0.950000) >= tau (0.5)` 
- ✅ `margin (0.898372) >= delta (0.0)`
- **Result**: **ROUTED to absence**

### Why These Numbers Make Sense

1. **Rule dominance**: The word "absent" is a perfect keyword match, giving maximum rule score (0.95)

2. **Classifier confidence**: The trained model is extremely confident (99.9%) this is an absence query

3. **Embedding support**: Semantic similarity confirms it's absence-related, though not as strongly

4. **Score fusion wisdom**: The max() operation picks the strongest signal (rules in this case)

5. **Margin safety**: Huge margin (0.898) ensures no ambiguity between apps

6. **OOS rejection**: Tiny OOS probabilities confirm this is clearly in-scope

This demonstrates the system's multi-layered approach working correctly - when one signal is very strong (rules), it dominates, but other signals provide confirmation and would take over if rules failed.