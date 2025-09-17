# Complete Chatbot Router System Analysis

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Components](#architecture-components)
3. [Step-by-Step Routing Process](#step-by-step-routing-process)
4. [Detailed Calculation Examples](#detailed-calculation-examples)
5. [Score Fusion Mathematics](#score-fusion-mathematics)
6. [Threshold and Gating Logic](#threshold-and-gating-logic)
7. [Code Walkthrough](#code-walkthrough)
8. [Training and Model Creation](#training-and-model-creation)
9. [Configuration Files](#configuration-files)
10. [Real Examples with Complete Calculations](#real-examples-with-complete-calculations)

## System Overview

This is a **hybrid multi-modal chatbot router** that combines multiple AI techniques to route user messages to the correct application with high accuracy. The system supports three main applications:

- **Absence Management**: Handle employee absence queries and actions
- **TDO Drafting**: Create Technical Design Outlines
- **Cost Estimation**: Provide cost estimates and predictions

### Key Features
- **Multi-modal scoring**: Rules, embeddings, classifier, and LLM arbitration
- **Robust OOS detection**: Multiple out-of-scope detection mechanisms
- **Margin gating**: Confidence thresholds with margin requirements
- **Calibrated probabilities**: Trained classifier with probability calibration
- **Synthetic data training**: Thousands of examples with noise, typos, and code-switching

## Architecture Components

### 1. Rule-Based Engine (`score_rules`)
- **Purpose**: Fast, deterministic keyword matching
- **Implementation**: Compiled regex patterns for positive and negative keywords
- **Score**: 0.95 for matches, 0.0 for no matches
- **Advantage**: High precision, fast execution

### 2. Embedding Similarity (`score_embed`)
- **Purpose**: Semantic similarity using sentence transformers
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (default, configurable)
- **Process**: 
  - Encode user message and app corpora
  - Compute cosine similarity (dot product with normalized embeddings)
  - Return similarity scores [-1, 1]
- **Advantage**: Handles semantic variations and synonyms

### 3. Trained Classifier (`_load_classifier`)
- **Purpose**: Learned patterns from synthetic training data
- **Architecture**: Logistic Regression over embeddings
- **Classes**: `absence`, `tdo_drafting`, `cost_estimation`, `OUT_OF_SCOPE`
- **Calibration**: Uses `CalibratedClassifierCV` for reliable probabilities
- **Advantage**: Learns complex patterns from data

### 4. OOS Head (`_load_oos_head`)
- **Purpose**: Binary out-of-scope detection
- **Architecture**: Separate binary classifier
- **Output**: Probability that message is out-of-scope
- **Advantage**: Specialized OOS detection

### 5. LLM Arbitration (`llm_arbitrate`)
- **Purpose**: Intelligent tie-breaking and entity extraction
- **Model**: Gemini 2.5 Flash (when API key available)
- **Gating**: Only called when other signals are inconclusive
- **Fallback**: Deterministic logic when no API key
- **Advantage**: Handles complex, ambiguous cases

## Step-by-Step Routing Process

### Step 1: Text Normalization
```python
def normalize_text(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)  # Remove HTML tags
    s = s.lower()                   # Convert to lowercase
    s = re.sub(r"\s+", " ", s).strip()  # Collapse whitespace
    return s
```

**Example**: `"Who was ABSENT last week?"` → `"who was absent last week?"`

### Step 2: Load Configuration
The system loads the application catalog from `apps.yaml`:

```yaml
apps:
  - id: "absence"
    display_name: "Absence Management"
    description: "View team absences, mark employees absent/present, date-range summaries."
    keywords: ["absent", "leave", "on leave", "who was out", "attendance", "PTO", "paid time off", "holiday"]
    neg_keywords: ["leave the page", "left join", "leave it", "leave me alone", "maternity leave policy pdf", "absent minded"]
    intents:
      - name: "get_absences"
        examples: ["who was absent last week", "show yesterday's absences", "who is on leave today"]
      - name: "mark_absent"
        examples: ["mark Manju absent for today", "mark Rahul on leave tomorrow"]
```

### Step 3: Rule-Based Scoring
For each app, compile regex patterns:
- **Positive pattern**: `\b(absent|leave|on\ leave|who\ was\ out|attendance|PTO|paid\ time\ off|holiday)\b`
- **Negative pattern**: `\b(leave\ the\ page|left\ join|leave\ it|leave\ me\ alone|maternity\ leave\ policy\ pdf|absent\ minded)\b`

**Scoring logic**:
```python
if pos_rx.search(msg) and not (neg_rx and neg_rx.search(msg)):
    score = 0.95
else:
    score = 0.0
```

### Step 4: Embedding Similarity
1. **Build app corpora**: Concatenate description + keywords + intent examples
2. **Encode**: Use sentence transformer to create embeddings
3. **Index**: Store in FAISS index for fast similarity search
4. **Query**: Encode user message and compute cosine similarity

**Example corpus for absence**:
```
View team absences, mark employees absent/present, date-range summaries.
absent
leave
on leave
who was out
attendance
PTO
paid time off
holiday
who was absent last week
show yesterday's absences
who is on leave today
mark Manju absent for today
mark Rahul on leave tomorrow
```

### Step 5: Classifier Scoring
1. **Load model**: Trained LogisticRegression with calibration
2. **Predict**: Get probability distribution over all classes
3. **Extract**: Separate app probabilities and OUT_OF_SCOPE probability

### Step 6: OOS Head Scoring
1. **Load binary model**: Specialized out-of-scope detector
2. **Predict**: Binary probability (in-scope vs out-of-scope)
3. **Extract**: OUT_OF_SCOPE probability

### Step 7: LLM Arbitration
**Gating conditions** (skip LLM if):
- Rule score ≥ 0.90 with margin ≥ 0.20
- Embed score ≥ 0.70 with margin ≥ 0.10

**LLM prompt structure**:
```
You are an enterprise router. Choose only from these apps: ['absence', 'tdo_drafting', 'cost_estimation'].
If the message is unrelated, output OUT_OF_SCOPE. If unsure, CONFUSED.
Return ONLY JSON that matches this schema: {...}

User message: ```{msg}```
Catalog: ```{catalog}```
Signals: rule_score={rule_score}, embed_score={embed_score}
```

### Step 8: Score Fusion
For each app, compute three components and take the maximum:

1. **Rule component**: `rule_score[app]`
2. **Classifier+Embed component**: `0.7 * classifier_prob[app] + 0.3 * embed_score[app]`
3. **LLM+Embed component**: `0.6 * llm_confidence[app] + 0.4 * embed_score[app]`

```python
combined[app] = max(
    rule_score.get(app, 0.0),
    0.7 * cls_scores.get(app, 0.0) + 0.3 * embed_score.get(app, 0.0),
    0.6 * (rd.confidence if rd.app_id == app and rd.route == "ROUTE" else 0.0) + 0.4 * embed_score.get(app, 0.0)
)
```

### Step 9: Margin Gating and Final Decision
1. **Find top 2 scores**: Sort apps by combined score
2. **Calculate margin**: `best_score - second_score`
3. **Load thresholds**: From `config/router_thresholds.json`
4. **Apply decision logic**: Multiple gates in priority order

## Detailed Calculation Examples

### Example 1: "who was absent last week"

#### Input Processing
- **Raw message**: `"who was absent last week"`
- **Normalized**: `"who was absent last week"` (no change needed)

#### Rule Scoring
- **absence**: Pattern `\b(absent|...)\b` matches "absent" → **0.950000**
- **tdo_drafting**: No keyword match → **0.000000**
- **cost_estimation**: No keyword match → **0.000000**

#### Embedding Scoring
- **absence**: High similarity to absence corpus → **0.588726**
- **tdo_drafting**: Low similarity → **0.060084**
- **cost_estimation**: Very low similarity → **0.023606**

#### Classifier Scoring
- **Raw probabilities**: `[2.36e-04, 9.996e-01, 1.10e-04, 3.47e-05]`
- **absence**: **0.999619** (very confident)
- **tdo_drafting**: **0.000035**
- **cost_estimation**: **0.000110**
- **OUT_OF_SCOPE**: **0.000236**

#### OOS Head Scoring
- **Binary OOS probability**: **0.000022** (very low, clearly in-scope)

#### LLM Arbitration
- **Gating triggered**: Rule score (0.95) ≥ 0.90 with margin ≥ 0.20
- **Decision**: Route to absence with confidence 0.95
- **Rationale**: "rule gate" (skipped LLM call)

#### Score Fusion
**absence**:
- Rule component: **0.950000**
- Classifier+Embed: 0.7 × 0.999619 + 0.3 × 0.588726 = **0.876352**
- LLM+Embed: 0.6 × 0.950000 + 0.4 × 0.588726 = **0.805491**
- **Combined (max)**: **0.950000**

**tdo_drafting**:
- Rule component: **0.000000**
- Classifier+Embed: 0.7 × 0.000035 + 0.3 × 0.060084 = **0.018050**
- LLM+Embed: 0.6 × 0.000000 + 0.4 × 0.060084 = **0.024034**
- **Combined (max)**: **0.024034**

**cost_estimation**:
- Rule component: **0.000000**
- Classifier+Embed: 0.7 × 0.000110 + 0.3 × 0.023606 = **0.007159**
- LLM+Embed: 0.6 × 0.000000 + 0.4 × 0.023606 = **0.009442**
- **Combined (max)**: **0.009442**

#### Final Decision
- **Top score**: absence (0.950000)
- **Second score**: tdo_drafting (0.024034)
- **Margin**: 0.950000 - 0.024034 = **0.925966**
- **Thresholds**: tau=0.5, delta=0.0
- **Decision**: ✅ **ROUTED to absence** (0.950000 ≥ 0.5 AND 0.925966 ≥ 0.0)

### Example 2: "what's the weather today" (Out-of-Scope)

#### Input Processing
- **Raw message**: `"what's the weather today"`
- **Normalized**: `"what's the weather today"`

#### Rule Scoring
- **All apps**: No keyword matches → **0.000000** each

#### Embedding Scoring
- **absence**: **0.133220** (some similarity due to "today")
- **tdo_drafting**: **-0.021019** (negative similarity)
- **cost_estimation**: **0.033101** (low similarity)

#### Classifier Scoring
- **Raw probabilities**: `[8.864e-01, 1.111e-01, 2.067e-03, 4.414e-04]`
- **absence**: **0.111075**
- **tdo_drafting**: **0.000441**
- **cost_estimation**: **0.002067**
- **OUT_OF_SCOPE**: **0.886416** (very confident it's out-of-scope)

#### OOS Head Scoring
- **Binary OOS probability**: **0.869361** (high confidence it's out-of-scope)

#### LLM Arbitration
- **No gating triggered**: All scores too low
- **No API key**: Falls back to deterministic logic
- **Decision**: CONFUSED with confidence 0.3

#### Score Fusion
**absence**:
- Rule component: **0.000000**
- Classifier+Embed: 0.7 × 0.111075 + 0.3 × 0.133220 = **0.117719**
- LLM+Embed: 0.6 × 0.000000 + 0.4 × 0.133220 = **0.053288**
- **Combined (max)**: **0.117719**

**tdo_drafting**:
- Rule component: **0.000000**
- Classifier+Embed: 0.7 × 0.000441 + 0.3 × (-0.021019) = **-0.005997** → **0.000000**
- LLM+Embed: 0.6 × 0.000000 + 0.4 × (-0.021019) = **-0.008407** → **0.000000**
- **Combined (max)**: **0.000000**

**cost_estimation**:
- Rule component: **0.000000**
- Classifier+Embed: 0.7 × 0.002067 + 0.3 × 0.033101 = **0.011377**
- LLM+Embed: 0.6 × 0.000000 + 0.4 × 0.033101 = **0.013240**
- **Combined (max)**: **0.013240**

#### OOS Analysis
- **Classifier OOS**: 0.886416
- **OOS Head**: 0.869361
- **Fused OOS**: max(0.886416, 0.869361) = **0.886416**
- **OOS vs best classifier margin**: 0.886416 - 0.111075 = **0.775341**

#### Final Decision
- **Strong OOS Gate**: 0.886416 ≥ 0.0825 AND 0.775341 ≥ 0.15
- **Decision**: ✅ **SAFE_REPLY (Strong OOS)**
- **Message**: "This seems outside my scope. I cover Absence, TDO Drafting, and Cost Estimation."

## Score Fusion Mathematics

The score fusion uses a **max-pooling strategy** across three different signal combinations:

### Component 1: Rule-Based (High Precision)
```
score₁ = rule_score[app]
```
- **Weight**: Direct score (0.95 or 0.0)
- **Purpose**: Catch high-confidence keyword matches
- **Advantage**: Fast, deterministic, high precision

### Component 2: Classifier + Embedding Blend (Balanced)
```
score₂ = 0.7 × classifier_prob[app] + 0.3 × embed_score[app]
```
- **Classifier weight**: 0.7 (higher weight for learned patterns)
- **Embedding weight**: 0.3 (semantic similarity boost)
- **Purpose**: Combine learned patterns with semantic understanding

### Component 3: LLM + Embedding Blend (Intelligent)
```
score₃ = 0.6 × llm_confidence[app] + 0.4 × embed_score[app]
```
- **LLM weight**: 0.6 (intelligent reasoning)
- **Embedding weight**: 0.4 (semantic grounding)
- **Purpose**: Handle complex cases with reasoning

### Final Fusion
```
combined[app] = max(score₁, score₂, score₃)
```

**Why max-pooling?**
- **Robustness**: If one component fails, others can still succeed
- **Complementary**: Different components excel in different scenarios
- **Conservative**: Takes the most confident signal

## Threshold and Gating Logic

### Primary Thresholds (from `config/router_thresholds.json`)
```json
{
  "tau": 0.5,           // Minimum confidence to route
  "delta": 0.0,         // Minimum margin between top 2 scores
  "oos_tau": 0.0825,    // OOS confidence threshold
  "oos_delta": 0.15     // OOS margin threshold
}
```

### Decision Gates (in priority order)

#### 1. Strong OOS Gate (Highest Priority)
```python
if p_oos_fused >= oos_tau and (p_oos_fused - max(cls_scores.values())) >= oos_delta:
    return SAFE_REPLY("This seems outside my scope...")
```
**Condition**: OOS probability high AND significantly higher than best app probability

#### 2. Routing Gate
```python
if best_score >= tau and margin >= delta:
    return ROUTED(best_app, confidence=best_score)
```
**Condition**: Best app score above threshold AND sufficient margin over second place

#### 3. Weak Signals Gate
```python
if max(rule_scores.values()) < 0.2 and max(embed_scores.values()) < 0.35:
    return SAFE_REPLY("I might not be the right assistant...")
```
**Condition**: All signals are weak (likely out-of-scope)

#### 4. OOS Probability Gate
```python
if p_oos_fused >= 0.5 and best_score < tau:
    return SAFE_REPLY("This seems outside my scope...")
```
**Condition**: High OOS probability but low routing confidence

#### 5. Default Fallback
```python
return SAFE_REPLY("I'm not fully sure what you mean...")
```
**Condition**: All other gates failed (confused state)

## Code Walkthrough

### Core Router Function (`calibrated_decision`)

```python
def calibrated_decision(msg: str, catalog_path: str | Path):
    # 1. Normalize input
    raw_msg = msg
    msg = normalize_text(msg)
    
    # 2. Load configuration
    catalog = load_catalog(catalog_path)
    
    # 3. Rule-based scoring
    rules = build_rules(catalog)
    rule_score = score_rules(msg, rules)
    
    # 4. Embedding similarity
    embedder, index, app_ids, embs = get_embedder_and_index(catalog)
    embed_score = score_embed(msg, embedder, embs, app_ids)
    
    # 5. Classifier scoring
    cls = _load_classifier()
    cls_scores = {aid: 0.0 for aid in app_ids}
    p_out = 0.0
    if cls is not None:
        pipe = cls["pipeline"]
        proba = pipe.predict_proba([raw_msg])[0]
        # ... extract probabilities
    
    # 6. OOS head scoring
    p_oos_head = 0.0
    oos_head = _load_oos_head()
    if oos_head is not None:
        # ... get binary OOS probability
    
    # 7. LLM arbitration
    rd = llm_arbitrate(raw_msg, catalog, rule_score, embed_score)
    
    # 8. Score fusion
    combined = {}
    for aid in app_ids:
        combined[aid] = max(
            rule_score.get(aid, 0.0),
            0.7 * cls_scores.get(aid, 0.0) + 0.3 * embed_score.get(aid, 0.0),
            0.6 * (rd.confidence if rd.app_id == aid and rd.route == "ROUTE" else 0.0) + 0.4 * embed_score.get(aid, 0.0),
        )
    
    # 9. Margin calculation
    sorted_apps = sorted(app_ids, key=lambda a: combined[a], reverse=True)
    best_app = sorted_apps[0]
    best_score = combined[best_app]
    second_score = combined[sorted_apps[1]] if len(sorted_apps) > 1 else 0.0
    margin = best_score - second_score
    
    # 10. Decision gates
    # ... apply gates in priority order
```

### Caching Strategy
The system uses intelligent caching to avoid reloading heavy models:

```python
# Global caches
_RULES_CACHE: dict[str, Dict[str, Tuple[re.Pattern, Optional[re.Pattern]]]] = {}
_EMBED_CACHE: dict[str, tuple] = {}
_CLS_CACHE: Optional[dict] = None
_OOS_CACHE: Optional[dict] = None
_THRESHOLDS: Optional[dict] = None
```

**Cache keys**:
- **Rules**: Based on catalog fingerprint (MD5 of relevant catalog parts)
- **Embeddings**: Based on model name + catalog fingerprint
- **Models**: Simple existence check

## Training and Model Creation

### Synthetic Data Generation (`scripts/generate_synth_data.py`)

The system generates thousands of training examples with realistic noise:

#### Template-Based Generation
```python
ABSENCE_TPL = [
    "who was absent {when}",
    "show absences {when}",
    "who is on leave {when}",
    # ... more templates
]

DATE_PHRASES = [
    "today", "yesterday", "tomorrow", "on Monday", "last week",
    "between 1st and 7th", "on 2025-09-01"
    # ... more date expressions
]
```

#### Noise Injection
1. **Typos**: Character swapping with 15% probability
2. **Code-switching**: Hindi/English mixing (e.g., "absent" → "chutti")
3. **Noise**: Random prefixes/suffixes, case changes, emojis

#### Example Generated Data
```csv
text,label
YO PUT SARA ON LEAVE TOMORROW,absence
DRAFT HLD FOR DATA PIPELINE,tdo_drafting
yo what is the budget for data pipeline,cost_estimation
logout please,OUT_OF_SCOPE
```

### Classifier Training (`scripts/train_classifier.py`)

#### Pipeline Architecture
```python
pipe = Pipeline([
    ("emb", EmbeddingFeaturizer(model_name=model_name, normalize=True)),
    ("clf", CalibratedClassifierCV(
        estimator=LogisticRegression(max_iter=2000, multi_class='multinomial'),
        cv=3,
        method='sigmoid'
    )),
])
```

#### Threshold Tuning
The system automatically tunes tau and delta thresholds:

```python
def tune_thresholds(labels, probas, class_names):
    tau_grid = np.linspace(0.5, 0.9, 9)
    delta_grid = np.linspace(0.0, 0.2, 5)
    
    for tau in tau_grid:
        for delta in delta_grid:
            # Simulate routing decisions
            y_pred = []
            for p in probas:
                top = int(np.argmax(p))
                margin = p[top] - p[second_best]
                if p[top] >= tau and margin >= delta:
                    y_pred.append(class_names[top])
                else:
                    y_pred.append("OUT_OF_SCOPE")
            
            # Evaluate performance
            macro_f1 = f1_score(y_true, y_pred, average='macro')
            # ... select best thresholds
```

## Configuration Files

### Application Catalog (`apps.yaml`)
Defines the applications, their keywords, and intent examples:

```yaml
apps:
  - id: "absence"
    display_name: "Absence Management"
    description: "View team absences, mark employees absent/present, date-range summaries."
    keywords: ["absent", "leave", "on leave", "who was out", "attendance", "PTO", "paid time off", "holiday"]
    neg_keywords: ["leave the page", "left join", "leave it", "leave me alone", "maternity leave policy pdf", "absent minded"]
    intents:
      - name: "get_absences"
        examples: ["who was absent last week", "show yesterday's absences", "who is on leave today"]
        entities:
          - name: "employee"
            type: "PERSON"
            required: false
          - name: "date_range"
            type: "DATE_RANGE"
            required: false
      - name: "mark_absent"
        examples: ["mark Manju absent for today", "mark Rahul on leave tomorrow"]
        entities:
          - name: "employee"
            type: "PERSON"
            required: true
          - name: "date"
            type: "DATE"
            required: true
```

### Router Thresholds (`config/router_thresholds.json`)
Automatically tuned thresholds for optimal performance:

```json
{
  "tau": 0.5,                    // Minimum confidence to route (50%)
  "delta": 0.0,                  // Minimum margin (no margin required)
  "oos_tau": 0.08250506950022037, // OOS threshold (8.25%)
  "oos_delta": 0.15              // OOS margin (15%)
}
```

### Model Files
- **`models/router.joblib`**: Main classifier with calibration
- **`models/oos_head.joblib`**: Binary out-of-scope detector

## Real Examples with Complete Calculations

### Example: Complex Absence Query

**Input**: `"Hey, can you tell me who all were absent yesterday? Thanks!"`

#### Step-by-Step Analysis

**1. Text Normalization**
- Raw: `"Hey, can you tell me who all were absent yesterday? Thanks!"`
- Normalized: `"hey, can you tell me who all were absent yesterday? thanks!"`

**2. Rule Scoring**
- **absence**: Matches "absent" → **0.950000**
- **tdo_drafting**: No match → **0.000000**
- **cost_estimation**: No match → **0.000000**

**3. Embedding Scoring** (hypothetical values)
- **absence**: High similarity → **0.742**
- **tdo_drafting**: Low similarity → **0.089**
- **cost_estimation**: Low similarity → **0.156**

**4. Classifier Scoring** (hypothetical values)
- **absence**: **0.891**
- **tdo_drafting**: **0.067**
- **cost_estimation**: **0.034**
- **OUT_OF_SCOPE**: **0.008**

**5. Score Fusion**

**absence**:
- Rule: **0.950000**
- Cls+Emb: 0.7 × 0.891 + 0.3 × 0.742 = 0.6237 + 0.2226 = **0.8463**
- LLM+Emb: 0.6 × 0.950 + 0.4 × 0.742 = 0.570 + 0.2968 = **0.8668**
- **Combined**: max(0.950, 0.8463, 0.8668) = **0.950000**

**tdo_drafting**:
- Rule: **0.000000**
- Cls+Emb: 0.7 × 0.067 + 0.3 × 0.089 = 0.0469 + 0.0267 = **0.0736**
- LLM+Emb: 0.6 × 0.000 + 0.4 × 0.089 = 0.000 + 0.0356 = **0.0356**
- **Combined**: max(0.000, 0.0736, 0.0356) = **0.0736**

**cost_estimation**:
- Rule: **0.000000**
- Cls+Emb: 0.7 × 0.034 + 0.3 × 0.156 = 0.0238 + 0.0468 = **0.0706**
- LLM+Emb: 0.6 × 0.000 + 0.4 × 0.156 = 0.000 + 0.0624 = **0.0624**
- **Combined**: max(0.000, 0.0706, 0.0624) = **0.0706**

**6. Final Decision**
- **Best**: absence (0.950000)
- **Second**: tdo_drafting (0.0736)
- **Margin**: 0.950000 - 0.0736 = **0.8764**
- **Decision**: ✅ **ROUTED to absence** (0.950 ≥ 0.5 AND 0.8764 ≥ 0.0)

### Example: Ambiguous Cost/TDO Query

**Input**: `"I need to design and estimate costs for a new microservice"`

This example shows how the system handles ambiguous queries that could match multiple apps.

#### Expected Behavior
1. **Rule scoring**: Both "design" (TDO) and "estimate costs" (Cost) keywords match
2. **Embedding scoring**: High similarity to both TDO and Cost corpora
3. **Classifier**: Likely confused between the two
4. **LLM arbitration**: Would be called to make intelligent decision
5. **Final decision**: Depends on which signal is strongest after fusion

## Performance Characteristics

### Accuracy Metrics
- **High precision**: Rule-based component ensures accurate keyword matching
- **High recall**: Embedding similarity catches semantic variations
- **Robust OOS detection**: Multiple OOS signals prevent false positives
- **Calibrated confidence**: Probability calibration provides reliable confidence scores

### Speed Optimization
- **Caching**: Models and embeddings cached in memory
- **LLM gating**: Expensive LLM calls only when needed
- **FAISS indexing**: Fast similarity search for embeddings
- **Compiled regex**: Fast rule-based matching

### Scalability
- **Stateless**: No session state, fully stateless routing
- **Configurable**: Easy to add new apps via YAML configuration
- **Model updates**: Models can be retrained and swapped without code changes
- **Threshold tuning**: Automatic threshold optimization

## Conclusion

This hybrid routing system combines the best of multiple AI approaches:

1. **Rules** provide fast, deterministic matching for clear cases
2. **Embeddings** handle semantic similarity and variations
3. **Classifiers** learn complex patterns from training data
4. **LLM arbitration** handles ambiguous cases with reasoning
5. **Multiple OOS detectors** prevent routing errors
6. **Calibrated probabilities** provide reliable confidence estimates
7. **Margin gating** ensures high-confidence decisions only

The mathematical fusion strategy using max-pooling allows each component to contribute its strengths while providing robustness against individual component failures. The multi-gate decision logic ensures that only high-confidence routing decisions are made, with graceful fallbacks to safe replies when uncertain.

This architecture achieves both high accuracy and high reliability, making it suitable for production enterprise chatbot routing scenarios.