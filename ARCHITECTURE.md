# Delhi AQI Intelligence Platform — Full Architecture & Technical Report

> A complete end-to-end explanation of how the system was built: every component,
> every design decision, and how data flows from raw sensor readings to the AI-generated
> analysis that reaches the user's screen.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture Diagram](#2-high-level-architecture-diagram)
3. [Component 1 — Document Ingestion Pipeline](#3-component-1--document-ingestion-pipeline)
4. [Component 2 — Real-Time Data Layer (OpenAQ v3)](#4-component-2--real-time-data-layer-openaq-v3)
5. [Component 3 — Vector Store & Hybrid Retrieval Index](#5-component-3--vector-store--hybrid-retrieval-index)
6. [Component 4 — RAG Retrieval Pipeline](#6-component-4--rag-retrieval-pipeline)
7. [Component 5 — LLM Integration (Groq / Llama 3.3 70B)](#7-component-5--llm-integration-groq--llama-33-70b)
8. [Component 6 — ML Forecasting Pipeline](#8-component-6--ml-forecasting-pipeline)
9. [Component 7 — Insights Engine](#9-component-7--insights-engine)
10. [Component 8 — Visualization Layer](#10-component-8--visualization-layer)
11. [Component 9 — Streamlit UI](#11-component-9--streamlit-ui)
12. [End-to-End Data Flow](#12-end-to-end-data-flow)
13. [Technology Stack Summary](#13-technology-stack-summary)
14. [Performance Metrics](#14-performance-metrics)
15. [Design Decisions & Trade-offs](#15-design-decisions--trade-offs)

---

## 1. System Overview

The **Delhi AQI Intelligence Platform** is a production-quality, AI-powered air-quality
analytics dashboard. It fuses four independent sources of intelligence:

| Source | What it provides |
|---|---|
| **OpenAQ v3 API** | Live sensor readings from 15+ monitoring stations across Delhi/NCR |
| **RAG Knowledge Base** | 232 semantically-indexed chunks from research papers, health guidelines, policy docs |
| **ML Forecasting Model** | 6-hour recursive LightGBM (regime-aware) trained on 43,461 hourly records |
| **LLM Analyst** | Llama 3.3 70B via Groq — contextualised, hallucination-guarded AI responses |

The system is built around a core **Python/Streamlit** stack with no external ML services
required (the LLM call is optional — the dashboard remains fully functional as a monitoring
and forecasting tool without it).

### Design Philosophy

- **Graceful degradation**: every API call has a synthetic fallback so the app always runs.
- **Strict causality**: the forecasting feature pipeline never leaks future information.
- **Production RAG quality**: hybrid retrieval (dense + BM25) → cross-encoder reranking →
  hallucination detection → confidence scoring.
- **Observatory-grade accuracy**: India NAQI sub-index breakpoints, unit conversion for
  every sensor output format encountered in the wild.

---

## 2. High-Level Architecture Diagram

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                              DELHI AQI INTELLIGENCE PLATFORM                              │
│                                   (app.py — Streamlit)                                    │
└──────────────┬────────────────────┬──────────────────────┬────────────────────────────────┘
               │                    │                      │
     ┌─────────▼──────────┐  ┌──────▼──────────┐  ┌──────▼──────────────────┐
     │  REAL-TIME DATA    │  │  ML FORECASTING  │  │   RAG AI ANALYST        │
     │  LAYER             │  │  PIPELINE        │  │   (Optional LLM)        │
     │  api/openaq_client │  │  forecasting/    │  │   rag/                  │
     └─────────┬──────────┘  └──────┬──────────┘  └──────┬──────────────────┘
               │                    │                      │
     ┌─────────▼──────────┐  ┌──────▼──────────┐  ┌──────▼──────────────────┐
     │  OpenAQ v3 API     │  │  LightGBM        │  │  Vector Store           │
     │  /locations        │  │  (regime-aware)  │  │  embeddings/            │
     │  /sensors/{id}/    │  │  +SHAP explain   │  │  FAISS + BM25           │
     │    measurements    │  │                  │  │  232 chunks / 768-dim   │
     │  [+ demo fallback] │  └──────────────────┘  └─────────────────────────┘
     └────────────────────┘
               │
     ┌─────────▼──────────────────────────────────┐
     │   INSIGHTS ENGINE                          │
     │   insights/engine.py                       │
     │   Trend alerts · Health context · Station  │
     │   comparisons · Forecast notes             │
     └────────────────────────────────────────────┘
               │
     ┌─────────▼──────────────────────────────────┐
     │   VISUALIZATION LAYER                      │
     │   visualization/plots.py                   │
     │   India NAQI AQI computation               │
     │   Plotly: gauge · sub-index · timeseries   │
     │   radar · WHO comparison · forecast bands  │
     └────────────────────────────────────────────┘
```

---

## 3. Component 1 — Document Ingestion Pipeline

**Location**: `ingestion/` + `embeddings/`

### Purpose
Convert raw research PDFs and text files into a searchable, semantically-rich vector
index that the RAG retriever can query at sub-second latency.

### Sub-components

#### 3.1 Document Loader (`ingestion/load_pdfs.py`)

Uses **pdfplumber** to extract text page-by-page from PDFs. Each page is returned
as a dict carrying `doc_name`, `page`, `text`, and `source_type`.

If pdfplumber fails (e.g. non-PDF files), the loader falls back to plain UTF-8 text
reading — ensuring `.txt` research summaries load without any special handling.

**Knowledge base contents** (10 curated documents):
- Delhi AQI comprehensive guide
- Health impacts of Delhi pollution
- Delhi geography & meteorology
- Historical AQI data (2010–2024)
- AQI methodology comparison (India NAQI vs US EPA vs WHO)
- Pollution sources & policy
- Stubble burning & industrial transport sources
- Global air quality research
- Outdoor activities & exercise guide
- Additional research papers

#### 3.2 Semantic Chunker (`ingestion/chunk_docs.py`)

Rather than splitting on a fixed character count (the naive approach), the chunker uses a
**semantic breakpoint detection** algorithm:

```
Full document text
      │
      ▼
 Sentence splitter (regex: punctuation + whitespace / double newline)
      │
      ▼
 Sentence embeddings (all-MiniLM-L6-v2, 384-dim)
      │
      ▼
 Cosine similarity between consecutive sentences
      │
      ▼
 Mark breakpoints where similarity < threshold (0.45)
      │
      ▼
 Merge sentences within each segment (max 512 tokens)
      │
 Add 200-token overlap between adjacent chunks
      │
      ▼
 Final semantic chunks (avg ~350 tokens each)
```

**Fallback**: when `sentence-transformers` is unavailable, the chunker falls back to a
simpler paragraph-aware splitter that respects the 512-token ceiling.

Each chunk is enriched with metadata:

| Metadata field | How it is derived |
|---|---|
| `topic` | Keyword scoring across 7 categories (health, policy, meteorology, …) |
| `year` | Regex extraction of the most-frequent 4-digit year in the chunk |
| `credibility_score` | Rule-based: research/journal/WHO sources → 0.92–0.95; plain text → 0.75 |
| `token_count` | `words × 1.3` approximation |

#### 3.3 Embedding Builder (`ingestion/embed_docs.py`)

Primary model: **BAAI/bge-base-en-v1.5** (768-dimensional dense embeddings).

BGE (Beijing Academy of Artificial Intelligence General Embedding) was chosen over the
older `all-MiniLM-L6-v2` for its superior retrieval performance on English text.

Fallback: **TF-IDF** (768 max features via scikit-learn) — used automatically if
`sentence-transformers` is not installed, enabling the system to run in minimal
environments.

The embedding pipeline:
1. Loads all pages from `data/pdfs/`
2. Chunks semantically
3. Encodes in batch (progress bar shown)
4. Stores FAISS index + metadata + BM25 index + embed info JSON

**Result**: 232 chunks, 768-dimensional vectors, saved to `embeddings/vector_store/`.

#### 3.4 Incremental Ingestion (`ingestion/incremental_ingest.py`)

New documents can be added **without rebuilding the entire index**:

```python
add_documents("data/pdfs/new_docs/")           # scans for new files only
add_text_snippet("Delhi AQI hit 480 today", "news_snippet")  # single snippet
```

The incremental ingestor tracks already-indexed `doc_name` values, embeds only new
content, and appends it to the existing FAISS index + BM25 index.

---

## 4. Component 2 — Real-Time Data Layer (OpenAQ v3)

**Location**: `api/openaq_client.py`

### OpenAQ v3 API Design

OpenAQ v3 changed significantly from v2. Key differences the client handles:

- `/locations/{id}/latest` returns `{sensorsId, value, datetime}` but **no parameter name**
- Parameter names must be looked up from the `/locations` response's `sensors` list
- The client builds a `sensorId → {parameter, units}` map per location before fetching readings

### Functions

| Function | Purpose |
|---|---|
| `list_locations()` | Returns active monitoring stations within 25 km of Delhi India Gate (28.6139°N, 77.2090°E), filtered to those with data in the last 30 days |
| `get_latest_city_measurements()` | Most-recent reading per parameter per station — deduplicates by keeping newest reading |
| `get_historical_city()` | Daily aggregates via `/sensors/{id}/days` for a given date range |
| `get_hourly_data()` | Raw measurements via `/sensors/{id}/measurements`, aggregated into hourly averages — the forecasting feed |
| `list_city_stations()` | Simplified station list for UI selectors |
| `get_station_latest()` | All pollutants for a single station by location ID |

### API Rate Limiting

The client enforces a `max_calls = 50` guard on the hourly data fetch to avoid
exhausting the free-tier API budget. Sensors per parameter are deduplicated — only the
newest sensor ID per parameter is queried (old sensors with low IDs are often defunct).

### Synthetic Fallback Data

When no API key is available (or the API is unreachable), `_demo_hourly()` generates
**realistic synthetic data** anchored to the current live readings if any are available:

```
synthetic_value(t) = base + noise(±8%) + diurnal_pattern(t) + shift_to_anchor_last_value
```

The diurnal pattern simulates Delhi's characteristic dual peaks:
- **Morning rush** (08:00–10:00 IST) — traffic emissions
- **Evening peak** (19:00–21:00 IST) — traffic + atmospheric boundary layer collapse

This ensures the ML forecast remains consistent with whatever pollutant values are
displayed in the live monitoring section.

### Unit Handling

OpenAQ sensors report CO in wildly inconsistent units. The conversion logic:

```
CO "ppb" with value < 100  →  treat as ppm  →  × 1.145 mg/m³
CO "ppb" with value ≥ 100  →  genuine ppb   →  × PPB_TO_UGM3["co"] mg/m³
CO "ppm"                   →  × 1.145 mg/m³
CO µg/m³                   →  ÷ 1000 mg/m³
```

This handles the real-world OpenAQ data quality issues discovered during development.

---

## 5. Component 3 — Vector Store & Hybrid Retrieval Index

**Location**: `embeddings/vector_store.py`, `embeddings/cache.py`

### VectorStore Class

Supports two backends transparently:

| Backend | When used | Storage |
|---|---|---|
| **FAISS** (`faiss-cpu`) | When faiss is installed (default) | `index.faiss` |
| **scikit-learn NearestNeighbors** | When faiss is unavailable | `embeddings.npy` |

The API is identical regardless of backend — the switch is handled internally.

### BM25 Keyword Index

A **pure-Python Okapi BM25** implementation (no external dependency) runs in parallel
with the dense FAISS index for sparse keyword matching:

```
BM25 score(query, doc) = Σ IDF(t) × tf(t,d) × (k1+1) / (tf(t,d) + k1·(1−b + b·dl/avgdl))
```

Parameters: k1=1.5, b=0.75 (standard BM25 settings).
The BM25 model is fitted on all chunk texts at build time and saved as `bm25.pkl`.

### Hybrid Search via Reciprocal Rank Fusion (RRF)

The hybrid search combines dense and sparse results using RRF:

```
RRF_score(doc) = dense_weight / (k + rank_dense + 1)
               + sparse_weight / (k + rank_sparse + 1)

defaults: dense_weight=0.6, sparse_weight=0.4, k=60
```

RRF is preferred over direct score combination because it is robust to scale
differences between the two scoring systems.

### Embedding Cache (`embeddings/cache.py`)

A thread-safe **LRU embedding cache** prevents re-encoding the same query twice:

- Keyed by SHA-256 hash of the query text (truncated to 24 chars)
- Capacity: 4096 entries
- Persisted to `embed_cache.npz` between restarts (`.npz` compressed NumPy format)
- Tracks hit/miss statistics

---

## 6. Component 4 — RAG Retrieval Pipeline

**Location**: `rag/retriever.py`, `rag/reranker.py`, `rag/query_classifier.py`

### Full Retrieval Pipeline (4 stages)

```
User question
      │
      ▼
┌─────────────────────────────────────┐
│  Stage 1: Query Intent Classification│
│  rag/query_classifier.py            │
│  7 intent types (live_data, health, │
│  historical, comparison, factual,   │
│  policy, general)                   │
│  → top_k, needs_api, preferred_topics│
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Stage 2: Hybrid Retrieval          │
│  rag/retriever.py                   │
│  BGE query embedding (768-dim)      │
│  with BGE query prefix              │
│  FAISS dense search (top-20)        │
│  BM25 sparse search (top-20)        │
│  RRF fusion → 20 candidates         │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Stage 3: Cross-Encoder Reranking   │
│  rag/reranker.py                    │
│  ms-marco-MiniLM-L-6-v2             │
│  CE score → sigmoid normalisation   │
│  Weighted with credibility_score    │
│  final_score = 0.85·CE + 0.15·cred  │
│  Adaptive top-k (score gap elbow)   │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Stage 4: Topic Boost (optional)    │
│  If intent has preferred_topics     │
│  matching chunks boosted ×1.3       │
└─────────────┬───────────────────────┘
              │
              ▼
       Top-k chunks (dict with text,
       doc_name, topic, final_score,
       credibility_score, year, …)
```

### Query Intent Classifier (`rag/query_classifier.py`)

The classifier uses **regex keyword scoring** (no ML model required — fast and
interpretable):

```python
INTENT_KEYWORDS = {
    "live_data":     [r"current|right now|today|live|real.?time|latest"],
    "health_advice": [r"safe|health|jog|run|exercise|outdoor|mask|…"],
    "historical":    [r"history|trend|past|year|season|winter|…"],
    "comparison":    [r"compar|versus|vs\.?|differ|who|standard|…"],
    "factual":       [r"what is|define|explain|how does|why|cause|…"],
    "policy":        [r"polic|government|grap|cpcb|ban|stubble|…"],
}
```

Each intent maps to a retrieval config: `{needs_api, needs_rag, top_k, preferred_topics}`.

### BGE Query Prefix

The BAAI/bge-base-en-v1.5 model is asymmetric — queries and documents are encoded
differently:

```
Query embedding:    BGE_QUERY_PREFIX + query_text
Document embedding: document_text (no prefix)
```

This asymmetric design allows the query to "seek" relevant passages even when the
vocabulary does not overlap.

---

## 7. Component 5 — LLM Integration (Groq / Llama 3.3 70B)

**Location**: `rag/llm_pipeline.py`, `rag/prompt_template.py`, `rag/memory.py`,
`rag/hallucination.py`, `rag/confidence.py`, `rag/tool_calling.py`

### LLM Choice

**Llama 3.3 70B** via the **Groq** inference API. Groq uses custom LPU (Language
Processing Unit) hardware for extremely low latency — typical response times of 1–3 s
for 1500-token responses.

The system message (`SYSTEM_MESSAGE` in `llm_pipeline.py`) establishes the analyst
persona with deep domain knowledge:
- India NAQI calculation methodology
- Delhi pollution sources (vehicular 40–50% of NOx, stubble burning, industrial)
- Seasonal patterns (winter inversion, monsoon washout, post-Diwali spike)
- Health science (PM2.5 alveolar penetration, life expectancy reduction)
- Policy context (GRAP stages I–IV, NCAP, BS-VI)
- WHO guidelines vs actual Delhi levels

### Prompt Architecture (`rag/prompt_template.py`)

The user message is assembled in layers:

```
1. "**My question:** {question}"

2. "[Query intent: {intent} (conf={confidence})]"  ← from classifier

3. "**[Live API Data — …]**"                       ← tool_calling result
   OR
   "**Context — Live AQI readings …**"             ← cached snapshot

4. "**Context — Knowledge-base excerpts** …"       ← RAG chunks
   "[Ref 1] (doc_name | topic | relevance=0.82): chunk_text"
   "[Ref 2] …"

5. "**How to answer:** …"                          ← lightweight guidance
```

This "question-first" prompt design is deliberate — LLMs tend to anchor on the first
tokens of a prompt, so the actual question is placed before context to reduce the risk
of the model generating a generic AQI report instead of answering what was asked.

### Tool Calling (`rag/tool_calling.py`)

Two tools are defined:

| Tool | Action |
|---|---|
| `get_current_aqi` | Fetches live sensor data → computes AQI → formats as context |
| `get_historical_trend` | Fetches N-day historical data → summarises per-pollutant stats |

`auto_tool_call()` checks query intent and either uses the session's cached snapshot
(most common path) or fetches fresh API data.

### Hallucination Detection (`rag/hallucination.py`)

After the LLM generates a response, the hallucination detector checks:

1. **Citation presence**: were `[Ref N]` citations used when RAG chunks were provided?
2. **Unsourced numeric claims**: >3 large numbers without citations → elevated risk
3. **Hedging language**: "I think", "probably", "maybe" etc. counted
4. **Source coverage**: if ≥3 relevant chunks existed but none cited → `"high"` risk

If `hallucination_risk == "high"` and RAG context was provided, the pipeline
**regenerates the response** with an explicit instruction to cite every factual claim.

### Confidence Scoring (`rag/confidence.py`)

A composite 0–1 confidence score is computed from 4 components:

| Component | Weight (typical) | How measured |
|---|---|---|
| Source quality | 30% | Reranker scores + credibility, exponentially weighted |
| Citation coverage | 25% | `[Ref N]` and URL citations ÷ number of sources |
| Relevance | 25% | Keyword overlap between query and response (stop-words removed) |
| Response quality | 20% | Length, markdown structure, numeric specificity |

Grades: **high** (≥70%), **medium** (40–69%), **low** (<40%).

### Conversation Memory (`rag/memory.py`)

Multi-turn conversation is managed by `ConversationMemory`:

- Keeps the most recent 6 messages verbatim
- When total conversation exceeds ~6000 tokens, older messages are **summarised** by the
  LLM itself into a 3–5 sentence block
- The summary is injected as a system message before the recent history
- Fallback extractive summarisation is used if the LLM is unavailable

---

## 8. Component 6 — ML Forecasting Pipeline

**Location**: `forecasting/train.py`, `forecasting/features.py`, `forecasting/inference.py`

### Data

Training data: Delhi hourly AQI records from 2020 to 2024 (~43,461 rows after cleaning).

**Time-based train/val/test split** (strict temporal order, no shuffling):
- Train: 2020–2022 (26,072 rows)
- Val: 2023 (8,605 rows)
- Test: 2024 (8,784 rows)

### Feature Engineering (`forecasting/features.py`)

58 features in total, all strictly causal (computed only from past data):

#### AQI Lag Features
```
aqi_lag{1,2,3,6,12,24,168}   ← 1h to 7 days back
```

#### Pollutant Lag Features
```
{pm25,pm10,no2,so2,nh3,co,o3,no,nox}_lag{1,3}
```

#### Rolling Statistics
```
aqi_rmean{3,6,24}    ← rolling mean (shift-1 to avoid leakage)
aqi_rstd{3,6,24}     ← rolling std
```

#### Rate-of-Change Features
```
aqi_delta{1,3,6}     ← diff over N hours
aqi_accel            ← second-order delta (delta of delta1)
```

#### Temporal / Calendar Features
```
hour, dow, month, is_weekend
hour_sin, hour_cos        ← cyclical encoding (2π×h/24)
dow_sin, dow_cos          ← cyclical encoding (2π×dow/7)
month_sin, month_cos      ← cyclical encoding (2π×m/12)
```

#### Meteorological Features
```
temperature, humidity, wind_speed, wind_dir
pressure, rainfall, solar_rad
wind_u = wind_speed × sin(wind_dir_rad)   ← vector decomposition
wind_v = wind_speed × cos(wind_dir_rad)
temp_hum_interaction = temperature × humidity / 100
solar_temp = solar_rad × temperature / 1000
```

#### Missing Data Handling
Small gaps (≤6 h) are interpolated linearly; larger gaps remain NaN and those rows
are dropped during feature creation. Outliers are winsorized at the 0.1–99.9th
percentile.

### Pollution Regime Clustering

The training pipeline uses **KMeans (k=3)** to identify distinct pollution regimes in the
historical data:

```
Daily aggregation:
  aqi_mean, aqi_std, aqi_max, aqi_min, aqi_range
  pm25_mean, pm10_mean
  temperature_mean, humidity_mean, wind_speed_mean

StandardScaler → KMeans(n_clusters=3)

Regime 0: 12,984 training hours  (likely clean / moderate days)
Regime 1:  7,808 training hours  (likely winter smog episodes)
Regime 2:  5,280 training hours  (likely extreme pollution events)
```

Daily regime labels are forward-filled to hourly resolution.

### Models Trained

| Model | Description |
|---|---|
| **Persistence baseline** | AQI(t) = AQI(t-1) — the simplest possible predictor |
| **LightGBM Global** | Single GBDT on all training data |
| **XGBoost Global** | XGBoost baseline with same hyperparameters |
| **LightGBM Regime** | Separate LightGBM per regime, fallback to global if too few samples |

**LightGBM hyperparameters**:
```python
num_leaves=127, learning_rate=0.05,
feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
n_estimators=2000 (with early_stopping=50)
```

### Model Selection

Best model is selected by 1-step MAE on the test set. The **LightGBM Regime** model
was selected as best (MAE = 0.49 AQI units on 1-step prediction).

### SHAP Interpretability

After training, SHAP TreeExplainer is run on a 500-sample subset of the test set to
produce:
- `shap_importance.png` — global feature importance bar chart
- `shap_beeswarm.png` — beeswarm plot showing direction of feature effects
- `shap_importance.json` — mean absolute SHAP values per feature (machine-readable)

The top features are invariably the recent AQI lags (`aqi_lag1`, `aqi_lag2`, `aqi_lag3`)
followed by `aqi_rmean3` and the pollutant concentrations.

### Recursive 6-Hour Inference (`forecasting/inference.py`)

At runtime, the inference module:

1. Receives live hourly data (OpenAQ long-format DataFrame)
2. **Pivots** long → wide format (one column per pollutant)
3. Computes per-hour India NAQI AQI from the pollutant columns
4. Builds all 58 features using the same pipeline as training
5. Runs a **recursive forecast loop**:

```
For each step h = 1 … 6:
   X_input = last row of feature matrix
   pred[h] = model.predict(X_input)
   Append new row with pred[h] as target value
   Update all lag/rolling/delta features for the new row
   Update temporal features for next timestamp
```

**Uncertainty bands** grow linearly with horizon:
```
lower = pred × (1 − (0.10 + 0.05×h))
upper = pred × (1 + (0.10 + 0.05×h))
```
→ ±15% at +1 h, ±40% at +6 h.

---

## 9. Component 7 — Insights Engine

**Location**: `insights/engine.py`

The insights engine generates **ranked, context-aware insights** from the current state
of the data. It runs 6 independent generators, each returning `Insight` objects tagged
with `category`, `severity`, `emoji`, `title`, `body`, and `priority` (0–100).

| Generator | Triggers | Example |
|---|---|---|
| `_trend_insights()` | ≥25% change in PM2.5/PM10/NO₂/O₃ over last 3 hours | "PM2.5 Spike Detected — rose 40% in 3 hours" |
| `_diurnal_insights()` | Time of day (IST) matches known pollution pattern windows | "Evening Pollution Peak — boundary layer collapse trapping pollutants" |
| `_comparison_insights()` | Any pollutant exceeds WHO 24-h guideline | "PM2.5: 12.3× WHO Limit — serious health risk" |
| `_forecast_insights()` | Model predicts >15% AQI increase in next hour | "AQI Expected to Worsen — from 285 to 330" |
| `_station_insights()` | Station spread ratio >2× for PM2.5 or PM10 | "PM2.5 Varies 2.8× Across Stations — hyperlocal factors" |
| `_health_insights()` | AQI > 100, > 150, > 200, > 300 thresholds | "Outdoor Exercise Advisory — avoid prolonged activity" |

All insights are sorted by priority (descending) and trimmed to a configurable maximum
(default: 6). The result is displayed in the UI as expandable cards.

---

## 10. Component 8 — Visualization Layer

**Location**: `visualization/plots.py` (1,178 lines)

### India NAQI AQI Computation

The AQI calculation follows the India National Air Quality Index standard:

```
For each pollutant p with concentration C:
  Find the breakpoint interval [Clo, Chi] that contains C
  Find corresponding AQI interval [Ilo, Ihi]
  Sub-index(p) = ((Ihi − Ilo) / (Chi − Clo)) × (C − Clo) + Ilo

AQI = max(Sub-index(p) for all pollutants)
Dominant pollutant = argmax(Sub-index)
```

Six-pollutant breakpoint tables for: PM2.5, PM10, NO₂, SO₂, CO, O₃.

AQI categories:

| AQI Range | Category | Colour |
|---|---|---|
| 0–50 | Good | `#009966` |
| 51–100 | Satisfactory | `#58B453` |
| 101–200 | Moderate | `#FFDE33` |
| 201–300 | Poor | `#FF9933` |
| 301–400 | Very Poor | `#CC0033` |
| 401–500 | Severe | `#7E0023` |

### Charts (all Plotly)

| Chart | Function | Description |
|---|---|---|
| AQI Gauge | `aqi_gauge()` | Speedometer with gradient sector bands + pointer |
| Sub-index breakdown | `sub_index_chart()` | Horizontal bars, colour-coded by AQI category |
| Pollutant vs WHO | `pollutant_vs_who()` | Grouped bars: measured vs guideline |
| Time-series | `timeseries_plot()` | Multi-pollutant line chart with AQI colour shading |
| Radar | `pollutant_radar()` | Spider chart for pollutant profile comparison |
| AQI scale bar | `aqi_scale_bar()` | Horizontal gradient band with marker |
| Station comparison | `station_comparison_chart()` | Grouped bars across stations |
| Station heatmap | `station_aqi_heatmap()` | Station × pollutant heatmap |
| Station detail | `station_detail_chart()` | Per-station pollutant bar chart |
| Forecast | `forecast_chart()` | Line + shaded uncertainty band for 6-h forecast |

All charts use a consistent dark-transparent theme:
- `paper_bgcolor="rgba(0,0,0,0)"` — transparent background (blends with UI cards)
- `font_family="Inter, sans-serif"`
- Hover labels: dark navy (`#1a1a2e`) with white text

---

## 11. Component 9 — Streamlit UI

**Location**: `app.py` (~1,500 lines)

### Page Architecture

The dashboard is a single-page Streamlit app with three conceptual sections:

```
┌── Navigation Bar ──────────────────────────────────────────────────────┐
│  🌫 Delhi AQI Intelligence  |  Live  |  Station  |  Forecast  |  AI   │
└────────────────────────────────────────────────────────────────────────┘
┌── Hero Banner ─────────────────────────────────────────────────────────┐
│            AQI: 285   POOR                                              │
│  Dominant: PM2.5  |  14 stations  |  Last updated: 14:32 IST           │
└────────────────────────────────────────────────────────────────────────┘
┌── Metric Cards ───────────────────────────────────────────────────────┐
│  PM2.5: 189 µg/m³  |  PM10: 295 µg/m³  |  NO₂: 48 µg/m³  | …       │
└───────────────────────────────────────────────────────────────────────┘
┌── Insights Panel ─────────────────────────────────────────────────────┐
│  [HIGH PRIORITY insights from insights engine]                         │
└───────────────────────────────────────────────────────────────────────┘
┌── Charts Row ─────────────────────────────────────────────────────────┐
│  AQI Gauge  |  Sub-index breakdown  |  Pollutant vs WHO               │
└───────────────────────────────────────────────────────────────────────┘
┌── Time-Series & Radar ────────────────────────────────────────────────┐
│  48-hour history  |  Pollutant radar                                   │
└───────────────────────────────────────────────────────────────────────┘
┌── Station Analysis ───────────────────────────────────────────────────┐
│  Station selector  |  Comparison chart  |  Heatmap                    │
└───────────────────────────────────────────────────────────────────────┘
┌── ML Forecast ────────────────────────────────────────────────────────┐
│  6-hour forecast chart with uncertainty bands                          │
│  Trend: Worsening / Stable / Improving                                 │
│  Confidence: High / Medium / Low                                       │
└───────────────────────────────────────────────────────────────────────┘
┌── AI Analyst Chat ────────────────────────────────────────────────────┐
│  Question input  |  Chat history                                       │
│  Confidence badge  |  Regeneration if hallucination detected           │
└───────────────────────────────────────────────────────────────────────┘
```

### Custom CSS

The app includes ~300 lines of custom CSS injected via `st.markdown(..., unsafe_allow_html=True)`:

- **Google Fonts**: Inter (UI) + JetBrains Mono (code/numbers)
- **Animations**: `fadeInUp`, `fadeIn`, `pulse`, `barGrow`, `shimmer`
- **Glassmorphism cards**: `backdrop-filter: blur(12px)` + translucent backgrounds
- **Live indicator**: animated pulse dot (CSS keyframe animation)
- **Responsive layout**: max-width 1280px with responsive column breakpoints

### Caching Strategy

Streamlit's `@st.cache_data` is used for:
- Live sensor data (TTL: 5 minutes)
- Hourly historical data (TTL: 30 minutes)
- Station list (TTL: 1 hour)

The ML model is loaded once at startup and cached at module level in `forecasting/inference.py`.

### Configuration (`/.streamlit/config.toml`)

```toml
[theme]
primaryColor = "#60a5fa"
backgroundColor = "#0f172a"
secondaryBackgroundColor = "#1e293b"
textColor = "#e2e8f0"
```

Dark navy theme throughout, consistent with the professional analytics aesthetic.

---

## 12. End-to-End Data Flow

### Flow A: Dashboard Loads / Refreshes

```
app.py startup
     │
     ├─► OpenAQ API → list_locations() → active stations (25 km radius)
     │         │
     │         └─► get_latest_city_measurements()
     │                   │
     │                   ├─► /locations/{id}/latest  ×15 stations
     │                   │   build sensorId→param map
     │                   │   parse {sensorsId, value, datetime}
     │                   └─► Demo fallback if API unavailable
     │
     ├─► compute_aqi(pollutant_vals)
     │         India NAQI sub-index calculation
     │         AQI = max(sub_indices)
     │
     ├─► generate_insights(aqi_val, pollutant_vals, df_latest, df_hourly, forecast)
     │         6 insight generators → ranked list
     │
     ├─► get_hourly_data() → 24h of hourly readings
     │         │
     │         └─► forecast_next_6_hours(df_hourly, current_aqi, pollutant_vals)
     │                   pivot long → wide
     │                   build 58 features
     │                   recursive 6-step prediction
     │                   uncertainty bands
     │
     └─► Render charts (Plotly), metric cards, insight panels
```

### Flow B: User Asks a Question (AI Analyst)

```
User types question
     │
     ▼
classify_query(question)
     │ intent + config (needs_api, needs_rag, top_k)
     │
     ├─[needs_api=True]──► auto_tool_call()
     │                           Use session snapshot or fetch fresh API data
     │                           → formatted context text
     │
     ├─[needs_rag=True]──► Retriever.retrieve(question, top_k)
     │                           1. embed_query() with BGE prefix
     │                              check EmbeddingCache
     │                           2. hybrid_search() → 20 candidates
     │                              FAISS dense (top-20) + BM25 sparse (top-20)
     │                              RRF fusion
     │                           3. rerank() with cross-encoder
     │                              ms-marco-MiniLM-L-6-v2
     │                              final_score = 0.85·CE_norm + 0.15·cred
     │                           4. return top-k chunks
     │
     ▼
build_prompt(question, chunks, snapshot, intent, tool_result)
     │ Assembles layered prompt (question → intent → live data → RAG → guidance)
     │
     ▼
memory.prepare_messages(chat_history)
     │ Keep recent 6 messages; summarise older ones
     │
     ▼
LLMPipeline.chat_with_guard(messages, query, chunks, intent)
     │ Call Groq API (Llama 3.3 70B)
     │ → response text
     │
     ▼
detect_hallucination(response, chunks, query)
     │ Check citations, numeric claims, hedging
     │ If high risk → regenerate with stricter prompt
     │
     ▼
compute_confidence(response, chunks, query, intent)
     │ 4-component score → grade (high/medium/low)
     │
     ▼
Display response + confidence badge in chat UI
```

---

## 13. Technology Stack Summary

| Layer | Technology | Version / Notes |
|---|---|---|
| **UI** | Streamlit | ≥1.30.0 |
| **Language** | Python | 3.11+ |
| **Real-time data** | OpenAQ v3 REST API | `requests`, JSON |
| **Embeddings** | BAAI/bge-base-en-v1.5 | sentence-transformers ≥3.0 |
| **Dense index** | FAISS (CPU) | `faiss-cpu` |
| **Sparse index** | BM25 (custom impl.) | No external dependency |
| **Cross-encoder** | ms-marco-MiniLM-L-6-v2 | sentence-transformers |
| **LLM** | Llama 3.3 70B | Groq API |
| **ML framework** | LightGBM + XGBoost | `lightgbm`, `xgboost` |
| **Interpretability** | SHAP TreeExplainer | `shap` |
| **Data processing** | pandas, numpy | standard |
| **Charts** | Plotly | ≥5.18.0 |
| **PDF parsing** | pdfplumber | — |
| **Model persistence** | joblib | — |
| **Deployment** | Streamlit Cloud | `runtime.txt`, no extra config |

---

## 14. Performance Metrics

### ML Forecasting (Test Set: 8,784 hours of 2024 data)

#### 1-Step Model Comparison

| Model | MAE | RMSE | R² | Spike MAE |
|---|---|---|---|---|
| Persistence (baseline) | 2.04 | 3.30 | 0.9992 | 2.09 |
| LightGBM Global | 0.53 | 0.92 | 0.9999 | 1.31 |
| XGBoost Global | 0.50 | 0.86 | 0.9999 | 1.22 |
| **LightGBM Regime** | **0.49** | **1.53** | **0.9998** | **0.66** |

*(Spike MAE = MAE on the top-10% highest AQI events)*

#### 6-Hour Recursive Forecast Performance (Best Model)

| Horizon | MAE | RMSE | R² |
|---|---|---|---|
| +1 h | 2.32 | 3.45 | 0.9991 |
| +2 h | 3.86 | 5.77 | 0.9974 |
| +3 h | 5.15 | 7.65 | 0.9954 |
| +4 h | 6.28 | 9.22 | 0.9934 |
| +5 h | 7.53 | 11.28 | 0.9901 |
| +6 h | 8.57 | 13.31 | 0.9862 |

MAE grows ~1.5 AQI units per additional forecast hour — a normal degradation for
recursive multi-step prediction. R² remains above 0.98 even at +6 h.

### Knowledge Base

| Metric | Value |
|---|---|
| Source documents | 10 |
| Total semantic chunks | 232 |
| Embedding dimension | 768 |
| Embedding model | BAAI/bge-base-en-v1.5 |
| Topic coverage | 7 categories |

### Training Data

| Split | Rows | Period |
|---|---|---|
| Train | 26,072 | 2020–2022 |
| Validation | 8,605 | 2023 |
| Test | 8,784 | 2024 |
| **Total** | **43,461** | **2020–2024** |
| Features | 58 | — |
| Pollution regimes | 3 | KMeans on daily profiles |

---

## 15. Design Decisions & Trade-offs

### Why LightGBM over deep learning (LSTM / Transformer)?

LightGBM regime models offer near-perfect R² on this dataset while being:
- **Instantaneous inference** (milliseconds vs seconds for neural models)
- **Interpretable** (SHAP values show exactly which features drive predictions)
- **Robust to missing meteorological features** at inference time (OpenAQ rarely provides all fields)
- **Deployable without GPU**

The AQI time series has very strong autocorrelation — lag-1 alone explains >99.9% of
variance — so the 168-hour lag window and rolling statistics capture the temporal
structure without recurrence.

### Why hybrid FAISS + BM25 over pure dense retrieval?

Dense-only retrieval struggles with **exact keyword queries** (e.g. "GRAP Stage III",
"PM2.5 AQI breakpoint 250"). BM25 handles these precisely. Conversely, BM25 fails on
semantic paraphrases ("what should I breathe?" → health advice). The hybrid approach
with RRF fusion consistently outperforms either alone.

### Why cross-encoder reranking?

Bi-encoder embeddings (BGE) optimise for fast approximate search but cannot compare a
query and document jointly. Cross-encoders (ms-marco-MiniLM-L-6-v2) attend to both
simultaneously, catching subtle relevance signals. The two-stage approach (fast
bi-encoder for recall, slow cross-encoder for precision) is the industry standard.

### Why graceful degradation everywhere?

Delhi monitoring data is unreliable — stations go offline, API rate limits are hit, the
Groq API may be unavailable. Every component has a well-tested fallback:
- No API key → realistic synthetic data anchored to last known readings
- No sentence-transformers → TF-IDF embeddings (lower quality but functional)
- No FAISS → scikit-learn NearestNeighbors (slower but correct)
- No cross-encoder → hybrid-score sorted results
- No Groq key → dashboard is fully functional (no AI chat, but all analytics work)

### Why Streamlit instead of React/Next.js?

The entire system is Python. Streamlit eliminates the context switch to JavaScript,
allows the same `pandas`/`plotly` objects to be rendered directly in the UI, and
deploys to Streamlit Cloud with zero configuration. For a data-science dashboard with
one primary view, this is the right trade-off.

---

_Generated from the live codebase. For questions about specific components, see the inline module docstrings or raise an issue._
