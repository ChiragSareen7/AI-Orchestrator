# Deterministic Platform — Multi Model

A self-contained, portable AI platform that runs multiple domain-specific language models and scores responses using lexical and semantic evaluation. Designed to showcase how non-deterministic AI outputs can be explored, evaluated, and ranked to produce the best possible answer.

---

## What This Project Does

Given a user query, the platform:

1. **Detects the domain** — chemistry, Python, Bhagavad Gita, or general
2. **Routes to the right model** — picks the primary model for the domain + uses past performance as a bias
3. **Generates 3 prompt variants** — concise, detailed, facts-only
4. **Runs all 4 models × 3 prompts** — up to 12 model calls per query
5. **Scores every response** — keyword overlap, Jaccard similarity, and optional semantic embeddings
6. **Ranks and returns the best** — sorted by accuracy, hallucination, latency
7. **Learns over time** — remembers which model performed best per domain

---

## Project Structure

```
deterministic_platform_multi_model/
│
├── start.sh              ← one-command startup for both services
├── README.md             ← this file
│
├── platform/             ← Multi Model Orchestration Platform (port 8020)
│   ├── run.py            ← entry point
│   ├── requirements.txt  ← Python dependencies
│   ├── store/
│   │   ├── logs.json           ← query logs (auto-populated)
│   │   └── model_performance.json  ← per-domain learning state
│   └── app/
│       ├── main.py             ← FastAPI app, routes
│       ├── services/
│       │   ├── orchestrator.py     ← core pipeline loop
│       │   ├── query_analyzer.py   ← domain/complexity/intent detection
│       │   ├── model_router.py     ← routing + learning bias
│       │   ├── prompt_generator.py ← 3 prompt variants
│       │   ├── model_executor.py   ← HTTP calls to models + Groq
│       │   ├── evaluator.py        ← lexical + semantic scoring
│       │   ├── ranker.py           ← sorts responses by score
│       │   ├── learning.py         ← persists model performance
│       │   ├── logger.py           ← JSON log writer
│       │   └── tracing.py          ← optional LangSmith tracing
│       ├── evaluation/
│       │   ├── pipeline.py         ← semantic evaluation orchestrator
│       │   ├── embeddings.py       ← sentence-transformer cosine similarity
│       │   ├── grounding.py        ← context grounding check
│       │   ├── nli.py              ← Natural Language Inference (optional)
│       │   ├── llm_judge.py        ← LLM-as-judge evaluation (optional)
│       │   ├── config.py           ← EvaluationThresholds (env-driven)
│       │   └── logging_hooks.py    ← eval event hooks
│       ├── utils/
│       │   ├── similarity.py       ← Jaccard, keyword overlap, pairwise
│       │   └── token_estimator.py  ← token count proxy for cost
│       └── static/
│           ├── index.html          ← browser dashboard UI
│           ├── app.js              ← dashboard JS (calls /query, /report)
│           └── styles.css          ← dashboard styles
│
└── models_api/           ← Fine-Tuned Models API (port 8010)
    ├── requirements.txt  ← Python dependencies
    ├── api/
    │   └── main.py       ← FastAPI serving 3 local models
    ├── scripts/
    │   ├── retrain_best_all.py   ← re-train models from CSV data
    │   └── smoke_test_models.py  ← quick inference check
    ├── utils/
    │   ├── dataset_loader.py     ← load CSV → HuggingFace dataset
    │   └── trainer_utils.py      ← Flan-T5 training config
    ├── faq_dataset.csv               ← Python FAQ training data
    ├── Organic_Compounds_Properties.csv  ← Chemistry training data
    ├── Bhagvad Gita.csv              ← Gita verse training data
    ├── python_model/    ← fine-tuned model weights (Flan-T5)
    ├── organic_model/   ← fine-tuned model weights (Flan-T5)
    └── gita_model/      ← fine-tuned model weights (Flan-T5)
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- `pip` or a virtual environment tool
- A free [Groq API key](https://console.groq.com) (for the general model + optional judge)

### Step 1 — Copy this folder anywhere you want

```bash
cp -r deterministic_platform_multi_model /path/to/wherever/you/want
cd /path/to/wherever/you/want/deterministic_platform_multi_model
```

### Step 2 — Set up environment

```bash
# Create .env in the project root and set at least GROQ_API_KEY
# (see Configuration section below for other variables)
touch .env
```

### Step 3 — Run (one command)

```bash
chmod +x start.sh
./start.sh
```

This will:
- Install dependencies if needed
- Start the Models API on **port 8010** (background)
- Wait for the models to load
- Start the Platform on **port 8020** (foreground)

### Step 4 — Open the dashboard

```
http://127.0.0.1:8020
```

---

## Manual Start (if you prefer terminals)

Open two terminal windows:

**Terminal 1 — Models API:**
```bash
cd models_api
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8010 --reload
```

**Terminal 2 — Platform:**
```bash
cd platform
pip install -r requirements.txt
python run.py
```

---

## API Endpoints

### Platform (port 8020)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`      | Browser dashboard UI |
| `GET`  | `/docs`  | Auto-generated API documentation |
| `POST` | `/query` | Run a query through the multi-model pipeline |
| `GET`  | `/report`| Aggregate stats: best model per domain, avg latency, etc. |

### Models API (port 8010)

| Method | Endpoint   | Description |
|--------|------------|-------------|
| `POST` | `/python`  | Python FAQ model |
| `POST` | `/organic` | Organic Chemistry model |
| `POST` | `/gita`    | Bhagavad Gita model |
| `POST` | `/ask`     | Auto-routes to the right model by keyword |

---

## Example Query

```bash
curl -X POST http://127.0.0.1:8020/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the boiling point of benzene?"}'
```

**Response:**
```json
{
  "best_answer": "80.1°C (176.18°F)",
  "best_model": "organic_model",
  "best_prompt": "v1",
  "metrics": {
    "accuracyScore": 0.82,
    "hallucinationScore": 0.18,
    "confidenceScore": 0.79,
    "latency": 340.5,
    "semantic": {
      "relevance": 0.91,
      "groundedness": 0.88,
      "hallucination": 0.0,
      "accuracy": 0.85,
      "confidence": 0.89
    }
  },
  "blue_metrics": {
    "behaviorStability": 0.74,
    "latency": 412.3,
    "usageCost": 0.000042,
    "errorRate": 0.0
  },
  "analysis": {
    "domain": "chemistry",
    "complexity": "low",
    "intent": "fact_lookup"
  },
  "all_responses": [ ... ]
}
```

---

## Models

Three fine-tuned **Flan-T5** seq2seq models, each trained on domain-specific CSV data:

| Model | Domain | Training Data |
|-------|--------|---------------|
| `python_model` | Python programming FAQ | `faq_dataset.csv` |
| `organic_model` | Organic chemistry properties | `Organic_Compounds_Properties.csv` |
| `gita_model` | Bhagavad Gita verses | `Bhagvad Gita.csv` |

Plus **Groq (Llama 3.1)** as the general-purpose model for queries that don't match any domain.

All local models use **greedy decoding** (`do_sample=False, num_beams=1`) — same input always produces same output.

---

## How Scoring Works

### Lexical Metrics (always computed, fast)

| Metric | Formula |
|--------|---------|
| `relevanceScore` | `query_keywords ∩ response_keywords / query_keywords` |
| `accuracyScore` | `0.65 × relevance + 0.35 × Jaccard` |
| `hallucinationScore` | `max(0, 1 − accuracy)` |
| `confidenceScore` | `accuracy − latency_penalty` |

### Semantic Metrics (optional, richer)

Enabled via `EVAL_ENABLE_SEMANTIC=true`:

| Metric | How |
|--------|-----|
| `relevance` | Cosine similarity between query and response embeddings |
| `groundedness` | % of response sentences with cosine sim ≥ 0.75 to context |
| `hallucination` | % of response sentences NOT grounded in context |
| `accuracy` | Cosine similarity to ground truth (or groundedness × relevance proxy) |
| `confidence` | Weighted avg of groundedness + relevance (+ NLI if enabled) |

### BLUE Metrics (cross-model stability)

| Metric | Meaning |
|--------|---------|
| `behaviorStability` | How consistent each model is across its 3 prompt variants |
| `latency` | Average across all 12 model calls |
| `usageCost` | Total estimated USD cost for this query |
| `errorRate` | Fraction of model calls that failed |

---

## Configuration

All settings are controlled by environment variables in `.env`. Key ones:

| Variable | Default | Effect |
|----------|---------|--------|
| `GROQ_API_KEY` | **required** | Groq API access |
| `EVAL_ENABLE_SEMANTIC` | `true` | Toggle semantic scoring |
| `EVAL_GROUNDING_MIN_SIM` | `0.75` | Grounding threshold |
| `EVAL_ENABLE_NLI` | `false` | NLI contradiction detection |
| `EVAL_ENABLE_LLM_JUDGE` | `false` | LLM-as-judge scoring |
| `USE_SEMANTIC_RANKING` | `true` | Semantic vs lexical ranking |
| `REQUEST_TIMEOUT_SECONDS` | `25` | Model call timeout |

See `PROJECT_CONTEXT.md` (Configuration section) for the full list with descriptions.

---

## Learning System

After every query, the platform updates `platform/store/model_performance.json`:

```json
{
  "chemistry": {
    "best_model": "organic_model",
    "avg_accuracy": 0.84,
    "avg_latency": 312.5,
    "count": 47
  },
  "python": {
    "best_model": "python_model",
    "avg_accuracy": 0.79,
    "avg_latency": 285.0,
    "count": 31
  }
}
```

This feeds back into routing — the historically best model per domain gets promoted in the ranking so it's tried earlier.

---

## Re-training Models (Optional)

If you want to retrain on the CSV data:

```bash
cd models_api
pip install -r requirements.txt
python scripts/retrain_best_all.py
```

To smoke-test that models load and run:

```bash
python scripts/smoke_test_models.py
```

---

## Copying to Another Machine

This folder is fully self-contained. To use it elsewhere:

```bash
# Copy the whole folder
cp -r deterministic_platform_multi_model /your/new/location/

# On the new machine:
cd /your/new/location/deterministic_platform_multi_model
# Create .env and set GROQ_API_KEY
touch .env
./start.sh
```

The model weights (`gita_model/`, `organic_model/`, `python_model/`) are included in the folder so no re-download or re-training is needed.

---

## Requirements

- Python 3.10 or higher
- No Docker required
- No database required
- Only external dependency: a Groq API key (free tier is sufficient)
