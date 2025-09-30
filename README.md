# FAQ Assistant

A production-ready FAQ assistant leveraging hybrid search (semantic + BM25) and LLM-based response generation with comprehensive evaluation pipelines.

## Features

- **Hybrid Retrieval** — Combines OpenAI embeddings (`text-embedding-3-small`) with BM25 keyword scoring for robust search across diverse query styles.
- **Context-Aware Generation** — GPT-4.1-mini generates responses with scenario-specific prompts (direct answers, ambiguous clarifications, greetings, capability overviews).
- **In-Memory Query Cache** — Normalized exact-match caching with FIFO eviction eliminates 2+ second latency on repeat queries; includes hit/miss tracking and statistics API.
- **Robust Error Handling** — Structured HTTP errors (400/422/500/502) with clear client guidance; layered exception handling at each pipeline stage.
- **Input Validation** — Query length limits (1-500 chars) with whitespace checks; Pydantic v2 validation for request/response schemas.
- **Docker Support** — Full containerization with Docker and docker-compose for consistent deployment.
- **Automated Evaluation** — LLM-judge scoring for generation quality; MRR@1 metrics for retrieval accuracy.
- **Configurable Settings** — Runtime-adjustable weights, temperature, and top-k via REST endpoints; persisted to disk.
- **Versioned Data Pipeline** — Classification, deduplication, and taxonomy evolution with auto-versioned outputs.

## Quick Start

### Prerequisites
- Python 3.9+ (for local development)
- [uv](https://docs.astral.sh/uv/) package manager
- Docker & Docker Compose (for containerized deployment)
- OpenAI API key

### Local Development Setup

```bash
# Install dependencies
make setup

# Set your OpenAI API key
cp .env.sample .env
# Edit .env and add your actual API key

# Run tests
make test

# Run evaluations
make eval-retrieval
make eval-generation
```

### Start the API Server

**Option 1: Local Development**
```bash
uv run uvicorn src.api:app --reload
```

**Option 2: Docker (Recommended for Production)**
```bash
# Set your OpenAI API key
cp .env.sample .env
# Edit .env and add your actual API key

# Build and run with Docker Compose
make docker-build
make docker-run

# View logs
make docker-logs

# Stop container
make docker-stop
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive API documentation.

## Project Structure

```
saas-group-assignment/
├── src/                        # Core application modules
│   ├── api.py                  # FastAPI routes and endpoints
│   ├── search.py               # Hybrid search implementation
│   ├── generation.py           # LLM response orchestration
│   ├── embeddings.py           # Embedding utilities
│   ├── config.py               # Settings management
│   └── pipelines/              # Data preparation modules
│       ├── data_cleaning.py    # LLM-based classification
│       └── deduplication.py    # Semantic deduplication
├── data/
│   ├── raw/                    # Original FAQ data
│   ├── processed/              # Classified and cleaned datasets
│   ├── configs/                # Intent taxonomy configurations
│   └── evals/                  # Evaluation datasets and results
├── scripts/                    # Data pipeline CLIs
│   ├── run_classification.py   # Classify FAQs by intent
│   ├── run_dedup.py            # Deduplicate similar questions
│   └── categorize_unknowns.py  # Cluster unknown FAQs
├── evals/                      # Evaluation modules
│   ├── evals_retrieval.py      # Retrieval metrics (MRR@1)
│   └── evals_generation.py     # LLM-judge evaluation
├── tests/                      # Pytest suite (13 tests across 5 modules)
├── docs/                       # Technical documentation
├── config/                     # Runtime app settings
├── Makefile                    # Automation commands
└── pyproject.toml              # Project metadata
```

## API Endpoints

### Query Endpoint
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I reset my password?"}'
```

Response includes the generated answer, confidence level, top-k search results, and metadata.

### Settings Management
```bash
# Get current settings
curl http://localhost:8000/settings

# Update settings
curl -X PUT http://localhost:8000/settings \
  -H "Content-Type: application/json" \
  -d '{"top_k": 5, "semantic_weight": 0.7, "bm25_weight": 0.3}'
```

### Health Check
```bash
curl http://localhost:8000/health
```

### Cache Management
```bash
# Get cache statistics (includes hit rate)
curl http://localhost:8000/cache/stats

# Clear cache and reset statistics (e.g., after FAQ updates)
curl -X POST http://localhost:8000/cache/clear
```

## Development Workflow

### Data Pipeline

```bash
# 1. Classify raw FAQs
uv run python scripts/run_classification.py

# 2. Deduplicate and clean
uv run python scripts/run_dedup.py --input data/processed/classified_faq_v2.json

# 3. Categorize any unknowns (if present)
uv run python scripts/categorize_unknowns.py --input data/processed/classified_faq_v2.json
```

### Running Evaluations

```bash
# Retrieval quality (MRR@1)
make eval-retrieval

# Generation quality (LLM judge)
make eval-generation

# Or run with custom options
uv run python -m evals.evals_retrieval --gold data/evals/gold_queries.json --print-metrics
uv run python -m evals.evals_generation --cases data/evals/generation_eval_cases.json --print-summary
```

### Testing

```bash
# Run full test suite
make test

# Run specific test file
uv run pytest tests/test_search.py -v
```

## Evaluation Results

### Retrieval Performance
- **MRR@1 (top-1 accuracy)**: 1.000 over 12 gold queries
- Covers direct and paraphrased queries across all core intents

### Generation Quality
- **16/16 cases** scored `good` by LLM judge
- Categories: direct answers, ambiguous clarifications, greetings/farewells, capability overviews, off-topic handling, unknown fallbacks

## Configuration

### Intent Taxonomy
Intent configurations live in `data/configs/intent_config_v*.json` and define the supported FAQ categories:
- Account management
- Security & privacy
- Billing & subscriptions
- Integrations
- Settings & preferences
- And more...

### Runtime Settings (`config/app_settings.json`)
```json
{
  "top_k": 3,
  "semantic_weight": 0.7,
  "bm25_weight": 0.3,
  "temperature": 0.2
}
```

## Key Design Decisions

1. **Hybrid Search** — Semantic embeddings (70%) + BM25 (30%) handles both natural language and keyword queries
2. **In-Memory Query Cache** — Normalized exact-match caching with FIFO eviction and hit/miss tracking (future: semantic similarity via `gptcache`)
3. **Scenario-Based Prompting** — Different response strategies for direct matches, ambiguities, greetings, off-topic, and unknowns
4. **Layered Error Handling** — Structured HTTP status codes (400/422/500/502) with stage-specific exception handling
5. **Input Protection** — Query length validation (max 500 chars) prevents abuse and excessive API costs
6. **LLM-Judge Evaluation** — Qualitative scoring that captures tone, helpfulness, and accuracy
7. **Versioned Datasets** — Full audit trail from raw → classified → cleaned with timestamps
8. **File-Backed Config** — Settings persist across restarts while supporting live updates
9. **Docker Containerization** — Reproducible deployment with multi-stage builds and health checks

## Production Considerations

For scaling beyond the current prototype:
- Replace in-memory search with FAISS or vector DB
- Add authentication and rate limiting
- Implement monitoring dashboards for query analytics
- Schedule classification/dedup as background jobs
- Version control intent configs with approval workflows
- **Version prompts via Git or LangSmith** — Currently hard-coded in `generation.py`; move to versioned prompt templates with A/B testing support
- Add A/B testing framework for prompt variations

## Documentation

- `docs/documentation.md` — Comprehensive design decisions, implementation details, and operational guide

## License

Internal project for educational purposes.

