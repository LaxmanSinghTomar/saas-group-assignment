# FAQ Assistant: Design & Implementation

## Problem Analysis

### What We're Solving

Customer support receives repetitive questions, but users rarely phrase queries exactly as written in our FAQ database. We need a system that:
1. Understands intent behind natural language queries
2. Returns relevant FAQ answers even with paraphrasing
3. Handles ambiguous, off-topic, and edge-case queries gracefully
4. Provides helpful responses even when no exact FAQ exists

### Data Quality Observations

Analyzing the 40-entry FAQ dataset revealed:
- **Malformed entries**: Empty questions, gibberish ("asdfghjkl")
- **Off-topic noise**: Pizza toppings, greeting fragments ("Hi")
- **Duplicates**: 3 username change variations, 2 account deletion variants
- **Mismatched pairs**: Questions that don't align with their answers

### Key Assumptions

1. **Scale**: <100 FAQs initially; moderate query volume (1K-10K/day)
2. **Query distribution**: Unknown ratio of relevant vs. off-topic (optimize for majority case)
3. **Language**: English-only for v1
4. **Update frequency**: FAQs change infrequently (weekly/monthly)
5. **Cost**: OpenAI API usage acceptable for business value
6. **Paraphrasing is common**: Users phrase the same question many different ways

---

## Architecture & Approach

### Core Design: Single-Stage Hybrid Search + LLM Generation

```
User Query → Hybrid Search (Semantic + BM25) → LLM Generation → Response
```

### Why This Approach?

**What We DIDN'T Do: Two-Stage Intent Classification**

```
❌ User Query → Classify Intent → Route to Intent-Specific Search → Answer
```

**Reasoning:**
1. **Unknown distribution** — Without usage data, can't optimize for actual off-topic frequency (could be 5% or 40%)
2. **Latency tax** — Adds +200ms LLM call to every query, hurting 70-80% legitimate requests
3. **Cost overhead** — 6% more expensive overall
4. **Premature optimization** — Adds complexity without proven benefit

**When to reconsider:** If production data shows off-topic queries >40% of traffic.

**What We DID Do: Single-Stage with Smart Generation**

Benefits:
- **Simpler**: Single code path, fewer failure points
- **Faster**: ~400ms vs. ~600ms for majority queries
- **Flexible**: LLM generation layer handles all edge cases naturally
- **Data-driven**: Instrument and optimize based on real metrics

### Key Decision: Hybrid Search (Semantic + BM25)

**Problem Observed:**

Pure semantic search underserved keyword queries:
- Query: `"password"` → Low scores on relevant FAQs
- Query: `"2FA"` → Missed exact technical term matches

**Solution:**

Blend semantic embeddings with keyword scoring:
```python
combined_score = 0.7 × semantic_similarity + 0.3 × bm25_score
```

**Why This Blend?**
- **Semantic** (70%): Handles paraphrasing ("I forgot my password" → "reset password")
- **BM25** (30%): Boosts exact term matches (technical keywords, short queries)
- **Configurable**: Weights adjustable via `/settings` API for tuning

**Result:** Keyword queries improved without sacrificing semantic understanding.

---

## Alternatives Considered

### 1. Pure Semantic Search
**Trade-off:** Simpler but fails on keyword queries → Rejected after empirical testing

### 2. Intent Classification + Scoped Search
**Trade-off:** Better off-topic filtering but adds latency/cost to majority case → Deferred until data justifies

### 3. Template-Based Responses
**Trade-off:** Faster but can't handle nuance or edge cases → LLM generation provides better UX

### 4. Vector Database (Pinecone, Weaviate)
**Trade-off:** Production-grade but overkill for 36 FAQs → In-memory NumPy sufficient for current scale

### 5. Fine-Tuned Classification Model
**Trade-off:** Lower per-query cost but requires training data we don't have → LLM-based classification more practical

### 6. Keep All FAQ Variations
**Trade-off:** More coverage but semantic search handles paraphrasing automatically → Canonical-only keeps database clean

---

## Implementation Details

### Data Preparation Pipeline

**Step 1: LLM-Based Classification**

Used GPT-4.1-mini (temp=0.1, JSON schema) to classify each FAQ:
```json
{
  "intent": "security",
  "quality_score": 0.95,
  "issues": [],
  "reasoning": "Clear password reset question with actionable answer"
}
```

**Why LLM over rules?** Handles edge cases ("Hi" could be testing/greeting), provides reasoning, scales to 4,000 entries.

**Result:** 38 relevant FAQs, 2 off-topic removed.

**Step 2: Semantic Deduplication**

Threshold tuning via empirical logging:
- 0.95 → No merges (too strict)
- 0.85 → Merged paraphrases, preserved distinct questions

Example: `"update username"` ↔ `"change username"` (similarity: 0.889) → Merged

**Result:** 36 canonical FAQs with duplicate lists for transparency.

**Step 3: Taxonomy Evolution**

6 `unknown` FAQs clustered via LLM → discovered 4 new intents (integrations, customer support, legal, product access).

Promoted to v2 taxonomy → reclassified → zero unknowns remaining.

**Key Innovation:** LLM-driven taxonomy growth vs. manual guesswork.

### Hybrid Search Implementation

**Semantic Layer:**
- OpenAI `text-embedding-3-small` (1536-dim)
- Pre-normalized embeddings for fast cosine similarity
- In-memory NumPy operations (~10ms for 36 FAQs)

**Keyword Layer:**
- BM25 (Okapi variant) via `rank-bm25`
- Simple tokenization (lowercase, alphanumeric)
- Normalized scores [0, 1]

**Blending:**
```python
semantic_norm = semantic_scores / max(semantic_scores)
bm25_norm = bm25_scores / max(bm25_scores)
combined = 0.7 * semantic_norm + 0.3 * bm25_norm
```

**Why normalize?** Prevents one scoring method from dominating; makes weights interpretable.

### LLM Response Generation

**Scenario-Specific Prompting:**

1. **Clear match** (confidence >0.85) → Direct FAQ answer, no extras
2. **Ambiguous** (multiple relevant FAQs) → Warm clarification with bullet options
3. **Off-topic** (no relevant context) → Polite refusal
4. **Unknown** (no FAQ match) → Acknowledge gap, provide support contact
5. **Greetings/farewells** → Brief friendly acknowledgment
6. **Capability queries** → List core support categories

**Why comprehensive prompting?**
- LLM naturally detects scenarios from context
- Maintainable (update prompt vs. code)
- Higher quality than templates
- Single generation path ensures consistency

**Temperature: 0.2** — Balance between consistency and natural variation.

---

## Setup Instructions

### Prerequisites
- Python 3.9+ (for local development)
- [uv](https://docs.astral.sh/uv/) package manager (fast, modern Python tooling)
- Docker & Docker Compose (for containerized deployment)
- OpenAI API key

### Quick Start (Local Development)

```bash
# 1. Clone repository
git clone <repo-url>
cd saas-group-assignment

# 2. Install dependencies
make setup

# 3. Set API key
cp .env.sample .env
# Edit .env and add your actual OpenAI API key

# 4. Run tests
make test

# 5. Start server
uv run uvicorn src.api:app --reload
```

Server runs at `http://localhost:8000`. Visit `/docs` for interactive API documentation.

### Quick Start (Docker)

```bash
# 1. Clone repository
git clone <repo-url>
cd saas-group-assignment

# 2. Set API key
cp .env.sample .env
# Edit .env and add your actual OpenAI API key

# 3. Build and run with Docker Compose
make docker-build
make docker-run

# 4. Check status
curl http://localhost:8000/health
```

**Docker Commands:**
```bash
make docker-build    # Build Docker image
make docker-run      # Start container in background
make docker-stop     # Stop and remove container
make docker-logs     # View logs
make docker-shell    # Open shell in container
```

### Verify Installation

```bash
# Run evaluations
make eval-retrieval    # Should show: MRR@1: 1.000
make eval-generation   # Should show: 16/16 good

# Test query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I reset my password?"}'
```

### Project Commands

**Development:**
```bash
make setup              # Install dependencies via uv
make test               # Run pytest suite
make eval-retrieval     # Retrieval metrics (MRR@1)
make eval-generation    # Generation quality (LLM judge)
make clean              # Remove caches and temp files
```

**Docker:**
```bash
make docker-build       # Build Docker image
make docker-run         # Start container with docker-compose
make docker-stop        # Stop and remove container
make docker-logs        # View container logs (follow mode)
make docker-shell       # Open bash shell in running container
```

---

## Decision Logic & Edge Case Handling

### When to Use FAQ vs. Other Methods

**Confidence-Based Strategy:**

```python
if top_semantic_score >= 0.85:
    # HIGH CONFIDENCE: Clear match
    → Return direct FAQ answer
    
elif top_semantic_score >= 0.65:
    # MEDIUM CONFIDENCE: Relevant but uncertain
    → Provide top FAQs, may ask for clarification
    
else:  # < 0.65
    # LOW CONFIDENCE: Likely off-topic or unknown
    → LLM handles gracefully (refusal or fallback)
```

**Why these thresholds?**

Empirical testing showed:
- >0.85: Almost always correct FAQ
- 0.65-0.85: Topically relevant but may need clarification
- <0.65: Usually off-topic or novel question

**Conservative philosophy:** Better to acknowledge uncertainty than confidently provide wrong answer.

### Edge Case Handling

| Scenario | Detection | Response |
|----------|-----------|----------|
| **Ambiguous query** ("password") | Multiple FAQs with similar scores | List options with clarifying question |
| **Off-topic** ("Tell me a joke") | No relevant context | "I'm an FAQ assistant and can't help with that request" |
| **Unknown but relevant** ("Do you support SSO?") | Low confidence, no FAQ match | "Sorry, I don't have information on <topic>. Please reach out to support@example.com" |
| **Greeting** ("Hey", "Hi") | LLM detects social interaction | Brief acknowledgment + FAQ availability reminder |
| **Capability query** ("What can you do?") | Generic help request | List core support categories |
| **Empty query** | String validation | Return 422 validation error |
| **Whitespace-only query** | Pydantic validator | Return 422 validation error |
| **Very long query** (>500 chars) | Length validation | Return 422 validation error |
| **Very short query** (1-2 words) | Treated as ambiguous | Force clarification path |

**User Experience Focus:**

- **Honest about uncertainty** — Never guess when unsure
- **Helpful even without FAQ** — Provide support contact for unknowns
- **Natural conversation** — Warm tone for greetings, appropriate refusal for off-topic
- **Actionable clarification** — Group related FAQs, ask specific question
- **Clear error messages** — Structured HTTP errors (400/500/502) with descriptive details
- **Input protection** — Query length limits prevent abuse and excessive costs

---

## API Error Handling

### HTTP Status Codes

The API returns appropriate status codes for different error scenarios:

| Status Code | Error Type | Cause | Example |
|-------------|------------|-------|---------|
| **400 Bad Request** | Validation Error | Invalid input (auto-handled by Pydantic) | Empty query, missing field |
| **422 Unprocessable Entity** | Validation Error | Input fails custom validation | Query >500 chars, whitespace-only |
| **500 Internal Server Error** | Configuration Error | Missing API key, missing FAQ index | `OPENAI_API_KEY` not set |
| **502 Bad Gateway** | External Service Error | OpenAI API failure, timeout, rate limit | API down, quota exceeded |

### Error Response Structure

All errors return JSON with a `detail` field:

```json
{
  "detail": "Configuration error: OPENAI_API_KEY not set; cannot generate embeddings"
}
```

Validation errors (422) include structured details:

```json
{
  "detail": [
    {
      "type": "string_too_long",
      "loc": ["body", "query"],
      "msg": "String should have at most 500 characters",
      "input": "...",
      "ctx": {"max_length": 500}
    }
  ]
}
```

### Input Validation

**Query Constraints:**
- **Min length**: 1 character
- **Max length**: 500 characters
- **Whitespace**: Not allowed as sole content (stripped and validated)

**Rationale**: 500 characters is generous for natural language questions while protecting against abuse, excessive latency, and unnecessary API costs.

### Error Handling Strategy

**Layered error handling** ensures appropriate responses at each stage:

1. **Index Loading**: Catches missing or corrupt FAQ index → 500
2. **Search Stage**: Catches embedding API errors → 500 (config) or 502 (service)
3. **Generation Stage**: Catches LLM API errors → 500 (config) or 502 (service)
4. **Catch-all**: Unexpected errors → 500 with generic message

**Client Guidance:**
- **422**: Fix input (reduce length, add content)
- **500**: Contact administrator (configuration issue)
- **502**: Retry with exponential backoff (temporary service issue)

---

## Testing Strategy

### Three-Layer Approach

**1. Pytest (Structural Validation)**
- Search pipeline returns correct data types
- Generation receives proper context
- Metrics calculations work
- API input validation (max length, whitespace, empty queries)
- Cache statistics structure and functionality
- Endpoint contract validation (health, cache stats)
- No import or wiring errors

**Test Coverage**: 13 tests across 5 test modules

**2. Automated Retrieval Metrics**
- **MRR@1:** 1.000 (perfect top-1 accuracy on 12 gold queries)
- Covers direct and paraphrased queries across all intents

**3. LLM-as-Judge (Qualitative)**
- **16/16 cases scored `good`** (100% pass rate)
- Tests: direct answers, ambiguity handling, greetings, capability overviews, off-topic, unknown fallbacks

**Why LLM Judge?**

Traditional assertions are brittle:
```python
# Breaks on rephrasing:
assert response == "Click 'Forgot Password'..."

# Insufficient:  
assert "password" in response  # Passes for "I don't know about passwords"
```

LLM judge evaluates:
- Semantic correctness (meaning, not exact text)
- Tone appropriateness (warm vs. robotic)
- Scenario adherence (ambiguous → clarification, not direct answer)

**Scalable:** Runs as fast as manual review, fully reproducible.

### Test Coverage

| Category | Example | Expected Behavior | Result |
|----------|---------|-------------------|--------|
| **Direct** | "How do I reset my password?" | Exact FAQ answer | ✓ High confidence |
| **Paraphrase** | "I forgot my password" | Correct FAQ via semantic match | ✓ High confidence |
| **Ambiguous** | "password" | List options + ask which | ✓ Clarification provided |
| **Off-topic** | "Order me a pizza" | Polite refusal | ✓ Standard message |
| **Unknown** | "Do you support SSO?" | Acknowledge gap + support email | ✓ Fallback message |
| **Greeting** | "Hey there" | Friendly response | ✓ Natural acknowledgment |
| **Capability** | "What can you help me with?" | List core categories | ✓ Overview provided |

---

## Performance & Efficiency

### Current Metrics

**Latency (without cache hit):**
- Embedding generation: ~100ms
- Search (hybrid): ~10ms
- LLM generation: ~2000-2500ms
- **Total: ~1-2.5** (exceeds <500ms target; dominated by LLM response time)

**Cost:**
- Embedding: $0.00002 per query
- Generation: $0.00030 per query
- **Total: $0.00032/query**

**At Scale:**
- 1,000 queries/day → $10/month
- 10,000 queries/day → $96/month

### Efficiency Considerations

**What We Avoided:**
- ❌ No upfront intent classification (saves +200ms on 70-80% of queries)
- ❌ No vector database overhead (in-memory sufficient for current scale)
- ❌ No fine-tuning complexity (LLM-based classification is practical)

**Simple Optimizations Included:**
- ✓ Pre-computed FAQ embeddings (cached to disk)
- ✓ In-memory FAQ index (loaded once at application startup)
- ✓ Normalized embeddings pre-computed (faster cosine similarity)
- ✓ Single LLM call per query (comprehensive prompt handles all scenarios)
- ✓ **In-memory query cache** — Exact match caching with query normalization (eliminates 2+ second latency on repeat queries)

**Query Caching Implementation:**
- Normalize queries: lowercase, strip whitespace, collapse multiple spaces
- Simple dictionary cache with FIFO eviction (max 1000 entries)
- Cache hit avoids embedding generation + LLM call (~2.1-2.6s saved)
- **Hit/miss tracking**: Full observability with hit rate metrics
- Cache stats exposed via `/cache/stats` and `/health` endpoints
- Clear cache and reset statistics with `/cache/clear` API endpoint

**Future Optimizations (When Data Justifies):**
- Semantic similarity caching via `gptcache` (cache similar queries, not just exact matches)
- Async batch processing for multiple queries
- Response streaming for large answers
- Pre-computed answers for frequent queries
- **Smaller open-source models** — Replace GPT-4.1-mini with locally-hosted models (Llama 3.1 8B, Mistral 7B) for sub-500ms response times and zero per-query costs

---

## Evaluation & Quality Assurance

### Retrieval Quality

**Metric: MRR@1 (Top-1 Accuracy)**

```bash
$ make eval-retrieval
Queries evaluated: 12
MRR@1 (top-1 accuracy): 1.000
```

**Why MRR@1?** With single relevant FAQ per query, MRR@1 = top-1 accuracy (clear business metric).

**Test Set:** 12 queries (4 direct, 8 paraphrased) covering all core intents.

**Result:** Perfect retrieval — every query returned correct FAQ in first position.

### Generation Quality

**Metric: LLM Judge Scoring**

```bash
$ make eval-generation
Cases evaluated: 16
good: 16
mixed: 0
bad: 0
```

**Coverage:**
- 8 direct/paraphrase (FAQ answer accuracy)
- 3 ambiguous (clarification behavior)
- 3 off-topic (polite refusal)
- 2 unknown (fallback messaging)
- 4 social (greetings, farewells)
- 3 capability (category overview)

**Judge evaluates:** Semantic correctness, tone, scenario adherence, scope.

**Result:** 100% pass rate — all responses meet quality expectations.

---

## Limitations & Future Improvements

### Current Limitations

1. **No conversation memory** — Each query treated independently; can't handle "What about billing?" as follow-up
2. **Exact match caching only** — Cache doesn't recognize paraphrases (future: semantic similarity via `gptcache`)
3. **English-only** — No multilingual support
4. **Manual FAQ updates** — Requires restart to reload index
5. **Single-instance only** — No horizontal scaling support yet (cache not shared across instances)
6. **No query analytics** — Can't identify common patterns or gaps without usage data
7. **Hard-coded prompts** — System prompts are embedded in `generation.py`; should be versioned externally via Git or LangSmith for A/B testing and rollback

### With More Time (Priority Order)

**Immediate (1-2 hours):**
1. **Query logging** — Foundation for all data-driven optimization
2. **Conversation memory** — Session management for follow-ups

**Short-term (1 week):**
1. **Prompt versioning** — Move prompts to external templates with Git versioning or LangSmith for A/B testing, rollback, and audit trails
2. **Usage dashboard** — Query volume, confidence distribution, unknown rate, cache hit rate
3. **Semantic caching** — Upgrade to `gptcache` for similarity-based cache (recognize paraphrases)
4. **Redis cache** — Shared cache for multi-instance deployment
5. **A/B testing framework** — Systematically test prompt variations
6. **Expanded eval sets** — Typos, very long queries, adversarial inputs

**Medium-term (1 month):**
1. **Auto FAQ gap detection** — Cluster unknowns, propose new FAQ topics
2. **Feedback loop** — Thumbs up/down to improve retrieval
3. **Advanced retrieval** — Cross-encoder re-ranking for top-K results
4. **Multi-instance support** — Redis for settings, load balancing

### With Real Usage Data

**What We'd Optimize:**

1. **Thresholds** — Currently 0.85/0.65 based on small test set; tune with production distribution
2. **Weights** — 70/30 semantic/BM25 blend may vary by query type
3. **Prompts** — A/B test variations, optimize based on feedback
4. **Cache strategy** — Identify actual common queries vs. guessing
5. **FAQ coverage** — Expand based on frequent unknowns, not assumptions

**Decision Framework:**

- If off-topic >40%: Add upfront classification
- If unknowns >15%: Prioritize FAQ expansion
- If ambiguous >20%: Add query expansion or adjust thresholds
- If latency P95 >800ms: Add caching or async processing

---

## Quick Reference

### Start the System

**Local Development:**
```bash
# One-time setup
make setup
cp .env.sample .env
# Edit .env and add your actual API key

# Run server
uv run uvicorn src.api:app --reload

# Verify
curl http://localhost:8000/health
```

**Docker Deployment:**
```bash
# One-time setup
cp .env.sample .env
# Edit .env and add your actual API key

# Build and run
make docker-build
make docker-run

# Verify
curl http://localhost:8000/health

# Test query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I reset my password?"}'
```

### Run Evaluations

```bash
make test               # Pytest suite (4 tests)
make eval-retrieval     # MRR@1 metric (should be 1.000)
make eval-generation    # LLM judge (should be 16/16 good)
```

### Data Pipeline (If Needed)

```bash
# Classify FAQs
uv run python scripts/run_classification.py

# Deduplicate
uv run python scripts/run_dedup.py --input data/processed/classified_faq_v2.json

# Cluster unknowns (if any)
uv run python scripts/categorize_unknowns.py
```

### Configuration & Cache Management

```bash
# View current settings
curl http://localhost:8000/settings

# Update settings (example: emphasize semantic search)
curl -X PUT http://localhost:8000/settings \
  -H "Content-Type: application/json" \
  -d '{"semantic_weight": 0.8, "bm25_weight": 0.2, "temperature": 0.15}'

# Check cache statistics
curl http://localhost:8000/cache/stats

# Clear cache (e.g., after FAQ updates)
curl -X POST http://localhost:8000/cache/clear
```

---

## For Deeper Technical Details

See `docs/TECHNICAL_REPORT.md` for:
- Complete implementation details
- Iterative development journey
- Full evaluation methodology
- Production scaling strategies
- Technology stack rationale
- Code examples and algorithms

---

**Ready for deployment and real-world usage.**

