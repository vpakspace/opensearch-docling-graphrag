# OpenSearch Docling GraphRAG

Fully local RAG pipeline combining **OpenSearch** (hybrid BM25 + k-NN vector search), **Neo4j** (knowledge graph), **Ollama** (LLM + embeddings), and **Docling** (document parsing). No cloud API keys required — **92% benchmark accuracy** (167/180), 229 tests, 18 commits, ~7,000 LOC, 6 search modes including Cog-RAG inspired cognitive retrieval.

## Architecture

```
Document ──► Docling ──► Chunker ──► Embedder (Ollama) ──► OpenSearch (k-NN + BM25)
                                        │
                                        ▼
                                  Entity Extractor (Ollama LLM)
                                        │
                                        ▼
                                  Neo4j Knowledge Graph
                                        │
         ┌──────────────────────────────┘
         ▼
    Retriever (6 modes: bm25 / vector / graph / hybrid / enhanced / cognitive)
         │                                          │
         ▼                                          ▼
    Semantic Cache (LRU + cosine)           Dynamic RRF Weights
         │
         ▼
    Generator (Ollama LLM) ──► Answer + Sources + Confidence + Hallucination Check
```

**Graph schema:**

```
(Document) ─HAS_CHUNK─► (Chunk)
(Entity) ─MENTIONED_IN─► (Chunk)
(Entity) ─RELATED_TO─► (Entity)
```

## Stack

| Component | Technology | Port |
|-----------|-----------|------|
| Vector + BM25 Search | OpenSearch 2.15 | 9200 |
| Knowledge Graph | Neo4j 5 | 7474 / 7687 |
| LLM + Embeddings | Ollama (llama3.1:8b + nomic-embed-text-v2-moe) | 11434 |
| Document Parsing | Docling (PDF, DOCX, PPTX, HTML, TXT, MD) | — |
| REST API | FastAPI + slowapi rate limiting | 8508 |
| UI | Streamlit (6 tabs) + PyVis | 8506 |
| Dashboards | OpenSearch Dashboards (optional) | 5601 |

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- [Ollama](https://ollama.com/) (or use the Docker service)
- Python 3.12+
- GPU recommended (for Docling and Ollama acceleration)

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/vpakspace/opensearch-docling-graphrag.git
cd opensearch-docling-graphrag

pip install -r requirements.txt
cp .env.example .env
```

### 2. Start infrastructure

```bash
docker compose up -d
```

This starts OpenSearch, Neo4j, and Ollama with healthchecks. Wait for all services to become healthy:

```bash
docker compose ps
```

### 3. Pull Ollama models

```bash
bash scripts/pull_models.sh
```

Downloads `llama3.1:8b` (LLM) and `nomic-embed-text-v2-moe` (embeddings, 768d, ~100 languages).

### 4. Ingest a document

```bash
python scripts/ingest.py data/sample_graphrag.txt
```

Options:
- `--skip-ner` — skip entity extraction (no graph building)
- `--use-gpu` — enable GPU acceleration for Docling

Ingest an entire directory:

```bash
python scripts/ingest.py ~/documents/
```

### 5. Launch the UI

```bash
streamlit run ui/streamlit_app.py --server.port 8506
```

Open http://localhost:8506 — all 6 tabs are ready.

### 6. Or use the API

```bash
python run_api.py
```

```bash
# Health check
curl http://localhost:8508/api/v1/health

# Ask a question (no auth)
curl -X POST http://localhost:8508/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"text": "What is OpenSearch?", "mode": "hybrid"}'

# With API key auth (if API_KEY env is set)
curl -X POST http://localhost:8508/api/v1/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"text": "What is OpenSearch?", "mode": "hybrid"}'
```

## Search Modes

| Mode | How it works |
|------|-------------|
| `bm25` | Full-text search via OpenSearch BM25 |
| `vector` | Semantic search via OpenSearch k-NN (cosine, HNSW) |
| `graph` | Entity match in Neo4j → traverse to chunks |
| `hybrid` | Dynamic RRF fusion of bm25 + vector + graph (adaptive weights) |
| `enhanced` | Query expansion + 3x candidates + dynamic RRF + cosine reranking |
| `cognitive` | Cog-RAG inspired 2-stage retrieval (theme + entity) + reranking + hallucination detection |

### Dynamic RRF Weights

Hybrid and enhanced modes automatically classify queries and adjust fusion weights:

| Query Type | BM25 Weight | Vector Weight | Graph Weight |
|-----------|-------------|---------------|-------------|
| **Keyword** (≤3 words, no `?`) | 1.5 | 0.8 | 0.7 |
| **Semantic** (longer/questions) | 0.8 | 1.5 | 0.7 |

## Benchmark

30 questions (Russian + English) x 6 modes = 180 evaluations. Keyword overlap judge with cross-language concept map (no external API).

| Mode | Score | Doc1 (RU) | Doc2 (EN) | Avg Latency |
|------|-------|-----------|-----------|-------------|
| `bm25` | **29/30 (96%)** | 15/15 | 14/15 | 4.0s |
| `hybrid` | **29/30 (96%)** | 14/15 | 15/15 | 3.0s |
| `enhanced` | **29/30 (96%)** | 14/15 | 15/15 | 5.6s |
| `vector` | 27/30 (90%) | 12/15 | 15/15 | 4.2s |
| `cognitive` | 27/30 (90%) | 12/15 | 15/15 | 5.3s |
| `graph` | 26/30 (86%) | 12/15 | 14/15 | 2.6s |
| **Overall** | **167/180 (92%)** | **79/90** | **88/90** | **p50=3.4s** |

By question type: `global` 100% | `simple` 95% | `multi_hop` 94% | `relation` 92% | `temporal` 75%

Zero persistent failures — every question passes in at least one mode. Latency: p50=3.4s, p95=9.5s, p99=21.5s.

LLM: `llama3.1:8b` | Embeddings: `nomic-embed-text-v2-moe` | GPU: RTX 4080

```bash
python scripts/run_benchmark.py
```

### Cog-RAG Features

Inspired by [Cog-RAG (AAAI 2026)](https://arxiv.org/abs/2505.02601):

- **Query Expansion** — LLM-based extraction of themes, entities, and related terms
- **Two-Stage Cognitive Retrieval** — Stage 1 (theme): BM25 + vector → RRF; Stage 2 (entity): BM25 + graph → RRF; merged via RRF + cosine reranking
- **Cosine Reranking** — post-retrieval reranking using embedding similarity (no numpy)
- **Hallucination Detection** — lightweight token overlap check between answer and context
- **Multi-Signal Confidence** — score consistency + token overlap + source diversity

## Security & API

### API Endpoints

| Method | Path | Auth | Rate Limit | Description |
|--------|------|------|-----------|-------------|
| `GET` | `/api/v1/health` | No | — | Service health (OpenSearch, Neo4j, Ollama) |
| `POST` | `/api/v1/query` | Yes | 60/min | RAG query → answer + sources + confidence |
| `POST` | `/api/v1/search` | Yes | 60/min | Search only (no generation) |
| `GET` | `/api/v1/graph/stats` | Yes | — | Knowledge graph statistics |

### API Key Authentication

Optional — disabled when `API_KEY` environment variable is not set.

```bash
# Enable API key auth
export API_KEY="your-secret-key"
python run_api.py
```

When enabled, all endpoints except `/api/v1/health` require `X-API-Key` header.

### Security Features

- **API key auth** — constant-time `hmac.compare_digest` (timing-attack safe), env `API_KEY`
- **Rate limiting** — 60 requests/minute on `/query` and `/search` via [slowapi](https://github.com/laurentS/slowapi)
- **CORS** — configurable origins via `CORS_ORIGINS` env (default: `http://localhost:8506`)
- **Security headers** — `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection`
- **Cypher injection prevention** — regex allowlist for Neo4j relationship types
- **No stacktrace leaks** — global exception handlers return JSON errors, no internal details
- **Config validation** — Pydantic Field constraints on all numeric settings (chunk_size >0, temperature 0.0–2.0, etc.)
- **Embedding dimension check** — EmbeddingError if Ollama returns unexpected vector dimensions
- **XSS prevention** — Streamlit uses `st.text()` for LLM-generated answers (no raw HTML rendering)
- **Path traversal prevention** — `os.path.realpath()` normalization for batch upload paths
- **Docker hardening** — all ports bound to `127.0.0.1`, not exposed externally
- **Connection pooling** — single cached `httpx.Client` per process for Ollama calls

## Semantic Cache

Built-in query cache that avoids redundant LLM calls for repeated or similar queries:

- **Exact hash lookup** — instant match for identical queries (no embedding needed)
- **Cosine similarity** — matches semantically similar queries (threshold: 0.95)
- **LRU eviction** — max 256 entries, oldest evicted first
- **TTL** — entries expire after 300 seconds
- **Zero config** — enabled by default, no external dependencies

## Streamlit UI (6 Tabs)

| Tab | Features |
|-----|----------|
| Home | Service status, document/chunk/entity counts |
| Upload | Drag-and-drop ingestion, optional NER, GPU toggle |
| Search & Q&A | 6 search modes, confidence bar, hallucination warning, expandable sources |
| Graph Explorer | Interactive PyVis visualization, entity type filter |
| Batch Process | Directory ingestion with progress bar |
| Settings | Configuration display, clear index / clear graph |

The UI supports **English and Russian** (sidebar language selector).

## Project Structure

```
opensearch-docling-graphrag/
├── opensearch_graphrag/           # Core pipeline (19 modules)
│   ├── config.py                  # Pydantic Settings + Field validation
│   ├── models.py                  # Chunk, Entity, SearchResult, QAResult
│   ├── loader.py                  # Docling document parser
│   ├── chunker.py                 # Markdown-aware splitting
│   ├── embedder.py                # Ollama embeddings + dimension check
│   ├── opensearch_store.py        # k-NN index + BM25 + hybrid search
│   ├── entity_extractor.py        # NER via Ollama LLM
│   ├── graph_builder.py           # Neo4j graph construction
│   ├── retriever.py               # 6-mode retriever + dynamic RRF fusion
│   ├── cognitive_retriever.py     # Cog-RAG 2-stage retriever
│   ├── query_expander.py          # LLM-based query expansion
│   ├── reranker.py                # Cosine similarity reranker
│   ├── hallucination_detector.py  # Token overlap grounding check
│   ├── cache.py                   # Semantic cache (LRU + cosine similarity)
│   ├── retry.py                   # Retry decorator for Ollama calls
│   ├── exceptions.py              # Custom exception hierarchy
│   ├── generator.py               # Ollama chat generation + confidence calibration
│   ├── utils.py                   # Shared cosine_similarity + RRF fusion
│   └── service.py                 # PipelineService orchestrator + cache
├── api/
│   ├── app.py                     # FastAPI factory + auth + rate limiting + exception handlers
│   ├── routes.py                  # REST endpoints with rate limits
│   ├── limiter.py                 # slowapi Limiter instance
│   └── deps.py                    # Dependency injection
├── ui/
│   ├── streamlit_app.py           # 6-tab UI (XSS-safe)
│   ├── i18n.py                    # EN/RU translations
│   └── components/graph_viz.py    # PyVis rendering
├── scripts/
│   ├── ingest.py                  # CLI ingestion
│   ├── run_benchmark.py           # 30-question benchmark runner
│   └── pull_models.sh             # Download Ollama models
├── benchmark/                     # Benchmark data
│   ├── questions.json             # 30 questions (RU + EN)
│   └── results.json               # Latest benchmark results
├── tests/                         # 229 tests (all mocked)
├── data/                          # Sample documents (Doc1 RU + Doc2 EN)
├── docker-compose.yml             # OpenSearch + Neo4j + Ollama
├── requirements.txt
├── pyproject.toml
├── run_api.py                     # uvicorn launcher (port 8508)
└── .env.example
```

## Configuration

All settings are controlled via environment variables (`.env` file) or Pydantic Settings defaults. All numeric fields have validation constraints (e.g., `gt=0`, `ge=0`, `le=2.0`).

| Variable | Default | Constraint | Description |
|----------|---------|-----------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | — | Ollama server URL |
| `OLLAMA_LLM_MODEL` | `llama3.1:8b` | — | LLM for generation and NER |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text-v2-moe` | — | Embedding model (768d) |
| `OLLAMA_EMBED_DIMENSIONS` | `768` | >0 | Expected embedding vector dimension |
| `OLLAMA_TEMPERATURE` | `0.0` | 0.0–2.0 | LLM temperature |
| `OPENSEARCH_HOST` | `localhost` | — | OpenSearch host |
| `OPENSEARCH_PORT` | `9200` | — | OpenSearch port |
| `OPENSEARCH_INDEX` | `rag_chunks` | — | Index name |
| `NEO4J_URI` | `bolt://localhost:7687` | — | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | — | Neo4j username |
| `NEO4J_PASSWORD` | `neo4j` | — | Neo4j password |
| `CHUNK_CHUNK_SIZE` | `512` | >0 | Chunk size in characters |
| `CHUNK_CHUNK_OVERLAP` | `64` | ≥0 | Overlap between chunks |
| `TOP_K_TOP_K_VECTOR` | `10` | >0 | Vector search results |
| `TOP_K_TOP_K_BM25` | `10` | >0 | BM25 search results |
| `TOP_K_TOP_K_GRAPH` | `10` | >0 | Graph search results |
| `TOP_K_TOP_K_FINAL` | `5` | >0 | Final results after fusion |
| `API_KEY` | *(empty)* | — | API key for auth (disabled when empty) |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Lint
ruff check .
```

229 tests, all mocked (no external services required). CI runs `pytest-cov` with 75% minimum coverage.

## Docker Services

```bash
# Start core services
docker compose up -d

# Start with OpenSearch Dashboards
docker compose --profile dashboards up -d

# Stop all
docker compose down
```

| Service | Container | Healthcheck |
|---------|-----------|-------------|
| OpenSearch 2.15 | osgr-opensearch | `/_cluster/health` |
| Neo4j 5 | osgr-neo4j | HTTP 7474 |
| Ollama | osgr-ollama | `/api/tags` |
| Dashboards | osgr-dashboards | Optional (profile) |

## License

MIT
