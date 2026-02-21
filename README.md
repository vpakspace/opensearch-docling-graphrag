# OpenSearch Docling GraphRAG

Fully local RAG pipeline combining **OpenSearch** (hybrid BM25 + k-NN vector search), **Neo4j** (knowledge graph), **Ollama** (LLM + embeddings), and **Docling** (document parsing). No cloud API keys required — **88% benchmark accuracy** (106/120), 120 tests, 9 commits, 5,464 LOC.

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
    Retriever (4 modes: bm25 / vector / graph / hybrid RRF)
         │
         ▼
    Generator (Ollama LLM) ──► Answer + Sources + Confidence
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
| REST API | FastAPI | 8508 |
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

# Ask a question
curl -X POST http://localhost:8508/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"text": "What is OpenSearch?", "mode": "hybrid"}'
```

## Search Modes

| Mode | How it works |
|------|-------------|
| `bm25` | Full-text search via OpenSearch BM25 |
| `vector` | Semantic search via OpenSearch k-NN (cosine, HNSW) |
| `graph` | Entity match in Neo4j → traverse to chunks |
| `hybrid` | RRF fusion of bm25 + vector + graph results |

## Benchmark

30 questions (Russian + English) x 4 modes = 120 evaluations. Keyword overlap judge (no external API).

| Mode | Score | Doc1 (RU) | Doc2 (EN) | Avg Latency |
|------|-------|-----------|-----------|-------------|
| `hybrid` | **28/30 (93%)** | 13/15 | 15/15 | 3.1s |
| `bm25` | 28/30 (93%) | 14/15 | 14/15 | 4.1s |
| `vector` | 27/30 (90%) | 12/15 | 15/15 | 4.1s |
| `graph` | 23/30 (76%) | 12/15 | 11/15 | 3.4s |
| **Overall** | **106/120 (88%)** | **51/60** | **55/60** | |

LLM: `llama3.1:8b` | Embeddings: `nomic-embed-text-v2-moe` | GPU: RTX 4080

```bash
python scripts/run_benchmark.py
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | Service health (OpenSearch, Neo4j, Ollama) |
| `POST` | `/api/v1/query` | RAG query → answer + sources + confidence |
| `POST` | `/api/v1/search` | Search only (no generation) |
| `GET` | `/api/v1/graph/stats` | Knowledge graph statistics |

## Streamlit UI (6 Tabs)

| Tab | Features |
|-----|----------|
| Home | Service status, document/chunk/entity counts |
| Upload | Drag-and-drop ingestion, optional NER, GPU toggle |
| Search & Q&A | 4 search modes, confidence bar, expandable sources |
| Graph Explorer | Interactive PyVis visualization, entity type filter |
| Batch Process | Directory ingestion with progress bar |
| Settings | Configuration display, clear index / clear graph |

The UI supports **English and Russian** (sidebar language selector).

## Project Structure

```
opensearch-docling-graphrag/
├── opensearch_graphrag/           # Core pipeline
│   ├── config.py                  # Pydantic Settings
│   ├── models.py                  # Chunk, Entity, SearchResult, QAResult
│   ├── loader.py                  # Docling document parser
│   ├── chunker.py                 # Markdown-aware splitting
│   ├── embedder.py                # Ollama embeddings (POST /api/embed)
│   ├── opensearch_store.py        # k-NN index + BM25 + hybrid search
│   ├── entity_extractor.py        # NER via Ollama LLM
│   ├── graph_builder.py           # Neo4j graph construction
│   ├── retriever.py               # 4-mode retriever + RRF fusion
│   ├── generator.py               # Ollama chat generation
│   └── service.py                 # PipelineService orchestrator
├── api/
│   ├── app.py                     # FastAPI factory + lifespan
│   ├── routes.py                  # REST endpoints
│   └── deps.py                    # Dependency injection
├── ui/
│   ├── streamlit_app.py           # 6-tab UI
│   ├── i18n.py                    # EN/RU translations
│   └── components/graph_viz.py    # PyVis rendering
├── scripts/
│   ├── ingest.py                  # CLI ingestion
│   ├── run_benchmark.py           # 30-question benchmark runner
│   └── pull_models.sh             # Download Ollama models
├── benchmark/                     # Benchmark data
│   ├── questions.json             # 30 questions (RU + EN)
│   └── results.json               # Latest benchmark results
├── tests/                         # 120 tests
├── data/                          # Sample documents (Doc1 RU + Doc2 EN)
├── docker-compose.yml             # OpenSearch + Neo4j + Ollama
├── requirements.txt
├── pyproject.toml
├── run_api.py                     # uvicorn launcher (port 8508)
└── .env.example
```

## Configuration

All settings are controlled via environment variables (`.env` file) or Pydantic Settings defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_LLM_MODEL` | `llama3.1:8b` | LLM for generation and NER |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text-v2-moe` | Embedding model (768d) |
| `OPENSEARCH_HOST` | `localhost` | OpenSearch host |
| `OPENSEARCH_PORT` | `9200` | OpenSearch port |
| `OPENSEARCH_INDEX` | `rag_chunks` | Index name |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `neo4j` | Neo4j password |
| `CHUNK_SIZE` | `512` | Chunk size in characters |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K_VECTOR` | `10` | Vector search results |
| `TOP_K_BM25` | `10` | BM25 search results |
| `TOP_K_GRAPH` | `10` | Graph search results |
| `TOP_K_FINAL` | `5` | Final results after fusion |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Lint
ruff check .
```

120 tests, all mocked (no external services required).

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
