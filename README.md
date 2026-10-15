# Private RAG Second Brain

Self-hosted retrieval-augmented generation (RAG) over a personal Notion workspace and Google Drive. It indexes ~5,500 documents (~22,000 chunks) into PostgreSQL + pgvector and answers questions with citations through a local LLM. No document text is sent to a hosted model.

**Results on a 32-question eval over real data:** hybrid search, a cross-encoder reranker, per-source candidate pools and recency weighting raised top-5 retrieval accuracy from 61% to 78% (Google Drive alone: 38% to 71%). Correct answers rose from 14 to 20 of 32, and median search time dropped from 8.1 to 5.3 seconds. Details in [docs/EVALS.md](docs/EVALS.md).

## How it works

```
Notion / Google Drive / local files
   │  scheduled, incremental sync (cursor per source)
   ▼
connectors → metadata contract → heading-aware chunking → Ollama nomic-embed-text → LangChain index()
   ▼
PostgreSQL 16 + pgvector  (chunks, vectors, IDF keyword index, sync state)
   ▲
FastAPI /search: vector + keyword search per source → RRF → cross-encoder rerank → recency boost → top 6
   ▲  bearer token scoped by source and sensitivity, over Tailscale
Open WebUI + Ollama qwen3.5:9b on a GPU workstation → cited answer
```

Two machines: an always-on Linux **data host** (WSL2) holds the index, embeddings and API; a **GPU workstation** runs the chat model and Open WebUI. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for diagrams, tables and the security model, and [docs/DECISIONS.md](docs/DECISIONS.md) for why each choice was made.

## Tech stack

| Layer | Tools |
|---|---|
| RAG pipeline | LangChain (core, community, langchain-ollama, langchain-postgres), indexing API with `SQLRecordManager` |
| Embeddings and reranking | Ollama `nomic-embed-text`; `jina-reranker-v1-turbo-en` via fastembed (ONNX Runtime, CPU) |
| Store and search | PostgreSQL 16, pgvector, Postgres full-text search (IDF-weighted, GIN index), SQLAlchemy, psycopg 3 |
| API | FastAPI, uvicorn, pydantic, httpx |
| Sources | Notion API, Google Drive API v3 (OAuth 2.0, read-only), pypdf, python-docx, openpyxl |
| Chat | Open WebUI (Docker) as an OpenAPI tool client, Ollama `qwen3.5:9b` on an Intel Arc GPU (Vulkan) |
| Operations | Linux / WSL2, systemd services and timers, Tailscale, uv |
| Quality | Custom eval harness (hit@k, MRR, LLM-as-judge), pytest |

## Try it on sample data

The demo indexes the fictional notes in `sample_data/` and runs the eval in `evals/questions.demo.jsonl`. No Notion or Google account is needed.

Prerequisites: Python 3.12+, [uv](https://docs.astral.sh/uv/), PostgreSQL 16 with the [pgvector](https://github.com/pgvector/pgvector) extension, and [Ollama](https://ollama.com/).

```bash
ollama pull nomic-embed-text
createdb second_brain && psql -d second_brain -c "CREATE EXTENSION vector"

cp config/settings.example.yaml config/settings.yaml
cp config/sources.example.yaml config/sources.yaml     # set files.enabled: true (and notion.enabled: false)
echo "DATABASE_URL=postgresql+psycopg://$USER@/second_brain?host=/var/run/postgresql" > .env
echo "API_TOKEN_LOCAL=change-me" >> .env

uv sync
uv run python -m second_brain sync files                # index sample_data/
uv run python -m second_brain search "current collision deductible"
uv run python -m evals.run_retrieval --sources files --questions evals/questions.demo.jsonl
uv run uvicorn second_brain.api:app --port 8765         # then POST /search with "Authorization: Bearer change-me"
```

The first search downloads the reranker model (~150 MB) to `~/.cache/fastembed`. The answer eval (`evals.run_answers`) also needs a chat model: set `chat.base_url` and `chat.model` in `config/settings.yaml`.

## Use it on your own data

- **Notion:** create an internal integration, share your top-level page with it, put the token in `.env` as `NOTION_TOKEN`, then `sync notion`.
- **Google Drive:** create a Desktop OAuth client, save it as `config/google_oauth_client.json`, run `login gdrive`, list folder IDs (`folders <name>`) in `config/sources.yaml`, then `sync gdrive`.
- **Private sources:** list Notion parent IDs or Drive folder IDs as private; only tokens with `include_private: true` can retrieve them.
- **Scheduling:** `deploy/` has systemd user units for the API and a 4-hour Notion sync.

## Repo layout

```
second_brain/
  connectors/   notion.py, gdrive.py, files.py, dummy.py (one file per source)
  models.py     metadata contract (pydantic)
  chunking.py   heading-aware splitting
  sync.py       incremental indexing with LangChain index()
  store.py      PGVector, record manager, tsvector column, sync state
  retrieval.py  hybrid search, per-source pools, reranking, recency
  api.py        FastAPI /search and /health with token scopes
evals/          retrieval and answer evals, demo questions
sample_data/    fictional notes for the demo
config/         example settings and sources
deploy/         systemd units
tests/          pytest
docs/           architecture, decisions, eval results
```

## Privacy

Real configuration, tokens, eval questions and results are gitignored; only `.example` files and fictional sample data are committed. The system is read-only toward every source.

## License

MIT. See [LICENSE](LICENSE).
