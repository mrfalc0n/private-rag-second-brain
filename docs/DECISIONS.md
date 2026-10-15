# Decisions

One line per real design decision and why, oldest first. Host names are generic: the **data host** is an always-on Linux box (WSL2 Ubuntu) that holds the index and the API; the **GPU workstation** is a laptop with an integrated GPU that runs the chat model and Open WebUI. A few environment-only decisions are left out of this public copy, so the numbering has gaps.

| ID | Decision | Why |
|---|---|---|
| D-001 | Build the second brain as a pre-built local index instead of live API search | Live search across sources is slow and fragile; a local index is fast and testable |
| D-003 | Split hosts: index, embeddings and API on the data host; chat model and Open WebUI on the GPU workstation | Generation needs the GPU; the index needs uptime and disk; sensitive data stays on one box |
| D-004 | Run the data plane in WSL2 Ubuntu with systemd, not native Windows services | pgvector and systemd timers are simple on Linux; the setup transfers to any Linux server |
| D-005 | Postgres + pgvector as the only store | Vectors, full-text search, metadata and sync state in one database |
| D-007 | Local embeddings (`nomic-embed-text`) through Ollama | Free and private; no document text leaves the data host |
| D-008 | One `Connector` interface and one `Document` metadata schema for every source | Adding a source is one file; nothing after ingestion depends on the source |
| D-009 | Write the eval set before any code | Every later change needs a number to be kept or reverted |
| D-013 | Read-only system; API tokens scoped by source | Limits damage from prompt injection; private sources never reach hosted models |
| D-014 | Hosts talk over Tailscale; the API listens on localhost and the Tailscale address only | Encrypted private network that also works away from home |
| D-015 | LangChain for the pipeline (splitters, embeddings, PGVector, indexing API) | Widely used framework; the indexing API with a record manager gives incremental updates |
| D-017 | Index account identifiers, but tag finance and health sources `sensitivity: private` | Real questions depend on account numbers; the tag keeps those chunks away from cloud clients |
| D-021 | Chat model `qwen3.5:9b` (Q4_K_M) on the integrated Intel Arc GPU through Ollama's Vulkan backend, thinking off | ~9 s per RAG answer with 2,000 tokens of context, against ~100 s on CPU |
| D-022 | Chunks of ~400 tokens (Markdown headings first, then size) prefixed with title and heading path; `search_document:` / `search_query:` prefixes for nomic | Headings keep sections whole; the prefix tells the reranker where a chunk comes from |
| D-023 | Notion through one dedicated read-only integration; database rows rendered as `Name: value` lines; private tagging by parent IDs in `sources.yaml` | Least privilege; rows are where most facts live; tagging by parent needs no per-page edits |
| D-024 | Retrieval API (FastAPI) as a systemd user service; Open WebUI calls it as an OpenAPI tool server with a bearer token | The chat UI needs no custom code; the API is the single place where scopes are enforced |
| D-025 | Hybrid retrieval: pgvector + Postgres full-text merged with reciprocal rank fusion | Vector-only hit@5 was 22% on the Notion questions; hybrid 56% |
| D-027 | Rerank merged candidates with `jinaai/jina-reranker-v1-turbo-en` via fastembed (ONNX, CPU, no PyTorch), keep 6 | Notion hit@5 56% → 78%, MRR 0.32 → 0.67; same quality as MiniLM-L-12 in about half the time |
| D-028 | Keyword search weights each query word by IDF (BM25-style) over a stored `tsvector` column with a GIN index; very common words add score but don't select candidates | `ts_rank_cd` has no IDF, so common words buried rare names and numbers |
| D-029 | Enforce D-017 in the API: each token has `include_private`, and both searches filter on `sensitivity` | A tag nothing filters on protects nothing |
| D-030 | Google Drive connector: desktop OAuth (`drive.readonly`), listed folders only, Changes API cursor; Docs as Markdown, PDF via pypdf, Word via python-docx, Excel and Sheets as table rows; scanned PDFs skipped | Incremental, least privilege, and common office formats become text |
| D-031 | Collect candidates per source (Notion and Drive pools), at most 3 chunks per file, rerank together, then add a recency boost (bigger when the question says latest or current) | Long Drive files crowded out short Notion pages; Drive hit@5 38% → 71%, overall 61% → 78%; correct answers 14 → 18 of 32 |
| D-032 | 20 candidates per source instead of 30 | Answers 20 correct / 1 partial / 11 wrong vs 18 / 4 / 10; median search 8.1 s → 5.3 s |
| D-033 | Local files connector plus fictional sample data and a demo eval | Anyone can run the full pipeline and the eval without Notion or Google accounts |
