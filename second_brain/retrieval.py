"""Retrieval: one function every client goes through.

mode "vector": pgvector similarity only.
mode "hybrid": vector + Postgres full-text keyword search, merged with reciprocal rank fusion (RRF).
Keyword search matters for short database rows and exact names ("ATT - Mobile", "YouTube Premium"),
which embeddings alone barely tell apart.
"""

import math
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from sqlalchemy import text

from second_brain import store
from second_brain.config import settings

RRF_K = 60  # standard RRF constant: score = sum over rankings of 1 / (RRF_K + rank)
RRF_RECENCY = 0.02  # about one top-rank RRF vote, for a file updated today

# Keyword score = sum of IDF weights of the query words a chunk contains (the core of BM25), plus a small
# ts_rank_cd tiebreak. Plain ts_rank_cd has no IDF, so a common word ("latest", in ~30% of chunks) outranks
# the rare word that matters ("HbA1c", in 11).
TERMS_SQL = text("SELECT lexeme FROM unnest(to_tsvector('english', :query))")
DOC_FREQ_SQL = text("SELECT count(*) FROM langchain_pg_embedding WHERE fts @@ CAST(:term AS tsquery)")
TOTAL_SQL = text("SELECT count(*) FROM langchain_pg_embedding")
COMMON_SHARE = 0.25  # words in more than this share of chunks only add score; they don't pull chunks in

KEYWORD_SQL = text("""
    WITH t AS (SELECT * FROM unnest(CAST(:terms AS text[]), CAST(:idfs AS float8[])) AS t(term, idf)),
    docs AS (
      SELECT e.id, e.document, e.cmetadata, e.fts AS tsv
      FROM langchain_pg_embedding e
      JOIN langchain_pg_collection c ON c.uuid = e.collection_id
      WHERE c.name = :collection
        AND e.fts @@ CAST(:filter AS tsquery)
        AND (CAST(:sources AS text[]) IS NULL OR e.cmetadata->>'source' = ANY(CAST(:sources AS text[])))
        AND (:include_private OR e.cmetadata->>'sensitivity' = 'normal')
    )
    SELECT id, document, cmetadata,
           (SELECT coalesce(sum(t.idf), 0) FROM t WHERE docs.tsv @@ CAST(quote_literal(t.term) AS tsquery))
           + 0.01 * ts_rank_cd(docs.tsv, CAST(:filter AS tsquery)) AS rank
    FROM docs
    ORDER BY rank DESC
    LIMIT :limit
""")

def _hit(doc_id, content: str, meta: dict, score: float) -> dict:
    return {
        "id": str(doc_id),
        "score": score,
        "title": meta.get("title"),
        "heading_path": meta.get("heading_path"),
        "source": meta.get("source"),
        "source_id": meta.get("source_id"),
        "url": meta.get("url"),
        "updated_at": meta.get("updated_at"),
        "sensitivity": meta.get("sensitivity", "normal"),
        "snippet": content,
    }


def vector_search(query: str, k: int, sources: list[str] | None, include_private: bool = True) -> list[dict]:
    conds = ([{"source": {"$in": sources}}] if sources else []) + \
            ([] if include_private else [{"sensitivity": {"$eq": "normal"}}])
    flt = {"$and": conds} if len(conds) > 1 else (conds[0] if conds else None)
    hits = store.vector_store().similarity_search_with_relevance_scores(query, k=k, filter=flt)
    return [_hit(doc.id, doc.page_content, doc.metadata, score) for doc, score in hits]


def keyword_search(query: str, k: int, sources: list[str] | None, include_private: bool = True) -> list[dict]:
    with store.engine().begin() as conn:
        terms = [r.lexeme for r in conn.execute(TERMS_SQL, {"query": query})]
        if not terms:
            return []
        total = conn.execute(TOTAL_SQL).scalar() or 1
        dfs = [conn.execute(DOC_FREQ_SQL, {"term": quote(t)}).scalar() for t in terms]
        idfs = [math.log(1 + (total - df + 0.5) / (df + 0.5)) for df in dfs]
        rare = [t for t, df in zip(terms, dfs) if 0 < df <= COMMON_SHARE * total] or terms
        rows = conn.execute(KEYWORD_SQL, {"terms": terms, "idfs": idfs, "filter": " | ".join(map(quote, rare)),
                                          "collection": store.collection_name(), "sources": sources,
                                          "include_private": include_private, "limit": k}).all()
    return [_hit(r.id, r.document, r.cmetadata, float(r.rank)) for r in rows]


def quote(lexeme: str) -> str:
    """A lexeme as a tsquery literal: 'it''s'."""
    return "'" + lexeme.replace("'", "''") + "'"


@lru_cache
def reranker(model_name: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder  # ONNX on CPU, no PyTorch
    # keep the model out of /tmp so it survives reboots without re-downloading
    return TextCrossEncoder(model_name=model_name, cache_dir=str(Path.home() / ".cache" / "fastembed"))


def rerank(query: str, hits: list[dict], model_name: str) -> list[dict]:
    """Re-score (question, passage) pairs with a cross-encoder, which reads both together."""
    scores = list(reranker(model_name).rerank(query, [h["snippet"] for h in hits]))
    for h, s in zip(hits, scores):
        h["score"] = float(s)
    return sorted(hits, key=lambda h: h["score"], reverse=True)


def search(query: str, k: int | None = None, sources: list[str] | None = None,
           mode: str | None = None, include_private: bool = True) -> list[dict]:
    """include_private=False drops chunks tagged sensitivity: private (D-017: cloud clients never see them).

    Candidates are gathered per source (so long Drive files can't crowd out short Notion pages) with at most
    max_chunks_per_doc from one file, reranked together, then nudged toward recently updated files.
    """
    cfg = settings()["retrieval"]
    k = k or cfg["top_k"]
    mode = mode or cfg["mode"]
    if mode == "vector":
        return vector_search(query, k, sources, include_private)

    pools = [[s] for s in sources] if sources and len(sources) > 1 else [sources]
    candidates = max(cfg.get("candidates", 30), -(-k // len(pools)))  # all pools together must hold k
    pooled: list[dict] = []
    for pool in pools:
        pooled += fuse(query, candidates, pool, include_private, cfg)
    ranked = sorted(pooled, key=lambda h: h["score"], reverse=True)
    if mode == "hybrid_rerank":
        ranked = recency_boost(query, rerank(query, ranked, cfg["reranker_model"]), cfg)
    return ranked[:k]


def fuse(query: str, n: int, sources: list[str] | None, include_private: bool, cfg: dict) -> list[dict]:
    """Vector + keyword rankings merged with RRF; the best n chunks, at most max_chunks_per_doc from one document.

    Looks 3x deeper than n, so the per-document cap frees slots for other files. For "latest/current"
    questions recency counts here too, else an old near-identical file (last decade's sheet) fills the pool.
    """
    per_doc = cfg.get("max_chunks_per_doc", 3)
    fused: dict[str, dict] = {}
    for ranking in (vector_search(query, 3 * n, sources, include_private),
                    keyword_search(query, 3 * n, sources, include_private)):
        for rank, hit in enumerate(ranking, start=1):
            entry = fused.setdefault(hit["id"], {**hit, "score": 0.0})
            entry["score"] += 1 / (RRF_K + rank)
    if TEMPORAL.search(query):
        for hit in fused.values():
            hit["score"] += RRF_RECENCY * freshness(hit, cfg)
    kept, per = [], {}
    for hit in sorted(fused.values(), key=lambda h: h["score"], reverse=True):
        if per.setdefault(hit["source_id"], 0) < per_doc:
            per[hit["source_id"]] += 1
            kept.append(hit)
    return kept[:n]


TEMPORAL = re.compile(r"\b(latest|most recent|newest|current|currently|now|last|recent|upcoming|this year)\b", re.I)


def recency_boost(query: str, hits: list[dict], cfg: dict) -> list[dict]:
    """Add weight * 0.5 ** (age / half_life) to each reranker score; a bigger weight when the question asks
    for the latest/current thing. Reranker scores spread about 1-3 points across the top 30."""
    weight = cfg.get("recency_weight_temporal", 1.0) if TEMPORAL.search(query) else cfg.get("recency_weight", 0.2)
    for h in hits:
        h["score"] += weight * freshness(h, cfg)
    return sorted(hits, key=lambda h: h["score"], reverse=True)


def freshness(hit: dict, cfg: dict) -> float:
    """1.0 for a file updated today, 0.5 one half-life ago, 0 without a date."""
    try:
        updated = datetime.fromisoformat(hit["updated_at"].replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        return 0.0
    age = (datetime.now(timezone.utc) - updated).days
    return 0.5 ** (max(age, 0) / cfg.get("recency_half_life_days", 365))
