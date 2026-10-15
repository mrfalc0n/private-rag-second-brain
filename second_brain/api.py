"""Retrieval API: POST /search for any client (OpenWebUI tool server, MCP later), bearer-token protected.

Run: uv run uvicorn second_brain.api:app --host 127.0.0.1 --port 8765
"""

import logging
import os
import secrets as pysecrets
import time
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from second_brain import store
from second_brain.config import load_env, settings
from second_brain.retrieval import search as do_search

log = logging.getLogger("second_brain.api")
SourceName = Literal["notion", "gdrive", "files", "gmail", "outlook", "sms_backup"]

app = FastAPI(
    title="Second Brain",
    version="0.1.0",
    description="Search the owner's private notes and files (Notion and Google Drive). "
                "Returns passages with title, source, date and link for citation.",
)


class Scope(BaseModel):
    sources: set[str]
    include_private: bool


def token_scope(authorization: str = Header(default="")) -> Scope:
    """Map the bearer token to the sources it may see and whether private chunks are included
    (settings.yaml api.tokens; include_private defaults to false, D-017)."""
    load_env()
    given = authorization.removeprefix("Bearer ").strip()
    for t in settings()["api"]["tokens"]:
        expected = os.environ.get(t["token_env"])
        if expected and given and pysecrets.compare_digest(given, expected):
            return Scope(sources=set(t["allowed_sources"]), include_private=bool(t.get("include_private", False)))
    raise HTTPException(status_code=401, detail="Missing or invalid bearer token")


class SearchRequest(BaseModel):
    query: str = Field(description="What to look for, in plain words (e.g. 'Wells Fargo Reflect promo end date').")
    k: int = Field(default=6, ge=1, le=20, description="How many passages to return.")
    sources: list[SourceName] | None = Field(default=None, description="Limit to these sources; omit for all.")


class Hit(BaseModel):
    score: float
    title: str | None
    heading_path: str | None
    source: str | None
    url: str | None
    updated_at: str | None
    text: str


class SearchResponse(BaseModel):
    results: list[Hit]


@app.post("/search", response_model=SearchResponse, operation_id="search_second_brain",
          summary="Search my private notes and return cited passages")
def search(req: SearchRequest, scope: Scope = Depends(token_scope)) -> SearchResponse:
    """Search the owner's personal knowledge base. Use for any question about their accounts, bills,
    health records, projects, decisions or notes. Cite the title and URL of each passage used."""
    wanted = [s for s in (req.sources or scope.sources) if s in scope.sources]
    if not wanted:
        return SearchResponse(results=[])
    t0 = time.perf_counter()
    hits = do_search(req.query, k=req.k, sources=wanted, include_private=scope.include_private)
    log.info("search %.0f ms k=%d sources=%s private=%s q=%r", (time.perf_counter() - t0) * 1000, req.k,
             wanted, scope.include_private, req.query)
    return SearchResponse(results=[Hit(text=h["snippet"], **{k: h[k] for k in
                                       ("score", "title", "heading_path", "source", "url", "updated_at")})
                                   for h in hits])


@app.get("/health", summary="Per-source sync status")
def health() -> dict:
    return {"status": "ok", "sources": store.status()}
