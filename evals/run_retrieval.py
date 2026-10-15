"""Retrieval eval: does the gold document show up in the top 5 / top 10? (hit@5, hit@10, MRR@10)

Usage: uv run python -m evals.run_retrieval [--sources notion] [--questions evals/private/questions.jsonl]

Only questions whose gold documents come from the indexed sources are scored. A question counts
as a hit if any of its gold documents appears; ranks are by document, not by chunk.
A question with "gold_latest": "<file name>" means "the highest -vN file of that name" (see resolve_latest),
so "what's my latest resume?" stays right when a new version lands.
Results go to evals/results/ (gitignored) with the git commit, so runs can be compared.
"""

import argparse
import json
import re
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

from sqlalchemy import text

from second_brain import store
from second_brain.retrieval import search

ROOT = Path(__file__).resolve().parent.parent
NOTION_ID = re.compile(r"([0-9a-f]{32})(?:[/?#]|$)")
DRIVE_ID = re.compile(r"/file/d/([\w-]+)")
TITLES_SQL = text("""SELECT DISTINCT cmetadata->>'source_id' AS id, cmetadata->>'title' AS title,
                     cmetadata->>'updated_at' AS updated FROM langchain_pg_embedding WHERE cmetadata->>'source' = 'gdrive'
                     AND cmetadata->>'title' ~* :pattern""")


def resolve_latest(q: dict) -> dict:
    """gold_latest "X" -> gold_docs = every copy of the newest "X" / "X-vN" Drive file (highest N, then newest
    edit); "{latest}" in reference_answer becomes that file name."""
    name = q.get("gold_latest")
    if not name:
        return q
    with store.engine().begin() as conn:
        rows = conn.execute(TITLES_SQL, {"pattern": f"^{re.escape(name)}(-v[0-9]+)?$"}).all()
    if not rows:
        return q
    version = lambda t: int(m.group(1)) if (m := re.search(r"-v(\d+)$", t)) else 1
    best = max(rows, key=lambda r: (version(r.title), r.updated)).title
    q["gold_docs"] = [f"https://drive.google.com/file/d/{r.id}/view" for r in rows if r.title == best]
    q["reference_answer"] = q["reference_answer"].replace("{latest}", best)
    return q


def gold_ids(url: str) -> tuple[str, str] | None:
    """Turn a gold link into (source, source_id) as stored in the index."""
    if "notion" in url:
        m = NOTION_ID.search(url.replace("-", ""))
        return ("notion", m.group(1)) if m else None
    if "drive.google.com" in url:
        m = DRIVE_ID.search(url)
        return ("gdrive", m.group(1)) if m else None
    if url.startswith("files:"):  # local files connector: "files:<path relative to its folder>"
        return ("files", url.removeprefix("files:"))
    return None  # email and text-message gold docs need their connectors first


CACHE = ROOT / "evals/private/notion_gold_expansion.json"


def expand_notion_databases(ids: set[str]) -> dict[str, set[str]]:
    """A gold link to a Notion database means "any row of it". Map database id -> its row page ids.

    Rows are looked up once through the Notion API and cached in evals/private (gitignored).
    """
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    missing = [i for i in ids if i not in cache]
    if missing:
        import httpx
        from second_brain.connectors.notion import NotionConnector
        c = NotionConnector()
        for db_id in missing:
            try:
                db = c._request("GET", f"/databases/{db_id}")
            except httpx.HTTPStatusError:
                cache[db_id] = []  # a page, not a database
                continue
            rows = []
            for ds in db.get("data_sources", []):
                body = {"page_size": 100}
                while True:
                    data = c._request("POST", f"/data_sources/{ds['id']}/query", json=body)
                    rows += [r["id"].replace("-", "") for r in data["results"]]
                    if not data.get("has_more"):
                        break
                    body["start_cursor"] = data["next_cursor"]
            cache[db_id] = rows
        CACHE.write_text(json.dumps(cache, indent=1))
    return {i: set(cache[i]) for i in ids if cache.get(i)}


def doc_ranking(query: str, sources: list[str], k_chunks: int = 30) -> list[str]:
    """Search, then collapse chunks into a ranked list of distinct source documents."""
    seen = []
    for hit in search(query, k=k_chunks, sources=sources):
        if hit["source_id"] not in seen:
            seen.append(hit["source_id"])
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["notion"])
    ap.add_argument("--questions", default=str(ROOT / "evals/private/questions.jsonl"))
    args = ap.parse_args()

    questions = [resolve_latest(json.loads(line)) for line in open(args.questions, encoding="utf-8") if line.strip()]
    notion_gold = {g[1] for q in questions for g in map(gold_ids, q["gold_docs"]) if g and g[0] == "notion"}
    db_rows = expand_notion_databases(notion_gold) if "notion" in args.sources else {}
    rows, latencies = [], []
    for q in questions:
        if not q.get("scored", True) or q["type"] == "unanswerable":
            continue
        gold = {g[1] for g in map(gold_ids, q["gold_docs"]) if g and g[0] in args.sources}
        if not gold:
            continue
        for db_id in list(gold):
            gold |= db_rows.get(db_id, set())
        t0 = time.perf_counter()
        ranking = doc_ranking(q["question"], args.sources)
        latencies.append(time.perf_counter() - t0)
        rank = next((i + 1 for i, d in enumerate(ranking[:10]) if d in gold), None)
        rows.append({"id": q["id"], "type": q["type"], "rank": rank, "question": q["question"]})

    n = len(rows)
    hit5 = sum(1 for r in rows if r["rank"] and r["rank"] <= 5) / n if n else 0
    hit10 = sum(1 for r in rows if r["rank"]) / n if n else 0
    mrr = sum(1 / r["rank"] for r in rows if r["rank"]) / n if n else 0
    p50 = statistics.median(latencies) if latencies else 0
    p95 = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)] if latencies else 0

    for r in rows:
        mark = "✓" if r["rank"] and r["rank"] <= 5 else ("~" if r["rank"] else "✗")
        print(f"{mark} {r['id']:4} rank={r['rank'] or '-':>2}  {r['type']:13} {r['question'][:70]}")
    print(f"\n{n} questions | hit@5 {hit5:.0%} | hit@10 {hit10:.0%} | MRR@10 {mrr:.2f} | "
          f"latency p50 {p50 * 1000:.0f} ms, p95 {p95 * 1000:.0f} ms")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    out = ROOT / "evals/results" / f"{datetime.now():%Y%m%d-%H%M%S}_retrieval_{commit}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"commit": commit, "sources": args.sources, "n": n, "hit@5": hit5, "hit@10": hit10,
                               "mrr@10": mrr, "p50_s": p50, "p95_s": p95, "rows": rows}, indent=2))
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
