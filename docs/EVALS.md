# Evals

Every retrieval change was kept or reverted on these numbers. The real question set is private (it is about the owner's own data); `evals/questions.demo.jsonl` runs the same harness on the fictional sample data.

## Method

- **Question set:** 32 answerable questions over real Notion pages and Drive files, plus 2 that must be answered "I don't know". Types: exact facts, fuzzy recall, temporal ("latest", "current"), cross-source and conflicting records. Each question has gold documents and a reference answer.
- **Retrieval** (`evals/run_retrieval.py`): hit@5 and hit@10 (a gold document in the top 5 or 10 distinct documents), MRR@10, p50 and p95 latency. Gold can be a Notion database ("any row of it") or "the highest-numbered version of a file", resolved at run time.
- **Answers** (`evals/run_answers.py`): retrieve, answer with the local chat model, and grade each answer correct, partial or wrong against the reference with an LLM judge; unanswerable questions must abstain. Judgments are spot-checked by hand.
- Each run is saved to `evals/results/` with the git commit.

## Baseline

| System | Correct (of 20 Notion questions) | Median time |
|---|---|---|
| Previous setup: local model with no retrieval index | 0 | n/a |
| Hosted-model agent searching Notion live | 16 (3 partial, 1 wrong) | ~51 s |

## Retrieval, Notion questions

| Change | hit@5 | MRR@10 |
|---|---|---|
| Vector search only | 22% | |
| + keyword search, merged with RRF (D-025) | 56% | 0.32 |
| + cross-encoder reranker (D-027) | 78% | 0.67 |

## Notion + Google Drive (32 questions)

| Change | hit@5 overall | hit@5 Drive | Correct / partial / wrong | Median search |
|---|---|---|---|---|
| Drive added, shared candidate pool | 61% | 38% | 14 / 4 / 13 | ~3 s |
| Per-source pools, per-file cap, recency boost (D-031) | 78% | 71% | 18 / 4 / 10 | 8.1 s |
| 20 candidates per source (D-032) | 75%\* | | 20 / 1 / 11 | 5.3 s |

\* After three gold answers were updated to newer lab results and holdings; hit@10 81%, MRR@10 0.60. All runs abstained correctly on both unanswerable questions.

## Demo set (fictional sample data)

`uv run python -m evals.run_retrieval --sources files --questions evals/questions.demo.jsonl`: 9 of 9 answerable questions find the gold file at rank 1; p50 0.44 s on CPU.

## Known misses

- Facts deep inside a long file (one tab of a spreadsheet, one line of a long PDF) when only 3 chunks per file reach the reranker.
- Scanned PDFs have no text layer and are skipped (no OCR yet).
- Records that state a security only by ticker are missed by questions that use the company name; the source data now carries both.
