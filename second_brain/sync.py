"""The sync runner: connector -> validate -> chunk -> index() -> save cursor."""

import fcntl
import logging
from pathlib import Path

from langchain_core.indexing import index

from second_brain import store
from second_brain.chunking import split_document
from second_brain.connectors.base import Connector
from second_brain.models import Deletion, validate_document

log = logging.getLogger(__name__)


def delete_source_ids(source_ids: list[str]) -> int:
    """Remove every chunk that came from these source documents."""
    rm, vs = store.record_manager(), store.vector_store()
    keys = rm.list_keys(group_ids=source_ids)
    if keys:
        vs.delete(ids=keys)
        rm.delete_keys(keys)
    return len(keys)


DOCS_PER_BATCH = 50  # index as we go, so a failure late in a long first sync keeps earlier work


def run(connector: Connector, full: bool = False) -> dict:
    """Sync one source. The cursor is saved only if everything succeeded.

    Chunks are indexed every DOCS_PER_BATCH documents; a rerun after a failure re-fetches,
    but index() skips chunks that are already stored, so only the remainder is embedded.
    """
    lock = open(Path("/tmp") / f"second-brain-sync-{connector.name}.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # one sync per source at a time
    except BlockingIOError:
        raise SystemExit(f"a {connector.name} sync is already running; skipping")
    store.ensure_schema()
    cursor = None if full else store.get_cursor(connector.name)
    totals = {"num_added": 0, "num_updated": 0, "num_skipped": 0, "num_deleted": 0}
    try:
        batch, deleted_ids, n_docs, n_chunks = [], [], 0, 0

        def flush():
            nonlocal batch
            if batch:
                result = index(batch, store.record_manager(), store.vector_store(),
                               cleanup="incremental", source_id_key="source_id", batch_size=32)
                for k in totals:
                    totals[k] += result[k]
                log.info("indexed batch: %s | %d docs so far", result, n_docs)
                batch = []

        for item in connector.fetch_changes(cursor):
            if isinstance(item, Deletion):
                deleted_ids.append(item.source_id)
                continue
            doc = validate_document(item)
            n_docs += 1
            chunks = split_document(doc)
            n_chunks += len(chunks)
            batch.extend(chunks)
            if n_docs % DOCS_PER_BATCH == 0:
                flush()
        flush()

        removed = delete_source_ids(deleted_ids) if deleted_ids else 0
        store.save_success(connector.name, connector.next_cursor(), n_docs)
        return {"documents": n_docs, "chunks": n_chunks, **totals, "deleted_docs": len(deleted_ids),
                "deleted_chunks": removed}
    except Exception as e:
        store.save_error(connector.name, repr(e))
        raise
