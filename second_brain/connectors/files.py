"""Local files connector: Markdown and text files in one folder (and its subfolders).

Made for the demo (sample_data/) and for notes kept as plain files. The cursor is the newest
modification time seen, so later runs pick up only new or edited files. A full run (--full) re-reads
everything. Files deleted from the folder are not detected; run with --full after deleting files.
Files under a subfolder listed in private_subfolders are tagged sensitivity: private (D-017).
"""

import logging
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document

from second_brain.config import ROOT, sources
from second_brain.models import Deletion

log = logging.getLogger(__name__)
SUFFIXES = {".md", ".txt"}


class FilesConnector:
    name = "files"

    def __init__(self):
        cfg = sources()["files"]
        self.root = (ROOT / cfg["path"]).resolve()
        self.private = [self.root / p for p in cfg.get("private_subfolders", [])]
        self._next = 0.0

    def fetch_changes(self, cursor: str | None) -> Iterator[Document | Deletion]:
        since = float(cursor or 0)
        self._next = since
        for path in sorted(self.root.rglob("*")):
            if path.suffix.lower() not in SUFFIXES or not path.is_file():
                continue
            mtime = path.stat().st_mtime
            self._next = max(self._next, mtime)
            if mtime <= since:
                continue
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                continue
            rel = path.relative_to(self.root).as_posix()
            if not text.startswith("# "):
                text = f"# {path.stem}\n\n{text}"
            stamp = datetime.fromtimestamp(mtime, timezone.utc).isoformat()
            log.info("file: %s (%d chars)", rel, len(text))
            yield Document(page_content=text, metadata={
                "source": "files",
                "source_id": rel,
                "title": path.stem,
                "url": path.as_uri(),
                "created_at": stamp,
                "updated_at": stamp,
                "sensitivity": "private" if any(path.is_relative_to(p) for p in self.private) else "normal",
            })

    def next_cursor(self) -> str:
        return str(self._next)
