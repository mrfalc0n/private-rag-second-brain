"""The Connector checklist: what every source (Notion, Gmail, ...) must provide."""

from collections.abc import Iterator
from typing import Protocol
from langchain_core.documents import Document
from second_brain.models import Deletion

class Connector(Protocol):
    name: str                                   # matches sources.yaml key

    def fetch_changes(self, cursor: str | None) -> Iterator[Document | Deletion]:
        """Yield new or changed LangChain Documents and deletions since cursor."""

    def next_cursor(self) -> str:
        """Cursor to save after the batch is indexed successfully."""