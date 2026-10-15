"""The metadata contract: the labels every Document must carry before it is indexed.

See docs/ARCHITECTURE.md, section 4.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from langchain_core.documents import Document
from pydantic import BaseModel, Field

# The only allowed values for `source`. Must match the keys in config/sources.yaml.
SourceName = Literal["notion", "gdrive", "files", "gmail", "outlook", "sms_backup"]


class DocMetadata(BaseModel):
    """Required and optional metadata keys. Pydantic checks types and rejects anything else."""

    source: SourceName
    source_id: str = Field(min_length=1)       # Notion page ID, Gmail message ID, ...
    title: str = Field(min_length=1)           # page title, email subject
    url: str                                   # link back to the original, for citations
    created_at: datetime
    updated_at: datetime
    participants: list[str] = []               # people on an email or text thread
    thread_id: str | None = None               # Gmail / Outlook thread
    heading_path: str | None = None            # added later by the splitter
    sensitivity: Literal["normal", "private"] = "normal"
    model_config = {"extra": "forbid"}         # unknown keys are an error, not silently kept


@dataclass
class Deletion:
    """A connector yields this when a document was deleted at the source."""

    source: SourceName
    source_id: str


def validate_document(doc: Document) -> Document:
    """Check a Document's metadata against the contract. Raises ValueError if it fails.

    Returns the Document with its metadata cleaned up (dates as ISO strings),
    ready to store as JSON in Postgres.
    """
    meta = DocMetadata.model_validate(doc.metadata)
    doc.metadata = meta.model_dump(mode="json")
    return doc
