"""Split validated Documents into chunks that keep their title and heading path."""

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from second_brain.config import settings

CHARS_PER_TOKEN = 4  # rough average for English; good enough for sizing chunks
HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def split_document(doc: Document) -> list[Document]:
    cfg = settings()["chunking"]
    by_heading = MarkdownHeaderTextSplitter(HEADERS, strip_headers=False).split_text(doc.page_content)
    by_size = RecursiveCharacterTextSplitter(
        chunk_size=cfg["target_tokens"] * CHARS_PER_TOKEN,
        chunk_overlap=cfg["overlap_tokens"] * CHARS_PER_TOKEN,
    )
    chunks = []
    for section in by_heading:
        heading_path = " > ".join(section.metadata[k] for _, k in HEADERS if k in section.metadata)
        for piece in by_size.split_text(section.page_content):
            prefix = doc.metadata["title"] + (f" > {heading_path}" if heading_path else "")
            meta = {**doc.metadata, "heading_path": heading_path or None}
            chunks.append(Document(page_content=f"{prefix}\n\n{piece}", metadata=meta))
    return chunks
