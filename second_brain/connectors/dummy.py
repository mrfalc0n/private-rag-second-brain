"""A fake source for testing the pipeline end to end without any API."""

from langchain_core.documents import Document

from second_brain.models import Deletion


class DummyConnector:
    name = "dummy"

    def fetch_changes(self, cursor):
        yield Document(
            page_content="# Investments\n\n## Stocks\n\nIndex funds are held in the brokerage account and the retirement account.",
            metadata={"source": "notion", "source_id": "dummy-001", "title": "Dummy Investments",
                      "url": "https://example.invalid/dummy-001",
                      "created_at": "2026-01-05T10:00:00Z", "updated_at": "2026-09-30T18:00:00Z"},
        )
        yield Deletion(source="notion", source_id="dummy-999")

    def next_cursor(self):
        return "dummy-cursor"
