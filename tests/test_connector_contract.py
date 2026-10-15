"""Session 2 exit test: a pretend connector follows the Connector checklist,
and what it hands over passes the metadata rules in models.py."""

from langchain_core.documents import Document

from second_brain.connectors.base import Connector
from second_brain.models import Deletion, validate_document


class DummyConnector:
    """A fake source. Real connectors (Notion, Gmail, ...) will look like this, but call an API."""

    name = "notion"

    def fetch_changes(self, cursor):
        yield Document(
            page_content="Index funds are held in the brokerage account and the retirement account.",
            metadata={
                "source": "notion",
                "source_id": "page-001",
                "title": "Investments",
                "url": "https://app.notion.com/p/page-001",
                "created_at": "2026-01-05T10:00:00Z",
                "updated_at": "2026-09-30T18:00:00Z",
            },
        )
        yield Deletion(source="notion", source_id="page-999")

    def next_cursor(self):
        return "2026-09-30T18:00:00Z"


def run_sync(connector: Connector):
    """A tiny stand-in for the real sync runner: collect what the connector hands over."""
    documents, deletions = [], []
    for item in connector.fetch_changes(cursor=None):
        if isinstance(item, Deletion):
            deletions.append(item)
        else:
            documents.append(validate_document(item))
    return documents, deletions, connector.next_cursor()


def test_dummy_connector_follows_the_contract():
    documents, deletions, cursor = run_sync(DummyConnector())

    assert len(documents) == 1
    assert documents[0].metadata["sensitivity"] == "normal"   # default from models.py
    assert deletions == [Deletion(source="notion", source_id="page-999")]
    assert cursor == "2026-09-30T18:00:00Z"
