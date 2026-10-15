import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from second_brain.models import validate_document


def good_metadata(**overrides):
    meta = {
        "source": "notion",
        "source_id": "0123456789abcdef0123456789abcdef",
        "title": "Investments",
        "url": "https://www.notion.so/0123456789abcdef0123456789abcdef",
        "created_at": "2026-01-05T10:00:00Z",
        "updated_at": "2026-09-30T18:00:00Z",
    }
    meta.update(overrides)
    return meta


def test_valid_document_passes():
    doc = validate_document(Document(page_content="Index funds: brokerage account", metadata=good_metadata()))
    assert doc.metadata["source"] == "notion"
    assert doc.metadata["participants"] == []


def test_unknown_source_is_rejected():
    with pytest.raises(ValidationError):
        validate_document(Document(page_content="x", metadata=good_metadata(source="dropbox")))


def test_missing_title_is_rejected():
    meta = good_metadata()
    del meta["title"]
    with pytest.raises(ValidationError):
        validate_document(Document(page_content="x", metadata=meta))


def test_extra_key_is_rejected():
    with pytest.raises(ValidationError):
        validate_document(Document(page_content="x", metadata=good_metadata(colour="blue")))


# D-017: finance and bills content is tagged private so the cloud token can never see it.
def test_sensitivity_defaults_to_normal():
    doc = validate_document(Document(page_content="x", metadata=good_metadata()))
    assert doc.metadata["sensitivity"] == "normal"


def test_sensitivity_private_is_allowed():
    doc = validate_document(Document(page_content="x", metadata=good_metadata(sensitivity="private")))
    assert doc.metadata["sensitivity"] == "private"


def test_sensitivity_other_values_are_rejected():
    with pytest.raises(ValidationError):
        validate_document(Document(page_content="x", metadata=good_metadata(sensitivity="secret")))
