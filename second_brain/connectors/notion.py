"""Notion connector: pages shared with the integration, newest edits first, as Markdown Documents.

Database rows are pages too; their properties are rendered as "Name: value" lines above the body,
after a "Database: <name>" line so questions like "which subscriptions are flagged for review" can find them.
Cursor = the newest last_edited_time seen. Deletions are not detected yet (weekly reconcile is a TODO).
"""

import logging
import time
from collections.abc import Iterator

import httpx
from langchain_core.documents import Document

from second_brain.config import secret, sources
from second_brain.models import Deletion

log = logging.getLogger(__name__)
API = "https://api.notion.com/v1"
NOTION_VERSION = "2026-03-11"
MIN_INTERVAL = 0.35  # Notion allows about 3 requests per second


class NotionConnector:
    name = "notion"

    def __init__(self):
        cfg = sources()["notion"]
        # D-017: pages whose own id or parent id is listed here are tagged sensitivity: private
        self.private_ids = {str(i).replace("-", "") for i in cfg.get("private_parent_ids") or []}
        self.client = httpx.Client(timeout=30, headers={
            "Authorization": f"Bearer {secret(cfg['token_env'])}",
            "Notion-Version": NOTION_VERSION,
        })
        self._last_call = 0.0
        self._newest_seen: str | None = None
        self._start_cursor: str | None = None
        self._db_titles: dict[str, str] = {}

    # --- HTTP with rate limiting and retries -------------------------------------------------
    def _request(self, method: str, path: str, **kwargs) -> dict:
        for attempt in range(6):
            wait = MIN_INTERVAL - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            r = self.client.request(method, API + path, **kwargs)
            if r.status_code == 429 or r.status_code >= 500:
                delay = float(r.headers.get("Retry-After", 2 ** attempt))
                log.warning("Notion %s on %s, retrying in %.0fs", r.status_code, path, delay)
                time.sleep(delay)
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()
        return r.json()

    # --- Connector protocol ------------------------------------------------------------------
    def fetch_changes(self, cursor: str | None) -> Iterator[Document | Deletion]:
        self._start_cursor = cursor
        for page in self._pages_edited_since(cursor):
            edited = page["last_edited_time"]
            if self._newest_seen is None or edited > self._newest_seen:
                self._newest_seen = edited
            if page.get("in_trash") or page.get("archived"):
                yield Deletion(source="notion", source_id=page["id"])
                continue
            try:
                yield self._page_to_document(page)
            except httpx.HTTPStatusError as e:
                log.warning("skipping page %s: %s", page["id"], e)

    def next_cursor(self) -> str:
        return self._newest_seen or self._start_cursor or ""

    # --- Listing -----------------------------------------------------------------------------
    def _pages_edited_since(self, cursor: str | None) -> Iterator[dict]:
        body = {"filter": {"property": "object", "value": "page"},
                "sort": {"timestamp": "last_edited_time", "direction": "descending"},
                "page_size": 100}
        while True:
            data = self._request("POST", "/search", json=body)
            for page in data["results"]:
                if cursor and page["last_edited_time"] < cursor:
                    return  # sorted newest first, so everything after this is older
                yield page
            if not data.get("has_more"):
                return
            body["start_cursor"] = data["next_cursor"]

    # --- Page -> Document --------------------------------------------------------------------
    def _database_title(self, parent: dict) -> str | None:
        """Name of the database a row belongs to (one lookup per database, then cached)."""
        ds_id, db_id = parent.get("data_source_id"), parent.get("database_id")
        if not (ds_id or db_id):
            return None
        key = ds_id or db_id
        if key not in self._db_titles:
            try:
                data = self._request("GET", f"/data_sources/{ds_id}" if ds_id else f"/databases/{db_id}")
                self._db_titles[key] = plain(data.get("title", [])).strip()
            except httpx.HTTPStatusError as e:
                log.warning("no title for database %s: %s", key, e)
                self._db_titles[key] = ""
        return self._db_titles[key] or None

    def _page_to_document(self, page: dict) -> Document:
        title, props = "", []
        parent = page.get("parent", {})
        if db_title := self._database_title(parent):
            props.append(f"- **Database:** {db_title}")
        for name, prop in page.get("properties", {}).items():
            value = render_property(prop)
            if prop["type"] == "title":
                title = value
            elif value:
                props.append(f"- **{name}:** {value}")
        title = title.strip() or "Untitled"
        body = "\n".join(self._blocks_to_markdown(page["id"]))
        text = f"# {title}\n\n" + ("\n".join(props) + "\n\n" if props else "") + body
        parent_id = (parent.get("data_source_id") or parent.get("database_id") or parent.get("page_id") or "").replace("-", "")
        page_id = page["id"].replace("-", "")
        private = page_id in self.private_ids or parent_id in self.private_ids
        log.info("notion page: %s (%d chars)", title, len(text))
        return Document(page_content=text.strip(), metadata={
            "source": "notion",
            "source_id": page_id,
            "title": title,
            "url": page.get("url", ""),
            "created_at": page["created_time"],
            "updated_at": page["last_edited_time"],
            "sensitivity": "private" if private else "normal",
        })

    def _blocks_to_markdown(self, block_id: str, depth: int = 0) -> list[str]:
        lines, params = [], {"page_size": 100}
        while True:
            data = self._request("GET", f"/blocks/{block_id}/children", params=params)
            for block in data["results"]:
                line = render_block(block)
                if line is not None:
                    lines.append("  " * depth + line)
                if block.get("has_children") and block["type"] not in ("child_page", "child_database"):
                    child_depth = depth + 1 if block["type"] not in ("column_list", "column", "synced_block") else depth
                    lines.extend(self._blocks_to_markdown(block["id"], child_depth))
            if not data.get("has_more"):
                return lines
            params["start_cursor"] = data["next_cursor"]


def plain(rich_text: list[dict]) -> str:
    return "".join(t.get("plain_text", "") for t in rich_text or [])


def render_block(block: dict) -> str | None:
    t = block["type"]
    b = block.get(t, {})
    text = plain(b.get("rich_text", []))
    simple = {
        "paragraph": text, "heading_1": f"# {text}", "heading_2": f"## {text}", "heading_3": f"### {text}",
        "bulleted_list_item": f"- {text}", "numbered_list_item": f"1. {text}", "toggle": f"- {text}",
        "quote": f"> {text}", "callout": f"> {text}",
    }
    if t in simple:
        return simple[t]
    if t == "to_do":
        return f"- [{'x' if b.get('checked') else ' '}] {text}"
    if t == "code":
        return f"```\n{text}\n```"
    if t == "table_row":
        return "| " + " | ".join(plain(cell) for cell in b.get("cells", [])) + " |"
    if t in ("bookmark", "link_preview", "embed"):
        return b.get("url")
    if t == "equation":
        return b.get("expression")
    if t == "child_page":
        return f"(sub-page: {b.get('title', '')})"
    if t == "child_database":
        return f"(database: {b.get('title', '')})"
    if t == "divider":
        return "---"
    return None  # images, files, columns and other layout blocks carry no text of their own


def render_property(prop: dict) -> str:
    t, v = prop["type"], prop.get(prop["type"])
    if v is None:
        return ""
    if t in ("title", "rich_text"):
        return plain(v)
    if t in ("select", "status"):
        return v.get("name", "")
    if t == "multi_select":
        return ", ".join(o["name"] for o in v)
    if t == "date":
        return v["start"] + (f" to {v['end']}" if v.get("end") else "")
    if t == "checkbox":
        return "yes" if v else "no"
    if t == "people":
        return ", ".join(p.get("name", "") for p in v)
    if t == "formula":
        inner = v.get(v.get("type"))
        return inner.get("start", "") if isinstance(inner, dict) else ("" if inner is None else str(inner))
    if t == "rollup":
        if v.get("type") == "array":
            return ", ".join(filter(None, (render_property(x) for x in v["array"])))
        inner = v.get(v.get("type"))
        return inner.get("start", "") if isinstance(inner, dict) else ("" if inner is None else str(inner))
    if t == "unique_id":
        return f"{v.get('prefix') or ''}{'-' if v.get('prefix') else ''}{v.get('number')}"
    if t in ("number", "url", "email", "phone_number", "created_time", "last_edited_time"):
        return str(v)
    return ""  # relations, files, buttons: no useful text
