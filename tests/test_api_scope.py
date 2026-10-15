"""D-017: the cloud token never gets private chunks; the local token does."""

from fastapi.testclient import TestClient

from second_brain import api


def test_private_only_for_local_token(monkeypatch):
    monkeypatch.setattr(api, "load_env", lambda: None)
    monkeypatch.setenv("TOK_LOCAL", "local-secret")
    monkeypatch.setenv("TOK_CLOUD", "cloud-secret")
    monkeypatch.setattr(api, "settings", lambda: {"api": {"tokens": [
        {"token_env": "TOK_LOCAL", "allowed_sources": ["notion", "gdrive"], "include_private": True},
        {"token_env": "TOK_CLOUD", "allowed_sources": ["notion"]},  # include_private left out -> false
    ]}})
    calls = []
    monkeypatch.setattr(api, "do_search", lambda q, k, sources, include_private: calls.append((sources, include_private)) or [])
    client = TestClient(api.app)

    assert client.post("/search", json={"query": "x"}, headers={"Authorization": "Bearer cloud-secret"}).status_code == 200
    assert client.post("/search", json={"query": "x", "sources": ["gdrive"]},
                       headers={"Authorization": "Bearer cloud-secret"}).json() == {"results": []}
    client.post("/search", json={"query": "x"}, headers={"Authorization": "Bearer local-secret"})
    assert calls[0] == (["notion"], False) and set(calls[1][0]) == {"notion", "gdrive"} and calls[1][1] is True
    assert client.post("/search", json={"query": "x"}, headers={"Authorization": "Bearer wrong"}).status_code == 401
