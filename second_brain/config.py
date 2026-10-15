"""Load config/settings.yaml, config/sources.yaml and secrets from .env."""

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path = ROOT / ".env") -> None:
    """Read KEY=value lines from .env into the environment (existing variables win)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


@lru_cache
def settings() -> dict:
    load_env()
    return yaml.safe_load((ROOT / "config" / "settings.yaml").read_text())


@lru_cache
def sources() -> dict:
    load_env()
    return yaml.safe_load((ROOT / "config" / "sources.yaml").read_text())


def secret(env_name: str) -> str:
    """Return a secret from the environment, or fail with a clear message."""
    load_env()
    value = os.environ.get(env_name)
    if not value:
        raise RuntimeError(f"{env_name} is not set. Add it to .env (see config/*.example.yaml).")
    return value


def database_url() -> str:
    return secret(settings()["database"]["dsn_env"])
