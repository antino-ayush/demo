"""Loads .env once, before anything reads os.environ for endpoints/tokens.

Centralized here so cli.py and generator.py call load_env() instead of each
doing their own find_dotenv/load_dotenv dance. Provider/config classes
(NemotronJudgeProvider, ServiceConfig, ...) deliberately do NOT call this
themselves -- they just read os.environ, same as ClaudeJudgeProvider /
OpenAIJudgeProvider already did, which keeps them trivially testable with
plain monkeypatch.setenv/delenv and no .env file involved.
"""
from __future__ import annotations

from pathlib import Path

_loaded = False


def load_env() -> None:
    global _loaded
    if _loaded:
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - exercised only if python-dotenv isn't installed
        _loaded = True
        return
    root = Path(__file__).resolve().parents[1]  # repo root, one level up from evalsuite/
    load_dotenv(root / ".env")
    _loaded = True
