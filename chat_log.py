"""Append-only JSONL logging of chat turns.

Each Q&A turn is written as one JSON object per line to a dated file under
``CHAT_LOG_PATH``. No database or server is required; the resulting ``.jsonl``
files map one line to one future DB row and can be bulk-loaded later
(``pandas.read_json(path, lines=True)`` or streamed into SQL). Gated by
``CHAT_LOG_ENABLED``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from langchain_core.documents import Document

from config import settings


def new_session_id() -> str:
    """A sortable, collision-resistant id grouping the turns of one session."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}_{uuid4().hex[:6]}"


def _source_summary(docs: list[Document]) -> list[dict]:
    return [
        {
            "title": d.metadata.get("title"),
            "article": d.metadata.get("article"),
        }
        for d in docs
    ]


def log_turn(
    session_id: str,
    turn: int,
    question: str,
    answer: str,
    sources: list[Document],
    usage: dict | None = None,
) -> None:
    """Append one Q&A turn as a JSON line. Best-effort: never raises to caller."""
    if not settings.CHAT_LOG_ENABLED:
        return

    record = {
        "session_id": session_id,
        "turn": turn,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer,
        "sources": _source_summary(sources),
        "llm_model": settings.LLM_MODEL,
        "rerank_enabled": settings.RERANK_ENABLED,
        "rerank_model": settings.RERANK_MODEL if settings.RERANK_ENABLED else None,
        # Generator token accounting (input / cached / output / total / cost_usd).
        "usage": usage or {},
    }

    try:
        log_dir = Path(settings.CHAT_LOG_PATH)
        log_dir.mkdir(parents=True, exist_ok=True)
        # One file per UTC day keeps individual files small and easy to browse.
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = log_dir / f"chat_{day}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        # Logging is non-critical; don't break the chat if the write fails.
        pass
