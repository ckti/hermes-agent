"""Local transcript store + request compaction for the proxy "lite" mode.

The normal ``hermes proxy`` server is a straight-through OpenAI-compatible
forwarder.  This module adds an opt-in mode that keeps the conversation state
on disk and forwards only:

* the client's own instruction messages,
* a short local recap of prior turns, and
* the current user message.

That keeps the upstream prompt small while preserving the full transcript in a
local SQLite store for later replay, inspection, or recovery.
"""

from __future__ import annotations

import copy
import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from hermes_cli.config import get_hermes_home
from hermes_cli.session_recap import build_recap

DEFAULT_SESSION_HEADER = "X-Hermes-Session-Id"
DEFAULT_SUMMARY_MAX_CHARS = 1800
DEFAULT_SYSTEM_DIRECTIVE = (
    "You are a compact local assistant. Use the recap below as background, "
    "answer the current user directly, and do not ask them to restate "
    "context that is already captured there."
)


def default_store_path() -> Path:
    """Default SQLite path for compact proxy state."""
    return get_hermes_home() / "proxy" / "context-lite.sqlite3"


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if isinstance(block, str):
                parts.append(block)
                continue
            if isinstance(block, Mapping):
                text = block.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        return "\n".join(parts)
    return str(value)


def _normalize_jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _normalize_jsonish(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, list):
        return [_normalize_jsonish(v) for v in value]
    return value


def _message_signature(message: Mapping[str, Any]) -> str:
    """Stable signature for prefix-diffing replayed message history."""
    keys = (
        "role",
        "content",
        "name",
        "tool_call_id",
        "tool_calls",
        "function_call",
        "reasoning",
        "reasoning_content",
    )
    data: dict[str, Any] = {}
    for key in keys:
        if key in message:
            data[key] = _normalize_jsonish(message.get(key))
    try:
        return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return repr(data)


def _last_user_index(messages: Sequence[Mapping[str, Any]]) -> int:
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "user":
            return idx
    return -1


def _instruction_block(
    instruction_messages: Sequence[Mapping[str, Any]],
    summary: str,
) -> str:
    parts: list[str] = []
    for msg in instruction_messages:
        text = _flatten_text(msg.get("content")).strip()
        if text:
            parts.append(text)
    if summary:
        parts.append("Local session recap (stored locally):\n" + summary.strip())
    parts.append(DEFAULT_SYSTEM_DIRECTIVE)
    return "\n\n".join(part for part in parts if part and part.strip())


def _compact_request_body(
    request_body: Mapping[str, Any],
    *,
    messages: Sequence[Mapping[str, Any]],
    summary: str,
    user_index: int,
    preserve_tools: bool,
) -> dict[str, Any]:
    compact = dict(request_body)
    instruction_messages = [
        msg
        for msg in messages[:user_index]
        if msg.get("role") in {"system", "developer"}
    ]
    current_user = copy.deepcopy(messages[user_index])
    instruction_text = _instruction_block(instruction_messages, summary)
    instruction_role = (
        "developer"
        if any(msg.get("role") == "developer" for msg in instruction_messages)
        else "system"
    )

    compact_messages: list[dict[str, Any]] = []
    if instruction_text:
        compact_messages.append({"role": instruction_role, "content": instruction_text})
    compact_messages.append(current_user)
    compact["messages"] = compact_messages

    if not preserve_tools:
        compact.pop("tools", None)
        compact.pop("tool_choice", None)

    return compact


@dataclass(frozen=True)
class ContextLiteConfig:
    """Runtime settings for compact proxy mode."""

    store_path: Path
    session_header: str = DEFAULT_SESSION_HEADER
    summary_max_chars: int = DEFAULT_SUMMARY_MAX_CHARS
    preserve_tools: bool = False


class ContextLiteStore:
    """SQLite-backed session store used by the compact proxy mode."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    messages_json TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT messages_json FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return []
        try:
            data = json.loads(row["messages_json"])
            if isinstance(data, list):
                return [m for m in data if isinstance(m, dict)]
        except Exception:
            pass
        return []

    def load_summary(self, session_id: str) -> str:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT summary FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return ""
        summary = row["summary"] if isinstance(row["summary"], str) else ""
        return summary or ""

    def _save_session(
        self,
        session_id: str,
        messages: Sequence[Mapping[str, Any]],
        summary: str,
    ) -> None:
        payload = json.dumps(list(messages), ensure_ascii=False, separators=(",", ":"))
        now = _utc_now()
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT created_at FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            created_at = existing["created_at"] if existing is not None else now
            conn.execute(
                """
                INSERT INTO sessions (session_id, messages_json, summary, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    messages_json = excluded.messages_json,
                    summary = excluded.summary,
                    updated_at = excluded.updated_at
                """,
                (session_id, payload, summary, created_at, now),
            )

    def append_and_compact(
        self,
        session_id: str,
        incoming_messages: Sequence[Mapping[str, Any]],
        *,
        summary_max_chars: int = DEFAULT_SUMMARY_MAX_CHARS,
    ) -> tuple[list[dict[str, Any]], str]:
        """Merge a replayed message list into the local transcript.

        The incoming request is assumed to replay the full conversation history
        the client currently knows about.  We diff it against the stored copy so
        a client that sends the same transcript on every request does not cause
        the SQLite history to grow without bound.
        """
        current = self.load_messages(session_id)
        incoming = [copy.deepcopy(msg) for msg in incoming_messages]

        prefix = 0
        for existing_msg, new_msg in zip(current, incoming):
            if _message_signature(existing_msg) != _message_signature(new_msg):
                break
            prefix += 1

        if prefix < len(current):
            current = current[:prefix]
        if prefix < len(incoming):
            current.extend(incoming[prefix:])

        summary = self._build_summary(
            session_id,
            current,
            summary_max_chars=summary_max_chars,
        )
        self._save_session(session_id, current, summary)
        return current, summary

    def _build_summary(
        self,
        session_id: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        summary_max_chars: int,
    ) -> str:
        if not messages:
            return ""

        user_index = _last_user_index(messages)
        if user_index <= 0:
            return ""

        recap = build_recap(
            messages[:user_index],
            session_id=session_id,
            platform="proxy",
        ).strip()
        if not recap:
            return ""
        if summary_max_chars > 0 and len(recap) > summary_max_chars:
            recap = recap[: max(0, summary_max_chars - 1)].rstrip() + "…"
        return recap

    def build_compact_request(
        self,
        request_body: Mapping[str, Any],
        *,
        session_id: str,
        summary_max_chars: int = DEFAULT_SUMMARY_MAX_CHARS,
        preserve_tools: bool = False,
    ) -> Optional[dict[str, Any]]:
        """Return a compacted request body, or ``None`` when compaction is unsafe."""
        messages = request_body.get("messages")
        if not isinstance(messages, list) or not messages:
            return None
        if any(not isinstance(msg, Mapping) for msg in messages):
            return None

        current = [msg for msg in messages if isinstance(msg, Mapping)]
        user_index = _last_user_index(current)
        if user_index < 0:
            return None

        canonical, summary = self.append_and_compact(
            session_id,
            current,
            summary_max_chars=summary_max_chars,
        )

        compact = _compact_request_body(
            request_body,
            messages=canonical,
            summary=summary,
            user_index=user_index,
            preserve_tools=preserve_tools,
        )
        return compact


def resolve_session_id(
    request_body: Mapping[str, Any],
    headers: Mapping[str, str],
    *,
    session_header: str = DEFAULT_SESSION_HEADER,
) -> str:
    """Best-effort session key extraction for compact mode."""
    header_key = (session_header or DEFAULT_SESSION_HEADER).strip()
    if header_key:
        for key, value in headers.items():
            if key.lower() == header_key.lower() and isinstance(value, str) and value.strip():
                return value.strip()

    direct_keys = (
        "session_id",
        "conversation_id",
        "conversationId",
        "chat_id",
        "chatId",
    )
    for key in direct_keys:
        value = request_body.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    metadata = request_body.get("metadata")
    if isinstance(metadata, Mapping):
        for key in direct_keys:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    user = request_body.get("user")
    if isinstance(user, str) and user.strip():
        return user.strip()

    return "default"


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "ContextLiteConfig",
    "ContextLiteStore",
    "DEFAULT_SESSION_HEADER",
    "DEFAULT_SUMMARY_MAX_CHARS",
    "DEFAULT_SYSTEM_DIRECTIVE",
    "default_store_path",
    "resolve_session_id",
]
