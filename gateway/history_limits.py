"""Shared history-size limits for interactive gateway frontends."""

from __future__ import annotations

from typing import Any, Dict, List

from gateway.config import Platform

API_SERVER_AGENT_HISTORY_LIMIT = 24


def truncate_api_server_agent_history(
    history: List[Dict[str, Any]],
    *,
    platform: Platform,
) -> List[Dict[str, Any]]:
    """Trim interactive API-server sessions to the newest turns.

    Desktop and other API-server-backed frontends keep one long-lived logical
    session. Without a cap, a tiny follow-up like ``hello?`` can replay tens of
    thousands of tokens from the entire chat into llama.cpp. Messaging
    platforms keep their full history because their transcripts are usually the
    actual source of truth and are already managed by per-platform rules.
    """
    if platform != Platform.API_SERVER:
        return history
    if len(history) <= API_SERVER_AGENT_HISTORY_LIMIT:
        return history
    return history[-API_SERVER_AGENT_HISTORY_LIMIT:]
