"""CLI handlers for the ``hermes proxy`` subcommand."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from hermes_cli.proxy.adapters import ADAPTERS, get_adapter
from hermes_cli.proxy.context_lite import ContextLiteConfig, default_store_path
from hermes_cli.proxy.server import (
    AIOHTTP_AVAILABLE,
    DEFAULT_HOST,
    DEFAULT_PORT,
    run_server,
)

logger = logging.getLogger(__name__)


def _print_aiohttp_missing() -> None:
    print(
        "hermes proxy requires aiohttp. Run `hermes setup` to install it.",
        file=sys.stderr,
    )


def cmd_proxy_start(args: Any) -> int:
    """Run the proxy server in the foreground.

    Returns process exit code (0 on clean shutdown).
    """
    if not AIOHTTP_AVAILABLE:
        _print_aiohttp_missing()
        return 1

    provider = getattr(args, "provider", None) or "nous"
    try:
        adapter = get_adapter(provider)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if not adapter.is_authenticated():
        auth_hint = getattr(adapter, "auth_hint", f"hermes auth add {adapter.name}")
        print(
            f"Not logged into {adapter.display_name}. "
            f"Run `{auth_hint}` first.",
            file=sys.stderr,
        )
        return 2

    host = getattr(args, "host", None) or DEFAULT_HOST
    port = getattr(args, "port", None) or DEFAULT_PORT
    compact_enabled = bool(getattr(args, "context_lite", False))
    context_lite = None
    if compact_enabled:
        store_path = getattr(args, "context_store", None) or default_store_path()
        context_lite = ContextLiteConfig(
            store_path=store_path,
            session_header=getattr(args, "context_session_header", None) or "X-Hermes-Session-Id",
            summary_max_chars=int(getattr(args, "context_summary_chars", 1800) or 1800),
            preserve_tools=bool(getattr(args, "context_preserve_tools", False)),
        )

    print(
        f"Starting Hermes proxy for {adapter.display_name}\n"
        f"  Listening on:  http://{host}:{port}/v1\n"
        f"  Forwarding to: (resolved per-request from your subscription)\n"
        f"  Use any bearer token in the client — the proxy attaches your real credential.\n"
        f"\n"
        f"Press Ctrl+C to stop.",
        file=sys.stderr,
    )
    if context_lite is not None:
        print(
            "  Compact context: enabled\n"
            f"    Store: {context_lite.store_path}\n"
            f"    Session header: {context_lite.session_header}\n"
            f"    Summary cap: {context_lite.summary_max_chars} chars\n"
            f"    Preserve tools: {'yes' if context_lite.preserve_tools else 'no'}",
            file=sys.stderr,
        )
    else:
        print("  Compact context: disabled", file=sys.stderr)

    try:
        asyncio.run(run_server(adapter, host=host, port=port, context_lite=context_lite))
    except KeyboardInterrupt:
        print("\nproxy: stopped", file=sys.stderr)
    except OSError as exc:
        print(f"proxy: failed to bind {host}:{port}: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_proxy_status(args: Any) -> int:
    """Print the status of each configured upstream adapter."""
    print("Hermes proxy upstream adapters\n")
    for name in sorted(ADAPTERS):
        adapter = get_adapter(name)
        if not adapter.is_authenticated():
            print(f"  [{name:8s}] {adapter.display_name} — not logged in")
            continue
        try:
            cred = adapter.get_credential()
        except Exception as exc:
            print(
                f"  [{name:8s}] {adapter.display_name} — credentials need attention "
                f"({exc})"
            )
            continue
        expires = f" (bearer expires {cred.expires_at})" if cred.expires_at else ""
        print(f"  [{name:8s}] {adapter.display_name} — ready{expires}")
    print(
        "\nStart the proxy with: hermes proxy start [--provider <name>]"
    )
    return 0


def cmd_proxy_list_providers(args: Any) -> int:
    """List available proxy upstream providers."""
    print("Available proxy upstream providers:")
    for name in sorted(ADAPTERS):
        adapter = get_adapter(name)
        print(f"  {name}  — {adapter.display_name}")
    return 0


def cmd_proxy(args: Any) -> int:
    """Dispatch ``hermes proxy <subcommand>``."""
    sub = getattr(args, "proxy_command", None)
    if sub == "start":
        return cmd_proxy_start(args)
    if sub == "status":
        return cmd_proxy_status(args)
    if sub in {"providers", "list"}:
        return cmd_proxy_list_providers(args)
    # No subcommand → print short help.
    print(
        "hermes proxy — local OpenAI-compatible proxy that attaches your\n"
        "OAuth-authenticated provider credentials to outbound requests.\n"
        "Use --context-lite on `start` to keep the transcript locally and\n"
        "forward only a recap plus the current user turn.\n"
        "\n"
        "Subcommands:\n"
        "  hermes proxy start [--provider nous|xai] [--host 127.0.0.1] [--port 8645]\n"
        "      Run the proxy in the foreground.\n"
        "  hermes proxy status\n"
        "      Show which upstream adapters are ready.\n"
        "  hermes proxy providers\n"
        "      List available upstream providers.\n",
        file=sys.stderr,
    )
    return 0


__all__ = [
    "cmd_proxy",
    "cmd_proxy_start",
    "cmd_proxy_status",
    "cmd_proxy_list_providers",
]
