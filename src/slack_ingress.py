"""Minimal Slack HTTP ingress that writes workflow events to jsonl."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid
import fcntl
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config_loader import resolve_env_value


@dataclass
class SlackIngressConfig:
    signing_secret: str
    announce_channel_id: str
    events_path: Path


class SlackEventWriter:
    """Persists Slack events in workflow-compatible jsonl format."""

    def __init__(self, events_path: Path):
        self.events_path = events_path
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        event_id: str,
        event_type: str,
        channel_id: str,
        text: str = "",
        thread_ts: str | None = None,
        message_ts: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        payload = {
            "event_id": event_id,
            "event_type": event_type,
            "channel_id": channel_id,
            "text": text,
            "thread_ts": thread_ts,
            "message_ts": message_ts,
            "metadata": metadata or {},
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class SlackRequestHandler(BaseHTTPRequestHandler):
    """Accepts Slack events, interactivity and commands."""

    writer: SlackEventWriter
    cfg: SlackIngressConfig
    on_event_persisted: Callable[[], None] | None = None

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        route = urlparse(self.path).path
        body = self._read_body()
        if not self._verify_signature(body):
            self._send_json(401, {"ok": False, "error": "invalid_signature"})
            return

        if route == "/slack/events":
            self._handle_events(body)
            return
        if route == "/slack/interactivity":
            self._handle_interactivity(body)
            return
        if route == "/slack/commands":
            self._handle_commands(body)
            return
        self._send_json(404, {"ok": False, "error": "not_found"})

    def _handle_events(self, body: bytes) -> None:
        raw = json.loads(body.decode("utf-8"))
        if raw.get("type") == "url_verification":
            self._send_json(200, {"challenge": raw.get("challenge", "")})
            return
        if raw.get("type") != "event_callback":
            self._send_json(200, {"ok": True, "ignored": True})
            return

        event = raw.get("event", {})
        if event.get("type") != "message":
            self._send_json(200, {"ok": True, "ignored": True})
            return
        if _should_ignore_message_event(raw, event):
            self._send_json(200, {"ok": True, "ignored": True})
            return

        channel_id = str(event.get("channel", ""))
        if not channel_id:
            self._send_json(200, {"ok": True, "ignored": True})
            return

        self.writer.write(
            event_id=str(raw.get("event_id") or f"evt-{uuid.uuid4()}"),
            event_type="message",
            channel_id=channel_id,
            text=str(event.get("text", "")),
            thread_ts=event.get("thread_ts") or event.get("ts"),
            message_ts=event.get("ts"),
            metadata={
                "source": "events_api",
                "user_id": event.get("user"),
            },
        )
        self._trigger_event_processing()
        self._send_json(200, {"ok": True})

    def _handle_interactivity(self, body: bytes) -> None:
        form = parse_qs(body.decode("utf-8"), keep_blank_values=True)
        payload_raw = form.get("payload", ["{}"])[0]
        payload = json.loads(payload_raw)
        if payload.get("type") != "block_actions":
            self._send_json(200, {"ok": True, "ignored": True})
            return

        action = (payload.get("actions") or [{}])[0]
        action_id = str(action.get("action_id") or "").strip()
        channel_id = str((payload.get("channel") or {}).get("id", ""))
        message = payload.get("message") or {}
        container = payload.get("container") or {}
        thread_ts = container.get("thread_ts") or message.get("thread_ts") or message.get("ts")
        message_ts = container.get("message_ts") or message.get("ts")
        text = str(action.get("value") or (action.get("text") or {}).get("text") or "approve")
        trigger_message_text = str(message.get("text") or "").strip()
        if not trigger_message_text:
            blocks = message.get("blocks")
            if isinstance(blocks, list):
                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                    block_text = (block.get("text") or {}).get("text")
                    if block_text:
                        trigger_message_text = str(block_text).strip()
                        break

        if not channel_id:
            self._send_json(200, {"ok": True, "ignored": True})
            return

        event_type = "approval_rejected" if action_id == "release_reject" else "approval_confirmed"
        self.writer.write(
            event_id=f"action-{payload.get('trigger_id') or uuid.uuid4()}",
            event_type=event_type,
            channel_id=channel_id,
            text=text,
            thread_ts=thread_ts,
            message_ts=message_ts,
            metadata={
                "source": "interactivity",
                "action_id": action_id,
                "block_id": action.get("block_id"),
                "user_id": (payload.get("user") or {}).get("id"),
                "trigger_message_text": trigger_message_text,
            },
        )
        self._trigger_event_processing()
        self._send_json(200, {"ok": True})

    def _handle_commands(self, body: bytes) -> None:
        form = parse_qs(body.decode("utf-8"), keep_blank_values=True)
        command = form.get("command", [""])[0]
        channel_id = form.get("channel_id", [""])[0]
        text = form.get("text", [""])[0]
        if command != "/release-start":
            self._send_json(200, {"ok": True, "ignored": True})
            return
        if not channel_id:
            self._send_json(200, {"ok": False, "error": "missing_channel"})
            return
        if channel_id != self.cfg.announce_channel_id:
            self._send_json(
                200,
                {
                    "response_type": "ephemeral",
                    "text": "Эта команда доступна только в релизном канале.",
                },
            )
            return

        self.writer.write(
            event_id=f"cmd-{form.get('trigger_id', [''])[0] or uuid.uuid4()}",
            event_type="manual_start",
            channel_id=channel_id,
            text=text,
            metadata={
                "source": "slash_command",
                "user_id": form.get("user_id", [""])[0],
            },
        )
        self._trigger_event_processing()
        self._send_json(
            200,
            {
                "response_type": "ephemeral",
                "text": "Запрос на старт релиза принят.",
            },
        )

    def _trigger_event_processing(self) -> None:
        callback = type(self).on_event_persisted
        if not callback:
            return
        callback()

    def _verify_signature(self, body: bytes) -> bool:
        timestamp = self.headers.get("X-Slack-Request-Timestamp", "")
        signature = self.headers.get("X-Slack-Signature", "")
        if not timestamp or not signature:
            return False
        try:
            ts = int(timestamp)
        except ValueError:
            return False
        if abs(int(time.time()) - ts) > 60 * 5:
            return False

        base = f"v0:{timestamp}:{body.decode('utf-8')}"
        digest = hmac.new(
            self.cfg.signing_secret.encode("utf-8"),
            base.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        expected = f"v0={digest}"
        return hmac.compare_digest(expected, signature)

    def _read_body(self) -> bytes:
        content_length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(content_length)

    def _send_json(self, code: int, payload: dict) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        # Keep default HTTP noise out of CLI output.
        return


def build_ingress_config(config: dict) -> SlackIngressConfig:
    workflow = config.get("workflow", {})
    storage = workflow.get("storage", {})
    slack = config.get("slack", {})

    channel_id = resolve_env_value(
        str(slack.get("channel_id", "")),
        fallback_env_var="SLACK_ANNOUNCE_CHANNEL",
    )
    if not channel_id:
        raise RuntimeError("SLACK_ANNOUNCE_CHANNEL must be set for Slack ingress.")

    signing_secret = os.getenv("SLACK_SIGNING_SECRET", "")
    if not signing_secret:
        raise RuntimeError("SLACK_SIGNING_SECRET must be set for Slack ingress.")

    events_path = Path(storage.get("slack_events_path", "artifacts/mock_slack_events.jsonl"))
    return SlackIngressConfig(
        signing_secret=signing_secret,
        announce_channel_id=channel_id,
        events_path=events_path,
    )


def _extract_bot_user_id(raw_event: dict) -> str:
    """Return bot user id from Slack Events envelope when available."""
    authorizations = raw_event.get("authorizations")
    if isinstance(authorizations, list):
        for auth in authorizations:
            if not isinstance(auth, dict):
                continue
            user_id = auth.get("user_id")
            if user_id:
                return str(user_id)

    # Older Events API payloads may include authed_user/authed_users fields.
    authed_user_id = raw_event.get("authed_user_id") or raw_event.get("authed_user")
    if authed_user_id:
        return str(authed_user_id)
    authed_users = raw_event.get("authed_users")
    if isinstance(authed_users, list) and authed_users:
        first_user = authed_users[0]
        if first_user:
            return str(first_user)
    return ""


def _should_ignore_message_event(raw_event: dict, event: dict) -> bool:
    subtype = event.get("subtype")
    if subtype in {"bot_message", "message_changed", "message_deleted"}:
        return True
    if event.get("bot_id") or event.get("bot_profile"):
        return True

    bot_user_id = _extract_bot_user_id(raw_event)
    if bot_user_id and str(event.get("user", "")) == bot_user_id:
        return True
    text = str(event.get("text", ""))
    if not bot_user_id or f"<@{bot_user_id}" not in text:
        return True
    return False


def run_slack_ingress(
    *,
    host: str,
    port: int,
    cfg: SlackIngressConfig,
    on_event_persisted: Callable[[], None] | None = None,
) -> None:
    writer = SlackEventWriter(events_path=cfg.events_path)

    class _Handler(SlackRequestHandler):
        pass

    _Handler.writer = writer
    _Handler.cfg = cfg
    _Handler.on_event_persisted = on_event_persisted

    server = ThreadingHTTPServer((host, port), _Handler)
    server.serve_forever()
