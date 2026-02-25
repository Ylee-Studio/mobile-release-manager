"""Slack tool contracts and local mock adapter."""
from __future__ import annotations

import fcntl
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


@dataclass
class SlackEvent:
    event_id: str
    event_type: str
    channel_id: str
    text: str = ""
    thread_ts: str | None = None
    message_ts: str | None = None
    metadata: dict[str, Any] | None = None


class SlackGateway:
    """Slack gateway with real Web API calls and local events queue."""

    def __init__(self, *, bot_token: str, events_path: Path, timeout_seconds: float = 15.0):
        self.bot_token = bot_token
        self.events_path = events_path
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("slack_gateway")
        self.timeout_seconds = timeout_seconds
        if not self.bot_token:
            raise RuntimeError("SLACK_BOT_TOKEN must be configured for SlackGateway.")

    def send_message(self, channel_id: str, text: str, *, thread_ts: str | None = None) -> dict[str, str]:
        payload: dict[str, Any] = {
            "channel": channel_id,
            "text": text,
        }
        if thread_ts:
            payload["thread_ts"] = thread_ts
        return self._request_ts(method_name="chat.postMessage", payload=payload)

    def send_approve(self, channel_id: str, text: str, approve_label: str) -> dict[str, str]:
        payload = {
            "channel": channel_id,
            "text": text,
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": text},
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "action_id": "release_approve",
                            "text": {"type": "plain_text", "text": approve_label},
                            "style": "primary",
                            "value": approve_label,
                        }
                    ],
                },
            ],
        }
        return self._request_ts(method_name="chat.postMessage", payload=payload)

    def update_message(self, channel_id: str, message_ts: str, text: str) -> dict[str, str]:
        return self._request_ts(
            method_name="chat.update",
            payload={
                "channel": channel_id,
                "ts": message_ts,
                "text": text,
            },
        )

    def poll_events(self) -> list[SlackEvent]:
        if not self.events_path.exists():
            return []
        event = self._consume_one_event()
        if event is None:
            return []
        return [event]

    def _consume_one_event(self) -> SlackEvent | None:
        consumed_line = ""
        with self.events_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            lines = handle.readlines()
            if not lines:
                return None

            consume_idx = None
            for idx, line in enumerate(lines):
                if line.strip():
                    consume_idx = idx
                    consumed_line = line.strip()
                    break

            if consume_idx is None:
                handle.seek(0)
                handle.truncate(0)
                handle.flush()
                os.fsync(handle.fileno())
                return None

            remaining = lines[consume_idx + 1 :]
            handle.seek(0)
            handle.truncate(0)
            handle.writelines(remaining)
            handle.flush()
            os.fsync(handle.fileno())

        try:
            raw = json.loads(consumed_line)
            return SlackEvent(
                event_id=raw["event_id"],
                event_type=raw["event_type"],
                channel_id=raw["channel_id"],
                text=raw.get("text", ""),
                thread_ts=raw.get("thread_ts"),
                message_ts=raw.get("message_ts"),
                metadata=raw.get("metadata"),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            self.logger.warning("dropped malformed slack event line: %s", consumed_line[:200])
            return None

    def _request_ts(self, *, method_name: str, payload: dict[str, Any]) -> dict[str, str]:
        response = self._request_json(method_name=method_name, payload=payload)
        message_ts = response.get("ts")
        if not message_ts:
            raise RuntimeError(f"Slack API {method_name} did not return message ts.")
        return {"message_ts": str(message_ts)}

    def _request_json(self, *, method_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        endpoint = f"https://slack.com/api/{method_name}"
        raw_payload = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        request = Request(
            url=endpoint,
            data=raw_payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json; charset=utf-8",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Slack API {method_name} failed with HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Slack API {method_name} failed: {exc.reason}") from exc

        try:
            parsed = json.loads(raw_body) if raw_body.strip() else {}
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Slack API {method_name} returned invalid JSON: {raw_body}") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError(f"Slack API {method_name} returned unexpected payload type.")
        if not parsed.get("ok", False):
            error_text = parsed.get("error", "unknown_error")
            raise RuntimeError(f"Slack API {method_name} failed: {error_text}")
        return parsed


class SlackMessageInput(BaseModel):
    channel_id: str = Field(..., description="Target Slack channel id.")
    text: str = Field(..., description="Message body.")
    thread_ts: str | None = Field(default=None, description="Optional thread ts.")


class SlackMessageTool(BaseTool):
    name: str = "slack_message"
    description: str = "Send a message to Slack channel or thread."
    args_schema: type[BaseModel] = SlackMessageInput

    gateway: SlackGateway

    def _run(self, channel_id: str, text: str, thread_ts: str | None = None) -> dict[str, str]:
        return self.gateway.send_message(channel_id=channel_id, text=text, thread_ts=thread_ts)


class SlackApproveInput(BaseModel):
    channel_id: str
    text: str
    approve_label: str = "Подтвердить"


class SlackApproveTool(BaseTool):
    name: str = "slack_approve"
    description: str = "Send an approval message to Slack."
    args_schema: type[BaseModel] = SlackApproveInput

    gateway: SlackGateway

    def _run(self, channel_id: str, text: str, approve_label: str = "Подтвердить") -> dict[str, str]:
        return self.gateway.send_approve(channel_id=channel_id, text=text, approve_label=approve_label)


class SlackUpdateInput(BaseModel):
    channel_id: str
    message_ts: str
    text: str


class SlackUpdateTool(BaseTool):
    name: str = "slack_update"
    description: str = "Update an existing Slack message."
    args_schema: type[BaseModel] = SlackUpdateInput

    gateway: SlackGateway

    def _run(self, channel_id: str, message_ts: str, text: str) -> dict[str, str]:
        return self.gateway.update_message(channel_id=channel_id, message_ts=message_ts, text=text)
