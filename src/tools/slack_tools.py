"""Slack tool contracts and local mock adapter."""
from __future__ import annotations

import fcntl
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


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
        self.client = WebClient(token=self.bot_token, timeout=self.timeout_seconds)

    def send_message(self, channel_id: str, text: str, *, thread_ts: str | None = None) -> dict[str, str]:
        try:
            response = self.client.chat_postMessage(
                channel=channel_id,
                text=text,
                thread_ts=thread_ts,
            )
        except SlackApiError as exc:
            raise _runtime_error_from_slack("chat.postMessage", exc) from exc
        return _message_ts_from_response(response=response, method_name="chat.postMessage")

    def send_approve(
        self,
        channel_id: str,
        text: str,
        approve_label: str,
        *,
        reject_label: str | None = None,
    ) -> dict[str, str]:
        action_elements: list[dict[str, Any]] = [
            {
                "type": "button",
                "action_id": "release_approve",
                "text": {"type": "plain_text", "text": approve_label},
                "style": "primary",
                "value": approve_label,
            }
        ]
        if reject_label:
            action_elements.append(
                {
                    "type": "button",
                    "action_id": "release_reject",
                    "text": {"type": "plain_text", "text": reject_label},
                    "style": "danger",
                    "value": reject_label,
                }
            )
        try:
            response = self.client.chat_postMessage(
                channel=channel_id,
                text=text,
                blocks=[
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": text},
                    },
                    {
                        "type": "actions",
                        "elements": action_elements,
                    },
                ],
            )
        except SlackApiError as exc:
            raise _runtime_error_from_slack("chat.postMessage", exc) from exc
        return _message_ts_from_response(response=response, method_name="chat.postMessage")

    def update_message(self, channel_id: str, message_ts: str, text: str) -> dict[str, str]:
        try:
            response = self.client.chat_update(
                channel=channel_id,
                ts=message_ts,
                text=text,
            )
        except SlackApiError as exc:
            raise _runtime_error_from_slack("chat.update", exc) from exc
        return _message_ts_from_response(response=response, method_name="chat.update")

    def add_reaction(self, channel_id: str, message_ts: str, emoji: str = "eyes") -> bool:
        """Add reaction to a message; treat duplicate reaction as success."""
        try:
            self.client.reactions_add(
                channel=channel_id,
                timestamp=message_ts,
                name=emoji,
            )
            return True
        except SlackApiError as exc:
            error_text = _slack_error_code(exc)
            if "already_reacted" in error_text or "missing_scope" in error_text:
                self.logger.debug(
                    "reaction skipped channel=%s ts=%s emoji=%s reason=%s",
                    channel_id,
                    message_ts,
                    emoji,
                    "already_reacted" if "already_reacted" in error_text else "missing_scope",
                )
                return False
            raise _runtime_error_from_slack("reactions.add", exc) from exc

    def poll_events(self) -> list[SlackEvent]:
        if not self.events_path.exists():
            return []
        event = self._consume_one_event()
        if event is None:
            return []
        return [event]

    def pending_events_count(self) -> int:
        if not self.events_path.exists():
            return 0
        with self.events_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            return sum(1 for line in handle if line.strip())

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


def _slack_error_code(exc: SlackApiError) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)
    code = response.get("error")
    return str(code or str(exc))


def _runtime_error_from_slack(method_name: str, exc: SlackApiError) -> RuntimeError:
    response = getattr(exc, "response", None)
    if response is None:
        return RuntimeError(f"Slack API {method_name} failed: {exc}")
    status_code = getattr(response, "status_code", None)
    error_code = response.get("error")
    if status_code:
        return RuntimeError(f"Slack API {method_name} failed with HTTP {status_code}: {error_code}")
    return RuntimeError(f"Slack API {method_name} failed: {error_code}")


def _message_ts_from_response(*, response: Any, method_name: str) -> dict[str, str]:
    message_ts = response.get("ts")
    if not message_ts:
        raise RuntimeError(f"Slack API {method_name} did not return message ts.")
    return {"message_ts": str(message_ts)}


class SlackMessageInput(BaseModel):
    channel_id: str = Field(..., min_length=1, description="Target Slack channel id.")
    text: str = Field(..., min_length=1, description="Message body.")
    thread_ts: str | None = Field(default=None, description="Optional thread ts.")

    @field_validator("channel_id", "text")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must be non-empty")
        return stripped

    @field_validator("thread_ts")
    @classmethod
    def _strip_optional_thread_ts(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class SlackApproveInput(BaseModel):
    channel_id: str = Field(..., min_length=1, description="Target Slack channel id.")
    text: str = Field(..., min_length=1, description="Approval request text.")
    approve_label: str = Field(default="Подтвердить", min_length=1, description="Button label.")
    reject_label: str | None = Field(
        default=None,
        description="Optional reject button label.",
    )

    @field_validator("channel_id", "text", "approve_label")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must be non-empty")
        return stripped

    @field_validator("reject_label")
    @classmethod
    def _strip_optional_reject_label(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class SlackUpdateInput(BaseModel):
    channel_id: str = Field(..., min_length=1, description="Target Slack channel id.")
    message_ts: str = Field(..., min_length=1, description="Slack message ts to update.")
    text: str = Field(..., min_length=1, description="Updated message text.")

    @field_validator("channel_id", "message_ts", "text")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must be non-empty")
        return stripped


