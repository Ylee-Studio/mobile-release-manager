"""Slack tool contracts and local mock adapter."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    """Small local adapter used by workflow and tests."""

    def __init__(self, *, outbox_path: Path, events_path: Path):
        self.outbox_path = outbox_path
        self.events_path = events_path
        self.outbox_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

    def send_message(self, channel_id: str, text: str, *, thread_ts: str | None = None) -> dict[str, str]:
        return self._append_outbox(
            {
                "op": "slack_message",
                "channel_id": channel_id,
                "text": text,
                "thread_ts": thread_ts,
            }
        )

    def send_approve(self, channel_id: str, text: str, approve_label: str) -> dict[str, str]:
        return self._append_outbox(
            {
                "op": "slack_approve",
                "channel_id": channel_id,
                "text": text,
                "approve_label": approve_label,
            }
        )

    def update_message(self, channel_id: str, message_ts: str, text: str) -> dict[str, str]:
        return self._append_outbox(
            {
                "op": "slack_update",
                "channel_id": channel_id,
                "message_ts": message_ts,
                "text": text,
            }
        )

    def poll_events(self) -> list[SlackEvent]:
        if not self.events_path.exists():
            return []
        events: list[SlackEvent] = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            events.append(
                SlackEvent(
                    event_id=raw["event_id"],
                    event_type=raw["event_type"],
                    channel_id=raw["channel_id"],
                    text=raw.get("text", ""),
                    thread_ts=raw.get("thread_ts"),
                    message_ts=raw.get("message_ts"),
                    metadata=raw.get("metadata"),
                )
            )
        return events

    def _append_outbox(self, payload: dict[str, Any]) -> dict[str, str]:
        lines = self.outbox_path.read_text(encoding="utf-8").splitlines() if self.outbox_path.exists() else []
        message_ts = f"{len(lines) + 1:08d}.000001"
        payload["message_ts"] = message_ts
        with self.outbox_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        return {"message_ts": message_ts}


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
