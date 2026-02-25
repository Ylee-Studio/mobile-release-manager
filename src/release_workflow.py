"""Release workflow runtime driven by Crew decisions."""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .crew_memory import SQLiteMemory
from .crew_runtime import CrewDecision, CrewRuntimeCoordinator
from .tools.slack_tools import SlackGateway
from .workflow_state import WorkflowState

@dataclass
class RuntimeConfig:
    slack_channel_id: str
    slack_bot_token: str
    timezone: str
    heartbeat_active_minutes: int
    heartbeat_idle_minutes: int
    jira_project_keys: list[str]
    readiness_owners: dict[str, str]
    memory_db_path: str
    audit_log_path: str
    slack_events_path: str
    agent_pid_path: str
    jira_base_url: str | None = None
    jira_email: str | None = None
    jira_api_token: str | None = None


class ReleaseWorkflow:
    """Thin runtime that delegates release decisions to Crew."""

    def __init__(
        self,
        *,
        config: RuntimeConfig,
        memory: SQLiteMemory,
        slack_gateway: SlackGateway,
        crew_runtime: CrewRuntimeCoordinator,
    ):
        self.config = config
        self.memory = memory
        self.slack_gateway = slack_gateway
        self.crew_runtime = crew_runtime
        self.audit_log_path = Path(config.audit_log_path)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("release_workflow")

    def tick(self, *, now: datetime | None = None) -> WorkflowState:
        state = self.memory.load_state()
        events = self.slack_gateway.poll_events()
        trace_id = str(uuid.uuid4())

        decision = self.crew_runtime.decide(
            state=state,
            events=events,
            config=self.config,
            now=now,
        )
        next_state = decision.next_state
        self.memory.save_state(next_state, reason=decision.audit_reason)
        self._append_audit(
            trace_id=trace_id,
            previous_state=state,
            decision=decision,
            next_state=next_state,
            events=[_event_to_dict(event) for event in events],
        )
        return next_state

    def _append_audit(
        self,
        *,
        trace_id: str,
        previous_state: WorkflowState,
        decision: CrewDecision,
        next_state: WorkflowState,
        events: list[dict[str, Any]],
    ) -> None:
        incoming_slack_messages = _extract_event_messages(events)
        payload = {
            "trace_id": trace_id,
            "actor": decision.actor,
            "event_count": len(events),
            "events": events,
            "incoming_slack_messages": incoming_slack_messages,
            "previous_step": previous_state.step.value,
            "next_step": decision.next_step.value,
            "tool_calls": decision.tool_calls,
            "audit_reason": decision.audit_reason,
            "state_patch": decision.state_patch,
            "next_state": next_state.to_dict(),
            "at": datetime.now(timezone.utc).isoformat(),
        }
        with self.audit_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self.logger.info(
            "trace_id=%s decision next_step=%s reason=%s events=%s incoming_slack_messages=%s",
            trace_id,
            decision.next_step.value,
            decision.audit_reason,
            len(events),
            incoming_slack_messages,
        )


def _event_to_dict(event: Any) -> dict[str, Any]:
    return {
        "event_id": getattr(event, "event_id", ""),
        "event_type": getattr(event, "event_type", ""),
        "channel_id": getattr(event, "channel_id", ""),
        "text": getattr(event, "text", ""),
        "thread_ts": getattr(event, "thread_ts", None),
        "message_ts": getattr(event, "message_ts", None),
        "metadata": getattr(event, "metadata", None),
    }


def _extract_event_messages(events: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for event in events:
        event_id = str(event.get("event_id", ""))
        event_type = str(event.get("event_type", ""))
        text = str(event.get("text", ""))
        if not text:
            continue
        messages.append(
            {
                "event_id": event_id,
                "event_type": event_type,
                "text": text,
            }
        )
    return messages
