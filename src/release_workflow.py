"""Release workflow runtime driven by Crew decisions."""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .crew_memory import CrewAIMemory
from .crew_runtime import CrewDecision, CrewRuntimeCoordinator
from .tool_calling import ToolCallValidationError, validate_and_normalize_tool_calls
from .tools.slack_tools import (
    SlackApproveInput,
    SlackGateway,
    SlackMessageInput,
    SlackUpdateInput,
)
from .workflow_state import ReleaseStep, WorkflowState

MANUAL_RELEASE_MESSAGE = (
    "Создайте релиз в https://instories.atlassian.net/jira/plans/1/scenarios/1/releases "
    "и проведите встречу фиксации релиза"
)
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


class WorkflowMemory(Protocol):
    def load_state(self) -> WorkflowState: ...
    def save_state(self, state: WorkflowState, *, reason: str) -> None: ...


class ReleaseWorkflow:
    """Thin runtime that delegates release decisions to Crew."""

    def __init__(
        self,
        *,
        config: RuntimeConfig,
        memory: WorkflowMemory | CrewAIMemory,
        slack_gateway: SlackGateway,
        crew_runtime: CrewRuntimeCoordinator | Any,
    ):
        self.config = config
        self.memory = memory
        self.slack_gateway = slack_gateway
        self.crew_runtime = crew_runtime
        self.audit_log_path = Path(config.audit_log_path)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("release_workflow")
        self._state = self.memory.load_state()

    def tick(
        self,
        *,
        now: datetime | None = None,
        trigger_reason: str = "heartbeat_timer",
    ) -> WorkflowState:
        state = WorkflowState.from_dict(self._state.to_dict())
        previous_state = WorkflowState.from_dict(state.to_dict())
        events = self.slack_gateway.poll_events()
        queue_remaining = self.slack_gateway.pending_events_count()
        trace_id = str(uuid.uuid4())

        decision = self._run_runtime_decision(
            state=state,
            events=events,
            now=now,
            trigger_reason=trigger_reason,
        )
        try:
            decision.tool_calls = validate_and_normalize_tool_calls(
                decision.tool_calls,
                schema_by_tool={
                    "slack_message": SlackMessageInput,
                    "slack_approve": SlackApproveInput,
                    "slack_update": SlackUpdateInput,
                },
            )
        except ToolCallValidationError as exc:
            self.logger.warning("tool_call validation failed: %s", exc)
            decision = CrewDecision(
                next_step=state.step,
                next_state=state,
                audit_reason=f"invalid_tool_call:{exc}",
                actor=decision.actor,
                tool_calls=[],
            )
        self._prune_completed_actions_on_user_transition(
            previous_state=previous_state,
            decision=decision,
            events=events,
        )
        self._execute_tool_calls(decision=decision, events=events)
        next_state = decision.next_state
        self.memory.save_state(next_state, reason=decision.audit_reason)
        self._state = WorkflowState.from_dict(next_state.to_dict())
        self._append_audit(
            trace_id=trace_id,
            previous_state=previous_state,
            decision=decision,
            next_state=next_state,
            events=[_event_to_dict(event) for event in events],
        )
        self.logger.info(
            "tick trace_id=%s reason=%s event_count=%s queue_remaining=%s",
            trace_id,
            trigger_reason,
            len(events),
            queue_remaining,
        )
        return next_state

    def _run_runtime_decision(
        self,
        *,
        state: WorkflowState,
        events: list[Any],
        now: datetime | None,
        trigger_reason: str,
    ) -> CrewDecision:
        return self.crew_runtime.kickoff(
            state=state,
            events=events,
            config=self.config,
            now=now,
            trigger_reason=trigger_reason,
        )

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

    def _prune_completed_actions_on_user_transition(
        self,
        *,
        previous_state: WorkflowState,
        decision: CrewDecision,
        events: list[Any],
    ) -> None:
        if not _is_user_initiated_transition(events):
            return
        if decision.next_step == previous_state.step:
            return

        target_order = _step_order(decision.next_step)
        if target_order is None:
            return

        removed_keys: list[str] = []
        for action_key in list(decision.next_state.completed_actions):
            action_step = _action_step(action_key)
            if action_step is None:
                continue
            action_order = _step_order(action_step)
            if action_order is None:
                continue
            # For manual user transitions, rebuild idempotency from the target step.
            if action_order < target_order:
                continue
            decision.next_state.completed_actions.pop(action_key, None)
            removed_keys.append(action_key)

        if not removed_keys:
            return
        decision.next_state.checkpoints.append(
            {
                "phase": "before",
                "tool": "state_cleanup",
                "release_version": (
                    decision.next_state.active_release.release_version
                    if decision.next_state.active_release
                    else ""
                ),
                "step": decision.next_step.value,
                "removed_completed_actions": sorted(removed_keys),
                "reason": "user_initiated_state_transition",
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.logger.info(
            "pruned completed_actions on user transition to %s: %s",
            decision.next_step.value,
            sorted(removed_keys),
        )

    def _execute_tool_calls(self, *, decision: CrewDecision, events: list[Any]) -> None:
        for call in decision.tool_calls:
            if not isinstance(call, dict):
                continue
            tool_name = str(call.get("tool", "")).strip()
            if tool_name == "slack_approve":
                self._execute_slack_approve(call=call, decision=decision, events=events)
            if tool_name == "slack_message":
                self._execute_slack_message(call=call, decision=decision, events=events)
            if tool_name == "slack_update":
                self._execute_slack_update(call=call, decision=decision, events=events)

    def _execute_slack_approve(
        self,
        *,
        call: dict[str, Any],
        decision: CrewDecision,
        events: list[Any],
    ) -> None:
        release = decision.next_state.active_release
        if release is None:
            return
        if decision.next_step == ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION:
            message_key = "manual_release_confirmation"
            default_text = (
                f"Подтвердите, что релиз {release.release_version} создан вручную в Jira plan, "
                "и можно переходить к следующему шагу."
            )
            default_approve_label = "Релиз создан"
        else:
            message_key = "start_approval"
            default_text = (
                f"Подтвердите старт релизного трейна {release.release_version}. "
                "Если нужен другой номер для релиза, напишите в треде."
            )
            default_approve_label = "Подтвердить"

        channel_id = release.slack_channel_id or _extract_event_channel_id(events) or self.config.slack_channel_id
        if not channel_id:
            self.logger.warning("skip slack_approve: empty channel_id for release=%s", release.release_version)
            return

        args_raw = call.get("args", {})
        args = args_raw if isinstance(args_raw, dict) else {}
        text = str(args.get("text") or "").strip() or default_text
        approve_label = str(args.get("approve_label") or default_approve_label)

        decision.next_state.checkpoints.append(
            {
                "phase": "before",
                "tool": "slack_approve",
                "release_version": release.release_version,
                "step": decision.next_step.value,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )
        try:
            result = self.slack_gateway.send_approve(
                channel_id=channel_id,
                text=text,
                approve_label=approve_label,
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            self.logger.exception("slack_approve failed for release=%s: %s", release.release_version, exc)
            return

        message_ts = str(result.get("message_ts", "")).strip()
        if not message_ts:
            self.logger.warning("slack_approve returned empty message_ts for release=%s", release.release_version)
            return

        release.slack_channel_id = channel_id
        release.message_ts[message_key] = message_ts
        release.thread_ts[message_key] = message_ts
        decision.next_state.checkpoints.append(
            {
                "phase": "after",
                "tool": "slack_approve",
                "release_version": release.release_version,
                "step": decision.next_step.value,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _execute_slack_message(
        self,
        *,
        call: dict[str, Any],
        decision: CrewDecision,
        events: list[Any],
    ) -> None:
        release = decision.next_state.active_release
        args_raw = call.get("args", {})
        args = args_raw if isinstance(args_raw, dict) else {}
        if decision.next_step == ReleaseStep.JIRA_RELEASE_CREATED:
            if release is None:
                return
            action_key = f"manual-release-instruction:{release.release_version}"
            message_key = "manual_release_instruction"
            text = MANUAL_RELEASE_MESSAGE
        else:
            if release is not None:
                message_key = "generic_message"
            else:
                message_key = ""
            text = str(args.get("text") or "").strip()
            if not text:
                self.logger.warning("skip slack_message: empty text after validation")
                return

        channel_id = (
            str(args.get("channel_id") or "").strip()
            or (release.slack_channel_id if release else "")
            or _extract_event_channel_id(events)
            or self.config.slack_channel_id
        )
        if not channel_id:
            release_version = release.release_version if release else "n/a"
            self.logger.warning("skip slack_message: empty channel_id for release=%s", release_version)
            return

        thread_ts = str(args.get("thread_ts") or "").strip()
        if not thread_ts and release is not None:
            thread_ts = release.thread_ts.get("manual_release_confirmation")
        if not thread_ts and release is not None:
            thread_ts = release.thread_ts.get("start_approval")
        if not thread_ts:
            thread_ts = _extract_latest_message_thread_ts(events)
        thread_ts_value = thread_ts or None

        decision.next_state.checkpoints.append(
            {
                "phase": "before",
                "tool": "slack_message",
                "release_version": release.release_version if release else "",
                "step": decision.next_step.value,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )
        try:
            result = self.slack_gateway.send_message(
                channel_id=channel_id,
                text=text,
                thread_ts=thread_ts_value,
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            self.logger.exception("slack_message failed for release=%s: %s", release.release_version, exc)
            return

        message_ts = str(result.get("message_ts", "")).strip()
        if message_ts and release is not None and message_key:
            release.message_ts[message_key] = message_ts
        if release is not None:
            release.slack_channel_id = channel_id
        decision.next_state.checkpoints.append(
            {
                "phase": "after",
                "tool": "slack_message",
                "release_version": release.release_version if release else "",
                "step": decision.next_step.value,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _execute_slack_update(
        self,
        *,
        call: dict[str, Any],
        decision: CrewDecision,
        events: list[Any],
    ) -> None:
        release = decision.next_state.active_release
        if release is None:
            return

        args_raw = call.get("args", {})
        args = args_raw if isinstance(args_raw, dict) else {}
        approval_event = _extract_latest_approval_event(events)
        event_message_ts = str(getattr(approval_event, "message_ts", "") or "").strip() if approval_event else ""
        message_ts = str(args.get("message_ts") or "").strip() or event_message_ts
        if not message_ts:
            self.logger.warning("skip slack_update: empty message_ts for release=%s", release.release_version)
            return

        channel_id = (
            str(args.get("channel_id") or "").strip()
            or release.slack_channel_id
            or _extract_event_channel_id(events)
            or self.config.slack_channel_id
        )
        if not channel_id:
            self.logger.warning("skip slack_update: empty channel_id for release=%s", release.release_version)
            return

        text = str(args.get("text") or "").strip()
        if not text and approval_event is not None:
            metadata = getattr(approval_event, "metadata", None)
            if isinstance(metadata, dict):
                text = str(metadata.get("trigger_message_text") or "").strip()
        if text and not text.endswith(":white_check_mark:"):
            text = f"{text} :white_check_mark:"
        if not text:
            self.logger.warning("skip slack_update: empty text for release=%s", release.release_version)
            return

        decision.next_state.checkpoints.append(
            {
                "phase": "before",
                "tool": "slack_update",
                "release_version": release.release_version,
                "step": decision.next_step.value,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )
        try:
            self.slack_gateway.update_message(
                channel_id=channel_id,
                message_ts=message_ts,
                text=text,
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            self.logger.exception("slack_update failed for release=%s: %s", release.release_version, exc)
            return

        release.slack_channel_id = channel_id
        decision.next_state.checkpoints.append(
            {
                "phase": "after",
                "tool": "slack_update",
                "release_version": release.release_version,
                "step": decision.next_step.value,
                "at": datetime.now(timezone.utc).isoformat(),
            }
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


def _extract_event_channel_id(events: list[Any]) -> str:
    for event in events:
        channel_id = str(getattr(event, "channel_id", "")).strip()
        if channel_id:
            return channel_id
    return ""


def _extract_latest_approval_event(events: list[Any]) -> Any | None:
    for event in reversed(events):
        if str(getattr(event, "event_type", "")).strip() == "approval_confirmed":
            return event
    return None


def _extract_latest_message_thread_ts(events: list[Any]) -> str:
    for event in reversed(events):
        if str(getattr(event, "event_type", "")).strip() != "message":
            continue
        thread_ts = str(getattr(event, "thread_ts", "") or "").strip()
        if thread_ts:
            return thread_ts
        message_ts = str(getattr(event, "message_ts", "") or "").strip()
        if message_ts:
            return message_ts
    return ""


_STEP_SEQUENCE: tuple[ReleaseStep, ...] = (
    ReleaseStep.IDLE,
    ReleaseStep.WAIT_START_APPROVAL,
    ReleaseStep.RELEASE_MANAGER_CREATED,
    ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION,
    ReleaseStep.JIRA_RELEASE_CREATED,
    ReleaseStep.WAIT_MEETING_CONFIRMATION,
    ReleaseStep.WAIT_READINESS_CONFIRMATIONS,
    ReleaseStep.READY_FOR_BRANCH_CUT,
)

_USER_EVENT_SOURCES: set[str] = {"slash_command", "events_api", "interactivity"}

_ACTION_PREFIX_TO_STEP: tuple[tuple[str, ReleaseStep], ...] = (
    ("start-approval:", ReleaseStep.WAIT_START_APPROVAL),
    ("start_approval_request", ReleaseStep.WAIT_START_APPROVAL),
    ("start_approval_confirmation", ReleaseStep.RELEASE_MANAGER_CREATED),
    ("release_manager_created", ReleaseStep.RELEASE_MANAGER_CREATED),
    ("create_crossspace_release", ReleaseStep.RELEASE_MANAGER_CREATED),
    ("manual-release-confirmation-request:", ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION),
    ("manual_release_confirmation", ReleaseStep.JIRA_RELEASE_CREATED),
    ("manual-release-instruction:", ReleaseStep.JIRA_RELEASE_CREATED),
    ("approval-confirmed-update:", ReleaseStep.JIRA_RELEASE_CREATED),
    ("manual_set_idle", ReleaseStep.IDLE),
    ("ack_manual_set_idle_message", ReleaseStep.IDLE),
)


def _step_order(step: ReleaseStep) -> int | None:
    try:
        return _STEP_SEQUENCE.index(step)
    except ValueError:
        return None


def _is_user_initiated_transition(events: list[Any]) -> bool:
    for event in events:
        metadata = getattr(event, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        source = str(metadata.get("source", "")).strip()
        if source in _USER_EVENT_SOURCES:
            return True
    return False


def _action_step(action_key: str) -> ReleaseStep | None:
    normalized = action_key.strip()
    if not normalized:
        return None
    for prefix, step in _ACTION_PREFIX_TO_STEP:
        if normalized.startswith(prefix):
            return step

    if normalized.startswith("slack-message:"):
        maybe_step = normalized.rsplit(":", 1)[-1].strip()
        try:
            return ReleaseStep(maybe_step)
        except ValueError:
            return None
    return None


