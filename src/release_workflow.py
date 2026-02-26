"""Release workflow runtime driven by Crew decisions."""
from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from crewai import Memory

from .crew_runtime import CrewDecision, CrewRuntimeCoordinator
from .tool_calling import ToolCallValidationError, validate_and_normalize_tool_calls
from .tools.slack_tools import (
    SlackApproveInput,
    SlackGateway,
    SlackMessageInput,
    SlackUpdateInput,
)
from .workflow_state import ReleaseStep, WorkflowState

MANUAL_RELEASE_MESSAGE = "Подтвердите создание релиза {release_version} в JIRA."

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


WORKFLOW_STATE_KEY = "release_workflow_state"
WORKFLOW_MEMORY_SCOPE = "/workflow/state"
WORKFLOW_MEMORY_SOURCE = "release_workflow"
WORKFLOW_MEMORY_CATEGORY = "workflow_state"


class ReleaseWorkflow:
    """Thin runtime that delegates release decisions to Crew."""

    def __init__(
        self,
        *,
        config: RuntimeConfig,
        memory: Memory,
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
        self._state = _load_state_from_memory(self.memory)

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
        self._add_processing_reactions(events=events)

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
        self._execute_tool_calls(decision=decision, events=events)
        next_state = decision.next_state
        _save_state_to_memory(self.memory, next_state, reason=decision.audit_reason)
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
        readiness_gate_decision = self._handle_wait_readiness_confirmations_gate(
            state=state,
            events=events,
        )
        if readiness_gate_decision is not None:
            return readiness_gate_decision

        if state.is_paused and _has_approval_confirmed(events):
            approval_event = _extract_latest_approval_event(events)
            pending_key = _pending_key_from_event(approval_event)
            flow_id = str(state.flow_execution_id or "").strip()
            if not flow_id:
                return CrewDecision(
                    next_step=state.step,
                    next_state=state,
                    audit_reason=f"flow_id_lookup_status:missing;pending_key:{pending_key}",
                    actor="flow_runtime",
                    flow_lifecycle="paused",
                )
            feedback = str(getattr(_extract_latest_approval_event(events), "text", "") or "").strip()
            resume_decision = self.crew_runtime.resume_from_pending(
                flow_id=flow_id,
                feedback=feedback,
                state=state,
            )
            if resume_decision.flow_lifecycle == "error":
                return resume_decision
            decision = self.start_or_continue_release(
                state=state,
                events=events,
                now=now,
                trigger_reason=trigger_reason,
            )
            decision.audit_reason = (
                f"flow_id_lookup_status:state;pending_key:{pending_key};{decision.audit_reason}"
            )
            return decision
        return self.start_or_continue_release(
            state=state,
            events=events,
            now=now,
            trigger_reason=trigger_reason,
        )

    def start_or_continue_release(
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

    def _handle_wait_readiness_confirmations_gate(
        self,
        *,
        state: WorkflowState,
        events: list[Any],
    ) -> CrewDecision | None:
        release = state.active_release
        if not state.is_paused or release is None:
            return None
        if state.step != ReleaseStep.WAIT_READINESS_CONFIRMATIONS:
            return None

        readiness_owners = self.config.readiness_owners or {}
        required_points = list(readiness_owners.keys())
        if not required_points:
            return None

        next_state = WorkflowState.from_dict(state.to_dict())
        next_release = next_state.active_release
        if next_release is None:
            return None

        readiness_map = _normalize_readiness_map(
            readiness_map=next_release.readiness_map,
            required_points=required_points,
        )
        readiness_thread_ts = _readiness_thread_ts(next_release)
        for event in events:
            if not _is_readiness_thread_message(event=event, readiness_thread_ts=readiness_thread_ts):
                continue
            text = str(getattr(event, "text", "") or "").strip()
            if not _is_readiness_confirmation_text(text):
                continue
            metadata = getattr(event, "metadata", None)
            confirmed_points = _extract_confirmed_readiness_points(
                text=text,
                readiness_owners=readiness_owners,
                metadata=metadata,
            )
            for point in confirmed_points:
                readiness_map[point] = True

        next_release.readiness_map = readiness_map
        confirmed_total = sum(1 for point in required_points if readiness_map.get(point) is True)
        required_total = len(required_points)
        if confirmed_total < required_total:
            return CrewDecision(
                next_step=ReleaseStep.WAIT_READINESS_CONFIRMATIONS,
                next_state=next_state,
                audit_reason=f"readiness_confirmations_progress:{confirmed_total}/{required_total}",
                actor="flow_runtime",
                flow_lifecycle="paused",
            )

        next_release.set_step(ReleaseStep.READY_FOR_BRANCH_CUT)
        next_state.flow_paused_at = None
        next_state.pause_reason = None
        return CrewDecision(
            next_step=ReleaseStep.READY_FOR_BRANCH_CUT,
            next_state=next_state,
            audit_reason=f"readiness_confirmations_complete:{confirmed_total}/{required_total}",
            actor="flow_runtime",
            flow_lifecycle="running",
        )

    def resume_on_event(
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

    def _add_processing_reactions(self, *, events: list[Any]) -> None:
        for event in events:
            channel_id = str(getattr(event, "channel_id", "") or "").strip()
            message_ts = str(getattr(event, "message_ts", "") or "").strip()
            if not channel_id or not message_ts:
                continue
            try:
                self.slack_gateway.add_reaction(
                    channel_id=channel_id,
                    message_ts=message_ts,
                    emoji="eyes",
                )
            except Exception as exc:  # pragma: no cover - runtime safety
                self.logger.warning(
                    "failed to add processing reaction channel=%s ts=%s: %s",
                    channel_id,
                    message_ts,
                    exc,
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
            "flow_lifecycle": decision.flow_lifecycle,
            "flow_event": _flow_event_from_lifecycle(decision.flow_lifecycle),
            "flow_id_lookup_status": _audit_field(decision.audit_reason, "flow_id_lookup_status"),
            "pending_key": _audit_field(decision.audit_reason, "pending_key"),
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
            "trace_id=%s lifecycle=%s decision next_step=%s reason=%s events=%s incoming_slack_messages=%s",
            trace_id,
            decision.flow_lifecycle,
            decision.next_step.value,
            decision.audit_reason,
            len(events),
            incoming_slack_messages,
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
            default_text = MANUAL_RELEASE_MESSAGE.format(release_version=release.release_version)
            default_approve_label = "Релиз создан"
        elif decision.next_step == ReleaseStep.WAIT_MEETING_CONFIRMATION:
            message_key = "meeting_confirmation"
            default_text = (
                f"Подтвердите, что встреча фиксации релиза {release.release_version} проведена "
                "и можно переходить к сбору readiness."
            )
            default_approve_label = "Митинг проведен"
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
        if release is not None:
            message_key = "generic_message"
        else:
            message_key = ""
        text = str(args.get("text") or "").strip()
        if not text:
            self.logger.warning("skip slack_message: empty text after validation")
            return

        if release is not None and decision.next_step == ReleaseStep.WAIT_READINESS_CONFIRMATIONS:
            text = _normalize_readiness_message_text(
                text=text,
                release_version=release.release_version,
                readiness_owners=self.config.readiness_owners,
            )

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

        if release is not None and decision.next_step == ReleaseStep.WAIT_READINESS_CONFIRMATIONS:
            # Readiness announcement must be a top-level channel message.
            thread_ts_value = None
        else:
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
        if (
            release is not None
            and decision.next_step == ReleaseStep.WAIT_READINESS_CONFIRMATIONS
            and message_ts
        ):
            release.thread_ts[message_key] = thread_ts_value or message_ts
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


def _load_state_from_memory(memory: Memory) -> WorkflowState:
    try:
        matches = memory.recall(
            query=WORKFLOW_STATE_KEY,
            scope=WORKFLOW_MEMORY_SCOPE,
            categories=[WORKFLOW_MEMORY_CATEGORY],
            limit=100,
            depth="shallow",
            source=WORKFLOW_MEMORY_SOURCE,
        )
    except Exception:
        return WorkflowState()

    if not isinstance(matches, list):
        return WorkflowState()

    for match in matches:
        record = getattr(match, "record", None)
        if record is None and isinstance(match, dict):
            entry = match
        else:
            entry = {
                "content": getattr(record, "content", ""),
                "metadata": getattr(record, "metadata", {}),
            }
        raw_state = _extract_workflow_state(entry)
        if isinstance(raw_state, dict):
            return WorkflowState.from_dict(raw_state)
    return WorkflowState()


def _save_state_to_memory(memory: Memory, state: WorkflowState, *, reason: str) -> None:
    state_payload = state.to_dict()
    metadata: dict[str, Any] = {
        "state_key": WORKFLOW_STATE_KEY,
        "state": state_payload,
        "reason": reason,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    content = json.dumps(state_payload, ensure_ascii=True)
    try:
        memory.remember(
            content=content,
            scope=WORKFLOW_MEMORY_SCOPE,
            categories=[WORKFLOW_MEMORY_CATEGORY],
            metadata=metadata,
            importance=1.0,
            source=WORKFLOW_MEMORY_SOURCE,
        )
    except Exception:
        return


def _extract_workflow_state(entry: dict[str, Any]) -> dict[str, Any] | None:
    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        state = metadata.get("state")
        if isinstance(state, dict):
            return state
        if isinstance(state, str):
            try:
                parsed = json.loads(state)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, dict):
                return parsed

    state = entry.get("state")
    if isinstance(state, dict):
        return state
    content = entry.get("content")
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    if isinstance(state, str):
        try:
            parsed = json.loads(state)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


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


def _normalize_readiness_map(
    *,
    readiness_map: dict[str, Any],
    required_points: list[str],
) -> dict[str, bool]:
    normalized: dict[str, bool] = {point: False for point in required_points}
    if not isinstance(readiness_map, dict):
        return normalized
    for point in required_points:
        normalized[point] = bool(readiness_map.get(point) is True)
    return normalized


def _readiness_thread_ts(release: Any) -> str:
    thread_ts = str(release.thread_ts.get("generic_message", "") or "").strip()
    if thread_ts:
        return thread_ts
    return str(release.message_ts.get("generic_message", "") or "").strip()


def _is_readiness_thread_message(*, event: Any, readiness_thread_ts: str) -> bool:
    if str(getattr(event, "event_type", "")).strip() != "message":
        return False
    if not readiness_thread_ts:
        return False
    thread_ts = str(getattr(event, "thread_ts", "") or "").strip()
    message_ts = str(getattr(event, "message_ts", "") or "").strip()
    if thread_ts != readiness_thread_ts:
        return False
    # Ignore root readiness announcement; only process replies in thread.
    return message_ts != readiness_thread_ts


def _is_readiness_confirmation_text(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    negative_markers = ("не готов", "not ready", "blocked", "блокер", "блокируется")
    if any(marker in normalized for marker in negative_markers):
        return False
    positive_markers = (
        "готов",
        "готово",
        "готовы",
        "готова",
        "ready",
        "done",
        "ok",
        "okay",
        "влит",
        "влито",
        "merged",
        "смержен",
        "смёржен",
    )
    return any(marker in normalized for marker in positive_markers)


def _extract_confirmed_readiness_points(
    *,
    text: str,
    readiness_owners: dict[str, str],
    metadata: Any,
) -> set[str]:
    confirmed: set[str] = set()
    normalized_text = _normalize_text(text)
    user_id = _event_user_id(metadata)
    for point, owner_mention in readiness_owners.items():
        aliases = _readiness_point_aliases(point)
        if any(alias in normalized_text for alias in aliases):
            confirmed.add(point)
            continue
        owner_id = _extract_slack_user_id(owner_mention)
        if owner_id and user_id and owner_id == user_id:
            confirmed.add(point)
    return confirmed


def _readiness_point_aliases(point: str) -> tuple[str, ...]:
    normalized_point = _normalize_text(point)
    aliases: list[str] = [normalized_point]
    mapping: dict[str, tuple[str, ...]] = {
        "growth": ("growth",),
        "core": ("core",),
        "autocut": ("autocut", "auto cut"),
        "ai tools": ("ai tools", "aitools", "ai"),
        "activation": ("activation",),
        "влит релиз движка в dev": ("движка", "движок", "engine"),
        "влит релиз featureskit в dev": ("featureskit", "feature kit", "фичерскит"),
    }
    extra_aliases = mapping.get(normalized_point, ())
    for alias in extra_aliases:
        normalized_alias = _normalize_text(alias)
        if normalized_alias and normalized_alias not in aliases:
            aliases.append(normalized_alias)
    return tuple(aliases)


def _extract_slack_user_id(owner_mention: str) -> str:
    text = str(owner_mention or "").strip()
    match = re.fullmatch(r"<@([A-Z0-9]+)>", text, flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).upper()


def _event_user_id(metadata: Any) -> str:
    if not isinstance(metadata, dict):
        return ""
    user_id = str(metadata.get("user_id", "") or "").strip()
    return user_id.upper()


def _normalize_text(value: str) -> str:
    normalized = str(value or "").lower()
    normalized = normalized.replace("ё", "е")
    normalized = re.sub(r"[^a-zа-я0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _readiness_jql_query(release_version: str) -> str:
    return f"fixVersion = {release_version} ORDER BY project ASC, type DESC, key ASC"


def _readiness_issues_url(release_version: str) -> str:
    encoded = quote(_readiness_jql_query(release_version), safe="")
    return f"https://instories.atlassian.net/issues/?jql={encoded}"


def _format_slack_owner_mention(owner: str) -> str:
    normalized = str(owner or "").strip()
    if not normalized:
        return normalized
    if normalized.startswith("<@") and normalized.endswith(">"):
        return normalized
    if normalized.startswith("@"):
        return normalized
    return f"@{normalized}"


def _normalize_readiness_message_text(
    *,
    text: str,
    release_version: str,
    readiness_owners: dict[str, str],
) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return f"Релиз <{_readiness_issues_url(release_version)}|{release_version}>"

    lines = normalized.splitlines()
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        lines[idx] = f"Релиз <{_readiness_issues_url(release_version)}|{release_version}>"
        break

    status_idx = next(
        (idx for idx, line in enumerate(lines) if line.strip() == "Статус готовности к срезу:"),
        None,
    )
    prompt_idx = next(
        (
            idx
            for idx, line in enumerate(lines)
            if line.strip().startswith("Напишите в треде по готовности своей части.")
        ),
        None,
    )
    if status_idx is not None and prompt_idx is not None and prompt_idx > status_idx:
        readiness_lines = [
            f":hourglass_flowing_sand: {team} {_format_slack_owner_mention(owner)}"
            for team, owner in readiness_owners.items()
        ]
        lines = [
            *lines[: status_idx + 1],
            *readiness_lines,
            "",
            *lines[prompt_idx:],
        ]

    normalized_message = "\n".join(lines)
    normalized_message = re.sub(
        r"\*{0,2}Важное напоминание\*{0,2}",
        "**Важное напоминание**",
        normalized_message,
        count=1,
    )
    return normalized_message


def _has_approval_confirmed(events: list[Any]) -> bool:
    for event in events:
        if str(getattr(event, "event_type", "")).strip() == "approval_confirmed":
            return True
    return False


def _flow_event_from_lifecycle(lifecycle: str) -> str:
    normalized = str(lifecycle or "").strip().lower()
    if normalized == "paused":
        return "flow_paused"
    if normalized == "completed":
        return "flow_completed"
    if normalized == "running":
        return "flow_resumed"
    return "flow_transition"


def _pending_key_from_event(event: Any | None) -> str:
    if event is None:
        return ""
    message_ts = str(getattr(event, "message_ts", "") or "").strip()
    thread_ts = str(getattr(event, "thread_ts", "") or "").strip()
    return message_ts or thread_ts


def _audit_field(audit_reason: str, key: str) -> str | None:
    target = f"{key}:"
    for part in str(audit_reason or "").split(";"):
        normalized = part.strip()
        if not normalized.startswith(target):
            continue
        return normalized.split(":", 1)[-1].strip() or None
    return None


