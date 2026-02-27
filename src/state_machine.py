"""Deterministic state machine gates for release workflow."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .workflow_state import ReleaseStep, WorkflowState


@dataclass
class TransitionOutcome:
    next_step: ReleaseStep
    next_state: WorkflowState
    audit_reason: str
    flow_lifecycle: str = "running"
    actor: str = "state_machine"


class StateMachineEngine:
    """Deterministic transitions evaluated before LLM runtime."""

    def apply_deterministic_gates(
        self,
        *,
        state: WorkflowState,
        events: list[Any],
        readiness_owners: dict[str, str],
    ) -> TransitionOutcome | None:
        return self._handle_wait_readiness_confirmations_gate(
            state=state,
            events=events,
            readiness_owners=readiness_owners,
        )

    def _handle_wait_readiness_confirmations_gate(
        self,
        *,
        state: WorkflowState,
        events: list[Any],
        readiness_owners: dict[str, str],
    ) -> TransitionOutcome | None:
        release = state.active_release
        if not state.is_paused or release is None:
            return None
        if state.step != ReleaseStep.WAIT_READINESS_CONFIRMATIONS:
            return None
        required_points = list((readiness_owners or {}).keys())
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
            return TransitionOutcome(
                next_step=ReleaseStep.WAIT_READINESS_CONFIRMATIONS,
                next_state=next_state,
                audit_reason=f"readiness_confirmations_progress:{confirmed_total}/{required_total}",
                flow_lifecycle="paused",
            )
        next_release.set_step(ReleaseStep.READY_FOR_BRANCH_CUT)
        next_state.flow_paused_at = None
        next_state.pause_reason = None
        return TransitionOutcome(
            next_step=ReleaseStep.READY_FOR_BRANCH_CUT,
            next_state=next_state,
            audit_reason=f"readiness_confirmations_complete:{confirmed_total}/{required_total}",
            flow_lifecycle="running",
        )


def has_approval_confirmed(events: list[Any]) -> bool:
    for event in events:
        if str(getattr(event, "event_type", "")).strip() == "approval_confirmed":
            return True
    return False


def extract_latest_approval_event(events: list[Any]) -> Any | None:
    for event in reversed(events):
        if str(getattr(event, "event_type", "")).strip() == "approval_confirmed":
            return event
    return None


def pending_key_from_event(event: Any | None) -> str:
    if event is None:
        return ""
    message_ts = str(getattr(event, "message_ts", "") or "").strip()
    thread_ts = str(getattr(event, "thread_ts", "") or "").strip()
    return message_ts or thread_ts


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

