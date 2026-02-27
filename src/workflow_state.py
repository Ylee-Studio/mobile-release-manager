"""Workflow state models for release train orchestration."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReleaseStep(str, Enum):
    IDLE = "IDLE"
    WAIT_START_APPROVAL = "WAIT_START_APPROVAL"
    WAIT_MANUAL_RELEASE_CONFIRMATION = "WAIT_MANUAL_RELEASE_CONFIRMATION"
    WAIT_MEETING_CONFIRMATION = "WAIT_MEETING_CONFIRMATION"
    WAIT_READINESS_CONFIRMATIONS = "WAIT_READINESS_CONFIRMATIONS"
    READY_FOR_BRANCH_CUT = "READY_FOR_BRANCH_CUT"


_STATUS_TO_STEP: dict[str, ReleaseStep] = {
    "AWAITING_START_APPROVAL": ReleaseStep.WAIT_START_APPROVAL,
    "WAITING_START_APPROVAL": ReleaseStep.WAIT_START_APPROVAL,
    "RELEASE_MANAGER_ACTIVE": ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION,
    "RELEASE_MANAGER_CREATED": ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION,
    "AWAITING_MANUAL_RELEASE_CONFIRMATION": ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION,
    "JIRA_RELEASE_READY": ReleaseStep.WAIT_MEETING_CONFIRMATION,
    "JIRA_RELEASE_CREATED": ReleaseStep.WAIT_MEETING_CONFIRMATION,
    "AWAITING_MEETING_CONFIRMATION": ReleaseStep.WAIT_MEETING_CONFIRMATION,
    "AWAITING_READINESS_CONFIRMATIONS": ReleaseStep.WAIT_READINESS_CONFIRMATIONS,
}


def _coerce_step(value: Any, *, fallback: ReleaseStep) -> ReleaseStep:
    if isinstance(value, ReleaseStep):
        return value
    if isinstance(value, str):
        normalized = value.strip().upper()
        if not normalized:
            return fallback
        if normalized in ReleaseStep._value2member_map_:
            return ReleaseStep(normalized)
        if normalized in _STATUS_TO_STEP:
            return _STATUS_TO_STEP[normalized]
    return fallback


class ReleaseContext(BaseModel):
    release_version: str
    step: ReleaseStep
    updated_at: str = Field(default_factory=utc_now_iso)
    slack_channel_id: str | None = None
    message_ts: dict[str, str] = Field(default_factory=dict)
    thread_ts: dict[str, str] = Field(default_factory=dict)
    readiness_map: dict[str, bool] = Field(default_factory=dict)

    def set_step(self, step: ReleaseStep) -> None:
        self.step = step
        self.updated_at = utc_now_iso()


class WorkflowState(BaseModel):
    active_release: ReleaseContext | None = None
    previous_release_version: str | None = None
    previous_release_completed_at: str | None = None
    flow_execution_id: str | None = None
    flow_paused_at: str | None = None
    pause_reason: str | None = None
    checkpoints: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def step(self) -> ReleaseStep:
        if not self.active_release:
            return ReleaseStep.IDLE
        return self.active_release.step

    @property
    def is_paused(self) -> bool:
        return bool(self.flow_execution_id and self.flow_paused_at)

    def to_dict(self) -> dict[str, Any]:
        active = None
        if self.active_release:
            active = self.active_release.model_dump()
            active["step"] = self.active_release.step.value
        return {
            "active_release": active,
            "previous_release_version": self.previous_release_version,
            "previous_release_completed_at": self.previous_release_completed_at,
            "flow_execution_id": self.flow_execution_id,
            "flow_paused_at": self.flow_paused_at,
            "pause_reason": self.pause_reason,
            "checkpoints": self.checkpoints,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "WorkflowState":
        if not raw:
            return cls()

        active_raw = raw.get("active_release")
        active_release = None
        if isinstance(active_raw, dict):
            release_version = str(
                active_raw.get("release_version")
                or active_raw.get("version")
                or ""
            ).strip()
            if release_version:
                step = _coerce_step(
                    active_raw.get("step") or active_raw.get("status"),
                    fallback=ReleaseStep.IDLE,
                )
                message_ts_raw = active_raw.get("message_ts", {})
                thread_ts_raw = active_raw.get("thread_ts", {})
                readiness_raw = active_raw.get("readiness_map", active_raw.get("readiness", {}))
                channel_id = active_raw.get("slack_channel_id") or active_raw.get("channel_id")

                message_ts = message_ts_raw if isinstance(message_ts_raw, dict) else {}
                thread_ts = thread_ts_raw if isinstance(thread_ts_raw, dict) else {}
                readiness_map = readiness_raw if isinstance(readiness_raw, dict) else {}

                active_release = ReleaseContext(
                    release_version=release_version,
                    step=step,
                    updated_at=active_raw.get("updated_at", utc_now_iso()),
                    slack_channel_id=channel_id,
                    message_ts=message_ts,
                    thread_ts=thread_ts,
                    readiness_map=readiness_map,
                )

        return cls(
            active_release=active_release,
            previous_release_version=raw.get("previous_release_version"),
            previous_release_completed_at=raw.get("previous_release_completed_at"),
            flow_execution_id=raw.get("flow_execution_id"),
            flow_paused_at=raw.get("flow_paused_at"),
            pause_reason=raw.get("pause_reason"),
            checkpoints=list(raw.get("checkpoints", [])),
        )


class ReleaseFlowState(BaseModel):
    """CrewAI Flow state wrapper for release workflow execution."""

    state: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    trigger_reason: str = "event_trigger"
    now_iso: str = ""
