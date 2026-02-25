"""Workflow state models for release train orchestration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReleaseStep(str, Enum):
    IDLE = "IDLE"
    WAIT_START_APPROVAL = "WAIT_START_APPROVAL"
    RELEASE_MANAGER_CREATED = "RELEASE_MANAGER_CREATED"
    JIRA_RELEASE_CREATED = "JIRA_RELEASE_CREATED"
    WAIT_MEETING_CONFIRMATION = "WAIT_MEETING_CONFIRMATION"
    WAIT_READINESS_CONFIRMATIONS = "WAIT_READINESS_CONFIRMATIONS"
    READY_FOR_BRANCH_CUT = "READY_FOR_BRANCH_CUT"


@dataclass
class ReleaseContext:
    release_version: str
    step: ReleaseStep
    updated_at: str = field(default_factory=utc_now_iso)
    slack_channel_id: str | None = None
    message_ts: dict[str, str] = field(default_factory=dict)
    thread_ts: dict[str, str] = field(default_factory=dict)
    readiness_map: dict[str, bool] = field(default_factory=dict)

    def set_step(self, step: ReleaseStep) -> None:
        self.step = step
        self.updated_at = utc_now_iso()


@dataclass
class WorkflowState:
    active_release: ReleaseContext | None = None
    previous_release_version: str | None = None
    previous_release_completed_at: str | None = None
    processed_event_ids: list[str] = field(default_factory=list)
    completed_actions: dict[str, str] = field(default_factory=dict)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)

    @property
    def step(self) -> ReleaseStep:
        if not self.active_release:
            return ReleaseStep.IDLE
        return self.active_release.step

    def mark_event_processed(self, event_id: str) -> None:
        if event_id in self.processed_event_ids:
            return
        self.processed_event_ids.append(event_id)
        if len(self.processed_event_ids) > 1000:
            self.processed_event_ids = self.processed_event_ids[-1000:]

    def is_action_done(self, action_key: str) -> bool:
        return action_key in self.completed_actions

    def mark_action_done(self, action_key: str) -> None:
        self.completed_actions[action_key] = utc_now_iso()

    def add_checkpoint(self, *, phase: str, step: ReleaseStep, tool: str, release_version: str) -> None:
        self.checkpoints.append(
            {
                "phase": phase,
                "step": step.value,
                "tool": tool,
                "release_version": release_version,
                "at": utc_now_iso(),
            }
        )
        if len(self.checkpoints) > 200:
            self.checkpoints = self.checkpoints[-200:]

    def to_dict(self) -> dict[str, Any]:
        active = None
        if self.active_release:
            active = asdict(self.active_release)
            active["step"] = self.active_release.step.value
        return {
            "active_release": active,
            "previous_release_version": self.previous_release_version,
            "previous_release_completed_at": self.previous_release_completed_at,
            "processed_event_ids": self.processed_event_ids,
            "completed_actions": self.completed_actions,
            "checkpoints": self.checkpoints,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "WorkflowState":
        if not raw:
            return cls()

        active_raw = raw.get("active_release")
        active_release = None
        if active_raw:
            active_release = ReleaseContext(
                release_version=active_raw["release_version"],
                step=ReleaseStep(active_raw["step"]),
                updated_at=active_raw.get("updated_at", utc_now_iso()),
                slack_channel_id=active_raw.get("slack_channel_id"),
                message_ts=dict(active_raw.get("message_ts", {})),
                thread_ts=dict(active_raw.get("thread_ts", {})),
                readiness_map=dict(active_raw.get("readiness_map", {})),
            )

        return cls(
            active_release=active_release,
            previous_release_version=raw.get("previous_release_version"),
            previous_release_completed_at=raw.get("previous_release_completed_at"),
            processed_event_ids=list(raw.get("processed_event_ids", [])),
            completed_actions=dict(raw.get("completed_actions", {})),
            checkpoints=list(raw.get("checkpoints", [])),
        )
