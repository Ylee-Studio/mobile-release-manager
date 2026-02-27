from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.runtime_engine import RuntimeDecision
from src.release_workflow import ReleaseWorkflow, RuntimeConfig
from src.tools.slack_tools import SlackGateway
from src.workflow_state import ReleaseContext, ReleaseStep, WorkflowState


def _runtime_config(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        slack_channel_id="C_RELEASE",
        slack_bot_token="xoxb-test-token",
        timezone="Europe/Moscow",
        jira_project_keys=["IOS"],
        readiness_owners={"Growth": "@owner"},
        memory_db_path=str(tmp_path / "workflow_state.jsonl"),
        audit_log_path=str(tmp_path / "workflow_audit.jsonl"),
        slack_events_path=str(tmp_path / "slack_events.jsonl"),
    )


def _append_event(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


class _PauseResumeRuntime:
    def kickoff(self, *, state: WorkflowState, events, config, now=None, trigger_reason="event_trigger"):  # noqa: ANN001
        _ = (config, now, trigger_reason)
        if state.step == ReleaseStep.IDLE:
            return RuntimeDecision(
                next_step=ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION,
                next_state=WorkflowState(
                    active_release=ReleaseContext(
                        release_version="5.104.0",
                        step=ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION,
                        slack_channel_id="C_RELEASE",
                    ),
                    flow_execution_id="flow-1",
                    flow_paused_at="2026-02-27T10:00:00+00:00",
                    pause_reason="awaiting_confirmation:WAIT_MANUAL_RELEASE_CONFIRMATION",
                ),
                audit_reason="manual_confirmation_requested",
                flow_lifecycle="paused",
            )
        has_approval = any(event.event_type == "approval_confirmed" for event in events)
        if state.is_paused and not has_approval:
            return RuntimeDecision(
                next_step=state.step,
                next_state=state,
                audit_reason="flow_paused_waiting_confirmation",
                flow_lifecycle="paused",
            )
        if has_approval:
            resumed = WorkflowState.from_dict(state.to_dict())
            resumed.active_release = ReleaseContext(
                release_version="5.104.0",
                step=ReleaseStep.WAIT_MEETING_CONFIRMATION,
                slack_channel_id="C_RELEASE",
            )
            resumed.flow_paused_at = None
            resumed.pause_reason = None
            return RuntimeDecision(
                next_step=ReleaseStep.WAIT_MEETING_CONFIRMATION,
                next_state=resumed,
                audit_reason="flow_resumed_from_manual_confirmation",
                flow_lifecycle="running",
            )
        return RuntimeDecision(next_step=state.step, next_state=state, audit_reason="no_change")

    def resume_from_pending(self, *, flow_id: str, feedback: str, state: WorkflowState):  # noqa: ANN001
        _ = (flow_id, feedback)
        resumed = WorkflowState.from_dict(state.to_dict())
        resumed.flow_paused_at = None
        resumed.pause_reason = None
        return RuntimeDecision(
            next_step=resumed.step,
            next_state=resumed,
            audit_reason="resume_dispatched",
            flow_lifecycle="running",
        )


@dataclass
class _MemoryRecord:
    content: str
    metadata: dict


@dataclass
class _MemoryMatch:
    record: _MemoryRecord


@dataclass
class _InMemoryNativeMemory:
    entries: list[dict] = field(default_factory=list)

    def remember(self, *, content: str, metadata: dict, **_: object) -> None:
        self.entries.append({"content": content, "metadata": metadata})

    def recall(self, **_: object) -> list[_MemoryMatch]:
        return [
            _MemoryMatch(record=_MemoryRecord(content=entry["content"], metadata=entry["metadata"]))
            for entry in reversed(self.entries)
        ]


def _build_workflow(tmp_path: Path) -> ReleaseWorkflow:
    cfg = _runtime_config(tmp_path)
    memory = _InMemoryNativeMemory()
    gateway = SlackGateway(bot_token=cfg.slack_bot_token, events_path=Path(cfg.slack_events_path))
    runtime = _PauseResumeRuntime()
    return ReleaseWorkflow(
        config=cfg,
        memory=memory,
        slack_gateway=gateway,
        legacy_runtime=runtime,
    )


def test_paused_flow_does_not_advance_without_events(tmp_path: Path) -> None:
    workflow = _build_workflow(tmp_path)

    first = workflow.tick(trigger_reason="manual_tick")
    second = workflow.tick(trigger_reason="manual_tick")

    assert first.step == ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION
    assert first.is_paused is True
    assert second.step == ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION
    assert second.is_paused is True
    assert second.flow_execution_id == "flow-1"


def test_approval_event_resumes_paused_flow_once(tmp_path: Path) -> None:
    workflow = _build_workflow(tmp_path)
    cfg = _runtime_config(tmp_path)
    events_path = Path(cfg.slack_events_path)

    paused = workflow.tick(trigger_reason="manual_tick")
    assert paused.is_paused is True

    _append_event(
        events_path,
        {
            "event_id": "ev-approve-1",
            "event_type": "approval_confirmed",
            "channel_id": "C_RELEASE",
            "text": "Подтвердить",
            "message_ts": "177.10",
            "thread_ts": "177.10",
        },
    )
    resumed = workflow.tick(trigger_reason="webhook_event")

    assert resumed.step == ReleaseStep.WAIT_MEETING_CONFIRMATION
    assert resumed.is_paused is False
    assert resumed.flow_execution_id == "flow-1"
