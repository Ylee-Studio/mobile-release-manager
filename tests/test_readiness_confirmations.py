from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.release_workflow import ReleaseWorkflow, RuntimeConfig
from src.runtime_engine import RuntimeDecision
from src.tools.slack_tools import SlackGateway
from src.workflow_state import ReleaseContext, ReleaseStep, WorkflowState


READINESS_OWNERS = {
    "Growth": "<@U00000001>",
    "Core": "<@U00000002>",
}


def _runtime_config(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        slack_channel_id="C_RELEASE",
        slack_bot_token="xoxb-test-token",
        timezone="Europe/Moscow",
        jira_project_keys=["IOS"],
        readiness_owners=READINESS_OWNERS,
        memory_db_path=str(tmp_path / "workflow_state.jsonl"),
        audit_log_path=str(tmp_path / "workflow_audit.jsonl"),
        slack_events_path=str(tmp_path / "slack_events.jsonl"),
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


@dataclass
class _RecordingRuntime:
    decisions: list[RuntimeDecision]
    received_events: list[list[Any]] = field(default_factory=list)

    def kickoff(self, *, state, events, config, now=None, trigger_reason="event_trigger"):  # noqa: ANN001
        _ = (state, config, now, trigger_reason)
        self.received_events.append(events)
        if self.decisions:
            return self.decisions.pop(0)
        raise AssertionError("no scripted decision left for runtime")


def _seed_wait_readiness_state(memory: _InMemoryNativeMemory) -> None:
    seeded_state = WorkflowState(
        active_release=ReleaseContext(
            release_version="5.104.0",
            step=ReleaseStep.WAIT_READINESS_CONFIRMATIONS,
            slack_channel_id="C_RELEASE",
            message_ts={"generic_message": "177.700"},
            thread_ts={"generic_message": "177.700"},
            readiness_map={"Growth": False, "Core": False},
        ),
        flow_execution_id="flow-readiness-1",
        flow_paused_at="2026-02-27T10:00:00+00:00",
        pause_reason="awaiting_confirmation:WAIT_READINESS_CONFIRMATIONS",
    )
    memory.remember(content="{}", metadata={"state": seeded_state.to_dict(), "reason": "seed_wait_readiness"})


def _append_event(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _message_event(*, event_id: str, text: str, user_id: str = "U99999999") -> dict:
    return {
        "event_id": event_id,
        "event_type": "message",
        "channel_id": "C_RELEASE",
        "text": text,
        "thread_ts": "177.700",
        "message_ts": f"177.{event_id}",
        "metadata": {
            "source": "events_api",
            "user_id": user_id,
        },
    }


def _build_workflow(tmp_path: Path, *, runtime: _RecordingRuntime) -> tuple[ReleaseWorkflow, Path]:
    cfg = _runtime_config(tmp_path)
    memory = _InMemoryNativeMemory()
    _seed_wait_readiness_state(memory)
    gateway = SlackGateway(bot_token=cfg.slack_bot_token, events_path=Path(cfg.slack_events_path))
    workflow = ReleaseWorkflow(
        config=cfg,
        memory=memory,
        slack_gateway=gateway,
        legacy_runtime=runtime,
    )
    return workflow, Path(cfg.slack_events_path)


def test_wait_readiness_state_follows_agent_decision_payload(tmp_path: Path) -> None:
    decision_state = WorkflowState(
        active_release=ReleaseContext(
            release_version="5.104.0",
            step=ReleaseStep.WAIT_READINESS_CONFIRMATIONS,
            slack_channel_id="C_RELEASE",
            message_ts={"generic_message": "177.700"},
            thread_ts={"generic_message": "177.700"},
            readiness_map={"Growth": True, "Core": False},
        ),
        flow_execution_id="flow-readiness-1",
        flow_paused_at="2026-02-27T10:00:00+00:00",
        pause_reason="agent_waiting_readiness",
    )
    runtime = _RecordingRuntime(
        decisions=[
            RuntimeDecision(
                next_step=ReleaseStep.WAIT_READINESS_CONFIRMATIONS,
                next_state=decision_state,
                audit_reason="agent_readiness_partial",
                actor="flow_agent",
                flow_lifecycle="paused",
            )
        ]
    )
    workflow, events_path = _build_workflow(tmp_path, runtime=runtime)
    _append_event(events_path, _message_event(event_id="1", text="Growth готов"))

    state = workflow.tick(trigger_reason="signal_trigger")

    assert len(runtime.received_events) == 1
    assert state.step == ReleaseStep.WAIT_READINESS_CONFIRMATIONS
    assert state.active_release is not None
    assert state.active_release.readiness_map == {"Growth": True, "Core": False}


def test_wait_readiness_transition_to_wait_branch_cut_is_agent_driven(tmp_path: Path) -> None:
    decision_state = WorkflowState(
        active_release=ReleaseContext(
            release_version="5.104.0",
            step=ReleaseStep.WAIT_BRANCH_CUT,
            slack_channel_id="C_RELEASE",
            message_ts={"generic_message": "177.700"},
            thread_ts={"generic_message": "177.700"},
            readiness_map={"Growth": True, "Core": True},
        ),
        flow_execution_id=None,
        flow_paused_at=None,
        pause_reason=None,
    )
    runtime = _RecordingRuntime(
        decisions=[
            RuntimeDecision(
                next_step=ReleaseStep.WAIT_BRANCH_CUT,
                next_state=decision_state,
                audit_reason="agent_readiness_complete",
                actor="flow_agent",
                flow_lifecycle="completed",
            )
        ]
    )
    workflow, events_path = _build_workflow(tmp_path, runtime=runtime)
    _append_event(events_path, _message_event(event_id="2", text="Core готов"))

    state = workflow.tick(trigger_reason="signal_trigger")

    assert len(runtime.received_events) == 1
    assert state.step == ReleaseStep.WAIT_BRANCH_CUT
    assert state.active_release is not None
    assert state.active_release.readiness_map == {"Growth": True, "Core": True}
