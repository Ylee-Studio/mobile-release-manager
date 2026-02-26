from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.release_workflow import ReleaseWorkflow, RuntimeConfig
from src.tools.slack_tools import SlackGateway
from src.workflow_state import ReleaseContext, ReleaseStep, WorkflowState


READINESS_OWNERS = {
    "Growth": "<@U00000001>",
    "Core": "<@U00000002>",
    "AutoCut": "<@U00000003>",
    "AI Tools": "<@U00000004>",
    "Activation": "<@U00000005>",
    "Влит релиз Движка в dev": "<@U00000006>",
    "Влит релиз FeaturesKit в dev": "<@U00000007>",
}


def _runtime_config(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        slack_channel_id="C_RELEASE",
        slack_bot_token="xoxb-test-token",
        timezone="Europe/Moscow",
        heartbeat_active_minutes=15,
        heartbeat_idle_minutes=240,
        jira_project_keys=["IOS"],
        readiness_owners=READINESS_OWNERS,
        memory_db_path=str(tmp_path / "crewai_memory.db"),
        audit_log_path=str(tmp_path / "workflow_audit.jsonl"),
        slack_events_path=str(tmp_path / "slack_events.jsonl"),
        agent_pid_path=str(tmp_path / "agent.pid"),
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


class _FailIfCalledCrewRuntime:
    def kickoff(self, **kwargs):  # noqa: ANN003
        raise AssertionError(f"crew runtime must not be called in readiness gate: {kwargs}")

    def resume_from_pending(self, **kwargs):  # noqa: ANN003
        raise AssertionError(f"resume_from_pending must not be called in readiness gate: {kwargs}")


def _seed_wait_readiness_state(memory: _InMemoryNativeMemory) -> None:
    seeded_state = WorkflowState(
        active_release=ReleaseContext(
            release_version="5.104.0",
            step=ReleaseStep.WAIT_READINESS_CONFIRMATIONS,
            slack_channel_id="C_RELEASE",
            message_ts={"generic_message": "177.700"},
            thread_ts={"generic_message": "177.700"},
            readiness_map={},
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


def _build_workflow(tmp_path: Path) -> tuple[ReleaseWorkflow, Path]:
    cfg = _runtime_config(tmp_path)
    memory = _InMemoryNativeMemory()
    _seed_wait_readiness_state(memory)
    gateway = SlackGateway(bot_token=cfg.slack_bot_token, events_path=Path(cfg.slack_events_path))
    workflow = ReleaseWorkflow(
        config=cfg,
        memory=memory,
        slack_gateway=gateway,
        crew_runtime=_FailIfCalledCrewRuntime(),
    )
    return workflow, Path(cfg.slack_events_path)


def test_wait_readiness_ignores_invalid_and_duplicate_confirmations(tmp_path: Path) -> None:
    workflow, events_path = _build_workflow(tmp_path)

    _append_event(events_path, _message_event(event_id="1", text="Growth не готов"))
    state = workflow.tick(trigger_reason="signal_trigger")
    assert state.step == ReleaseStep.WAIT_READINESS_CONFIRMATIONS
    assert state.active_release is not None
    assert state.active_release.readiness_map["Growth"] is False

    _append_event(events_path, _message_event(event_id="2", text="Growth готов"))
    state = workflow.tick(trigger_reason="signal_trigger")
    assert state.step == ReleaseStep.WAIT_READINESS_CONFIRMATIONS
    assert state.active_release is not None
    assert state.active_release.readiness_map["Growth"] is True

    _append_event(events_path, _message_event(event_id="3", text="Growth готов"))
    state = workflow.tick(trigger_reason="signal_trigger")
    assert state.step == ReleaseStep.WAIT_READINESS_CONFIRMATIONS
    assert state.active_release is not None
    assert state.active_release.readiness_map["Growth"] is True


def test_wait_readiness_moves_to_next_state_only_after_all_seven_confirmed(tmp_path: Path) -> None:
    workflow, events_path = _build_workflow(tmp_path)
    confirmations = [
        ("1", "Growth готов"),
        ("2", "Core готово"),
        ("3", "AutoCut ready"),
        ("4", "AI Tools done"),
        ("5", "Activation ok"),
        ("6", "Влит релиз Движка в dev готов"),
        ("7", "Влит релиз FeaturesKit в dev готово"),
    ]

    for idx, (event_id, text) in enumerate(confirmations, start=1):
        _append_event(events_path, _message_event(event_id=event_id, text=text))
        state = workflow.tick(trigger_reason="signal_trigger")
        assert state.active_release is not None
        if idx < len(confirmations):
            assert state.step == ReleaseStep.WAIT_READINESS_CONFIRMATIONS
            assert state.is_paused is True
        else:
            assert state.step == ReleaseStep.READY_FOR_BRANCH_CUT
            assert state.is_paused is False

    final_map = state.active_release.readiness_map
    assert set(final_map.keys()) == set(READINESS_OWNERS.keys())
    assert all(final_map[point] is True for point in READINESS_OWNERS)


def test_wait_readiness_accepts_owner_message_without_team_name(tmp_path: Path) -> None:
    workflow, events_path = _build_workflow(tmp_path)

    _append_event(
        events_path,
        _message_event(event_id="8", text="готово", user_id="U00000001"),
    )
    state = workflow.tick(trigger_reason="signal_trigger")

    assert state.step == ReleaseStep.WAIT_READINESS_CONFIRMATIONS
    assert state.active_release is not None
    assert state.active_release.readiness_map["Growth"] is True
