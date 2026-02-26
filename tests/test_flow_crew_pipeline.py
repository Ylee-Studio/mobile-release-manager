from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from src.crew_memory import CrewAIMemory
from src.crew_runtime import CrewRuntimeCoordinator
from src.policies import PolicyConfig
from src.release_workflow import ReleaseWorkflow, RuntimeConfig
from src.tools.slack_tools import SlackGateway
from src.workflow_state import ReleaseStep, WorkflowState


def _policy() -> PolicyConfig:
    return PolicyConfig.from_dict({})


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        slack_channel_id="C_RELEASE",
        timezone="Europe/Moscow",
        jira_project_keys=["IOS"],
        readiness_owners={"Growth": "owner"},
    )


@dataclass
class _KickoffProbeRuntime:
    trigger_reason: str | None = None

    def kickoff(self, *, state, events, config, now=None, trigger_reason="heartbeat_timer"):  # noqa: ANN001
        self.trigger_reason = trigger_reason
        return SimpleNamespace(
            next_step=state.step,
            next_state=state,
            state_patch={},
            tool_calls=[],
            audit_reason="probe",
            actor="orchestrator",
        )


def _runtime_config(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        slack_channel_id="C_RELEASE",
        slack_bot_token="xoxb-test-token",
        timezone="Europe/Moscow",
        heartbeat_active_minutes=15,
        heartbeat_idle_minutes=240,
        jira_project_keys=["IOS"],
        readiness_owners={"Growth": "owner"},
        memory_db_path=str(tmp_path / "crewai_memory.db"),
        audit_log_path=str(tmp_path / "workflow_audit.jsonl"),
        slack_events_path=str(tmp_path / "slack_events.jsonl"),
        agent_pid_path=str(tmp_path / "agent.pid"),
    )


def test_release_workflow_propagates_trigger_reason_to_runtime(tmp_path: Path) -> None:
    config = _runtime_config(tmp_path)
    memory = CrewAIMemory(db_path=config.memory_db_path)
    gateway = SlackGateway(bot_token="xoxb-test-token", events_path=Path(config.slack_events_path))
    runtime = _KickoffProbeRuntime()
    workflow = ReleaseWorkflow(
        config=config,
        memory=memory,
        slack_gateway=gateway,
        crew_runtime=runtime,
    )

    state = workflow.tick(trigger_reason="signal_trigger")

    assert state.step == ReleaseStep.IDLE
    assert runtime.trigger_reason == "signal_trigger"


def test_kickoff_returns_safe_decision_on_invalid_structured_output(monkeypatch, tmp_path: Path) -> None:
    class _InvalidCrewResult:
        raw = "not-a-json-payload"
        tasks_output = []

    gateway = SlackGateway(bot_token="xoxb-test-token", events_path=tmp_path / "events.jsonl")
    coordinator = CrewRuntimeCoordinator(
        policy=_policy(),
        slack_gateway=gateway,
        memory_db_path=str(tmp_path / "memory.db"),
    )

    monkeypatch.setattr(
        coordinator,
        "_execute_crew",
        lambda **kwargs: _InvalidCrewResult(),  # noqa: ARG005
    )
    coordinator._orchestrator_crew = object()

    initial_state = WorkflowState()
    decision = coordinator.kickoff(
        state=initial_state,
        events=[],
        config=_config(),
    )

    assert decision.next_step == initial_state.step
    assert decision.next_state == initial_state
    assert decision.audit_reason.startswith("crew_runtime_error:")
