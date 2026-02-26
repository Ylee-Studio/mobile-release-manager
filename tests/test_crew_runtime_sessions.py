from __future__ import annotations

from types import SimpleNamespace

from src.crew_runtime import CrewDecision, CrewRuntimeCoordinator, FlowTurnContext
from src.policies import PolicyConfig
from src.tools.slack_tools import SlackEvent, SlackGateway
from src.workflow_state import ReleaseContext, ReleaseStep, WorkflowState


def _policy() -> PolicyConfig:
    return PolicyConfig.from_dict({})


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        slack_channel_id="C_RELEASE",
        timezone="Europe/Moscow",
        jira_project_keys=["IOS"],
        readiness_owners={"Growth": "owner"},
    )


def _build_coordinator(monkeypatch, tmp_path) -> CrewRuntimeCoordinator:
    monkeypatch.setattr("src.crew_runtime.build_orchestrator_agent", lambda *, policies: object())
    monkeypatch.setattr("src.crew_runtime.build_orchestrator_task", lambda *, agent, slack_tools: {"task": "orchestrator"})
    monkeypatch.setattr("src.crew_runtime.build_release_manager_agent", lambda *, policies, release_version: object())
    monkeypatch.setattr("src.crew_runtime.build_release_manager_task", lambda *, agent, slack_tools: {"task": "release_manager"})
    gateway = SlackGateway(bot_token="xoxb-test-token", events_path=tmp_path / "events.jsonl")
    return CrewRuntimeCoordinator(
        policy=_policy(),
        slack_gateway=gateway,
        memory_db_path=str(tmp_path / "memory.db"),
    )


def test_paused_flow_without_confirmation_event_does_not_execute(monkeypatch, tmp_path) -> None:
    coordinator = _build_coordinator(monkeypatch, tmp_path)
    flow_calls = {"kickoff": 0}

    class _FlowStub:
        def kickoff(self, *, inputs):  # noqa: ANN001
            flow_calls["kickoff"] += 1
            return CrewDecision(next_step=ReleaseStep.IDLE, next_state=WorkflowState())

    coordinator.flow = _FlowStub()
    state = WorkflowState(
        active_release=ReleaseContext(release_version="5.104.0", step=ReleaseStep.WAIT_MEETING_CONFIRMATION),
        flow_execution_id="flow-1",
        flow_paused_at="2026-01-01T00:00:00+00:00",
        pause_reason="awaiting_confirmation:WAIT_MEETING_CONFIRMATION",
    )
    decision = coordinator.kickoff(state=state, events=[], config=_config())

    assert flow_calls["kickoff"] == 0
    assert decision.flow_lifecycle == "paused"
    assert decision.audit_reason.startswith("flow_paused_waiting_confirmation:")


def test_paused_flow_with_confirmation_event_runs_kickoff(monkeypatch, tmp_path) -> None:
    coordinator = _build_coordinator(monkeypatch, tmp_path)
    flow_calls = {"kickoff": 0}

    class _FlowStub:
        def kickoff(self, *, inputs):  # noqa: ANN001
            _ = inputs
            flow_calls["kickoff"] += 1
            return CrewDecision(
                next_step=ReleaseStep.WAIT_READINESS_CONFIRMATIONS,
                next_state=WorkflowState(
                    active_release=ReleaseContext(
                        release_version="5.104.0",
                        step=ReleaseStep.WAIT_READINESS_CONFIRMATIONS,
                    )
                ),
                flow_lifecycle="running",
            )

    coordinator.flow = _FlowStub()
    state = WorkflowState(
        active_release=ReleaseContext(release_version="5.104.0", step=ReleaseStep.WAIT_MEETING_CONFIRMATION),
        flow_execution_id="flow-1",
        flow_paused_at="2026-01-01T00:00:00+00:00",
        pause_reason="awaiting_confirmation:WAIT_MEETING_CONFIRMATION",
    )
    decision = coordinator.kickoff(
        state=state,
        events=[
            SlackEvent(
                event_id="ev-1",
                event_type="approval_confirmed",
                channel_id="C_RELEASE",
            )
        ],
        config=_config(),
    )

    assert flow_calls["kickoff"] == 1
    assert decision.next_step == ReleaseStep.WAIT_READINESS_CONFIRMATIONS
    assert decision.flow_lifecycle == "running"


def test_maybe_pause_flow_marks_state_and_persists_pending(monkeypatch, tmp_path) -> None:
    coordinator = _build_coordinator(monkeypatch, tmp_path)

    decision = CrewDecision(
        next_step=ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION,
        next_state=WorkflowState(
            active_release=ReleaseContext(
                release_version="5.104.0",
                step=ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION,
            )
        ),
        flow_lifecycle="running",
    )
    ctx = FlowTurnContext(
        payload={"events": []},
        current_state=WorkflowState(),
        orchestrator_payload={},
        orchestrator_native_calls=[],
        orchestrator_decision=decision,
    )

    coordinator.maybe_pause_flow(flow=coordinator.flow, decision=decision, context=ctx)

    assert decision.flow_lifecycle == "paused"
    assert decision.next_state.flow_execution_id
    assert decision.next_state.flow_paused_at
    pending = coordinator._flow_persistence.load_pending_feedback(decision.next_state.flow_execution_id)  # noqa: SLF001
    assert pending is not None


def test_resume_from_pending_uses_from_pending_flow(monkeypatch, tmp_path) -> None:
    coordinator = _build_coordinator(monkeypatch, tmp_path)
    called = {"from_pending": 0, "resume": 0}

    class _RestoredFlow:
        def resume(self, feedback=""):  # noqa: ANN001
            _ = feedback
            called["resume"] += 1
            return "ok"

    def _from_pending(cls, flow_id, persistence=None, **kwargs):  # noqa: ANN001, ANN003
        _ = (cls, flow_id, persistence, kwargs)
        called["from_pending"] += 1
        return _RestoredFlow()

    monkeypatch.setattr(type(coordinator.flow), "from_pending", classmethod(_from_pending), raising=False)
    state = WorkflowState(
        active_release=ReleaseContext(release_version="5.104.0", step=ReleaseStep.WAIT_MEETING_CONFIRMATION),
        flow_execution_id="flow-1",
        flow_paused_at="2026-01-01T00:00:00+00:00",
        pause_reason="awaiting_confirmation:WAIT_MEETING_CONFIRMATION",
    )

    decision = coordinator.resume_from_pending(flow_id="flow-1", feedback="Подтвердить", state=state)

    assert called["from_pending"] == 1
    assert called["resume"] == 1
    assert decision.flow_lifecycle == "running"
    assert decision.audit_reason == "resume_dispatched"
