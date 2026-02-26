from __future__ import annotations

from types import SimpleNamespace

from src.crew_runtime import CrewRuntimeCoordinator
from src.policies import PolicyConfig
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


def _orchestrator_payload(*, release_version: str | None, invoke_release_manager: bool) -> dict:
    if release_version:
        next_step = ReleaseStep.RELEASE_MANAGER_CREATED.value
        active_release = {
            "release_version": release_version,
            "step": next_step,
            "message_ts": {},
            "thread_ts": {},
            "readiness_map": {"Growth": False},
        }
    else:
        next_step = ReleaseStep.IDLE.value
        active_release = None
    return {
        "next_step": next_step,
        "next_state": {
            "active_release": active_release,
            "previous_release_version": None,
            "previous_release_completed_at": None,
            "processed_event_ids": [],
            "completed_actions": {},
            "checkpoints": [],
        },
        "tool_calls": [],
        "audit_reason": "orchestrator_decision",
        "invoke_release_manager": invoke_release_manager,
    }


def _release_manager_payload(release_version: str) -> dict:
    return {
        "next_step": ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION.value,
        "next_state": {
            "active_release": {
                "release_version": release_version,
                "step": ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION.value,
                "message_ts": {},
                "thread_ts": {},
                "readiness_map": {"Growth": False},
            },
            "previous_release_version": None,
            "previous_release_completed_at": None,
            "processed_event_ids": [],
            "completed_actions": {},
            "checkpoints": [],
        },
        "tool_calls": [],
        "audit_reason": "release_manager_decision",
    }


def test_orchestrator_session_created_once_and_reused(monkeypatch, tmp_path) -> None:
    calls = {"orchestrator_agent": 0, "orchestrator_task": 0}

    def _build_orchestrator_agent(*, policies):  # noqa: ANN001
        calls["orchestrator_agent"] += 1
        return object()

    def _build_orchestrator_task(*, agent, slack_tools):  # noqa: ANN001
        calls["orchestrator_task"] += 1
        return object()

    monkeypatch.setattr("src.crew_runtime.build_orchestrator_agent", _build_orchestrator_agent)
    monkeypatch.setattr("src.crew_runtime.build_orchestrator_task", _build_orchestrator_task)

    agent_ids: list[int] = []

    def _run_orchestrator(self, *, payload):  # noqa: ANN001
        agent_ids.append(id(self._orchestrator_agent))
        return _orchestrator_payload(release_version=None, invoke_release_manager=False), []

    monkeypatch.setattr(CrewRuntimeCoordinator, "_run_orchestrator", _run_orchestrator)

    gateway = SlackGateway(bot_token="xoxb-test-token", events_path=tmp_path / "events.jsonl")
    coordinator = CrewRuntimeCoordinator(
        policy=_policy(),
        slack_gateway=gateway,
        memory_db_path=str(tmp_path / "memory.db"),
    )

    state = WorkflowState()
    coordinator.kickoff(state=state, events=[], config=_config())
    coordinator.kickoff(state=state, events=[], config=_config())

    assert calls["orchestrator_agent"] == 1
    assert calls["orchestrator_task"] == 0
    assert len(agent_ids) == 2
    assert agent_ids[0] == agent_ids[1]


def test_release_manager_session_reused_per_release(monkeypatch, tmp_path) -> None:
    calls = {"release_manager_agent": 0, "release_manager_task": 0}

    monkeypatch.setattr("src.crew_runtime.build_orchestrator_agent", lambda *, policies: object())
    monkeypatch.setattr("src.crew_runtime.build_orchestrator_task", lambda *, agent, slack_tools: {"task": "orchestrator"})

    def _build_release_manager_agent(*, policies, release_version):  # noqa: ANN001
        calls["release_manager_agent"] += 1
        return {"release_version": release_version}

    def _build_release_manager_task(*, agent, slack_tools):  # noqa: ANN001
        calls["release_manager_task"] += 1
        return {"task": "release_manager"}

    monkeypatch.setattr("src.crew_runtime.build_release_manager_agent", _build_release_manager_agent)
    monkeypatch.setattr("src.crew_runtime.build_release_manager_task", _build_release_manager_task)

    def _run_orchestrator(self, *, payload):  # noqa: ANN001
        return _orchestrator_payload(release_version="5.104.0", invoke_release_manager=True), []

    seen_rm_agents: list[int] = []

    def _run_release_manager(self, *, agent, payload):  # noqa: ANN001
        seen_rm_agents.append(id(agent))
        return _release_manager_payload("5.104.0"), []

    monkeypatch.setattr(CrewRuntimeCoordinator, "_run_orchestrator", _run_orchestrator)
    monkeypatch.setattr(CrewRuntimeCoordinator, "_run_release_manager", _run_release_manager)

    gateway = SlackGateway(bot_token="xoxb-test-token", events_path=tmp_path / "events.jsonl")
    coordinator = CrewRuntimeCoordinator(
        policy=_policy(),
        slack_gateway=gateway,
        memory_db_path=str(tmp_path / "memory.db"),
    )

    state = WorkflowState()
    first = coordinator.kickoff(state=state, events=[], config=_config())
    second = coordinator.kickoff(state=first.next_state, events=[], config=_config())

    assert first.actor == "release_manager"
    assert second.actor == "release_manager"
    assert len(seen_rm_agents) == 2
    assert seen_rm_agents[0] == seen_rm_agents[1]
    assert calls["release_manager_agent"] == 1
    assert calls["release_manager_task"] == 0


def test_release_manager_sessions_cleared_on_idle_transition(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.crew_runtime.build_orchestrator_agent", lambda *, policies: object())
    monkeypatch.setattr("src.crew_runtime.build_orchestrator_task", lambda *, agent, slack_tools: {"task": "orchestrator"})
    monkeypatch.setattr("src.crew_runtime.build_release_manager_agent", lambda *, policies, release_version: object())
    monkeypatch.setattr("src.crew_runtime.build_release_manager_task", lambda *, agent, slack_tools: {"task": "release_manager"})

    run_count = {"value": 0}

    def _run_orchestrator(self, *, payload):  # noqa: ANN001
        run_count["value"] += 1
        if run_count["value"] == 1:
            return _orchestrator_payload(release_version="5.104.0", invoke_release_manager=True), []
        return _orchestrator_payload(release_version=None, invoke_release_manager=False), []

    def _run_release_manager(self, *, agent, payload):  # noqa: ANN001
        return _release_manager_payload("5.104.0"), []

    monkeypatch.setattr(CrewRuntimeCoordinator, "_run_orchestrator", _run_orchestrator)
    monkeypatch.setattr(CrewRuntimeCoordinator, "_run_release_manager", _run_release_manager)

    gateway = SlackGateway(bot_token="xoxb-test-token", events_path=tmp_path / "events.jsonl")
    coordinator = CrewRuntimeCoordinator(
        policy=_policy(),
        slack_gateway=gateway,
        memory_db_path=str(tmp_path / "memory.db"),
    )

    first = coordinator.kickoff(state=WorkflowState(), events=[], config=_config())
    assert first.next_step == ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION
    assert list(coordinator._release_manager_agents.keys()) == ["5.104.0"]

    second = coordinator.kickoff(state=first.next_state, events=[], config=_config())
    assert second.next_step == ReleaseStep.IDLE
    assert coordinator._release_manager_agents == {}
