import json
from pathlib import Path

from src.crew_memory import CrewAIMemory
from src.crew_runtime import CrewDecision
from src.release_workflow import MANUAL_RELEASE_MESSAGE, ReleaseWorkflow, RuntimeConfig
from src.tools.slack_tools import SlackGateway
from src.workflow_state import ReleaseContext, ReleaseStep, WorkflowState


def _append_event(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


class _KickoffCompatibleRuntime:
    def kickoff(  # noqa: ANN001
        self,
        *,
        state: WorkflowState,
        events,
        config,
        now=None,
        trigger_reason="heartbeat_timer",
    ):
        _ = trigger_reason
        return self.decide(state=state, events=events, config=config, now=now)


class FakeCrewRuntime(_KickoffCompatibleRuntime):
    def decide(self, *, state: WorkflowState, events, config, now=None):  # noqa: ANN001
        if state.step == ReleaseStep.IDLE and any(e.event_type == "manual_start" for e in events):
            next_state = WorkflowState(
                active_release=ReleaseContext(
                    release_version="1.0.0",
                    step=ReleaseStep.WAIT_START_APPROVAL,
                    slack_channel_id=config.slack_channel_id,
                    readiness_map={team: False for team in config.readiness_owners},
                ),
                previous_release_version=state.previous_release_version,
                previous_release_completed_at=state.previous_release_completed_at,
                processed_event_ids=["ev-1"],
                completed_actions={"start-approval:1.0.0": "done"},
                checkpoints=[],
            )
            return CrewDecision(
                next_step=ReleaseStep.WAIT_START_APPROVAL,
                next_state=next_state,
                audit_reason="manual_start_processed",
                tool_calls=[
                    {
                        "tool": "slack_approve",
                        "reason": "start confirmation",
                        "args": {
                            "channel_id": config.slack_channel_id,
                            "text": "Подтвердите старт релиза 1.0.0.",
                            "approve_label": "Подтвердить",
                        },
                    }
                ],
            )

        return CrewDecision(
            next_step=state.step,
            next_state=state,
            audit_reason="no_change",
        )


class StartApprovalCrewRuntime(_KickoffCompatibleRuntime):
    def decide(self, *, state: WorkflowState, events, config, now=None):  # noqa: ANN001
        if state.step == ReleaseStep.IDLE and any(e.event_type == "manual_start" for e in events):
            return CrewDecision(
                next_step=ReleaseStep.WAIT_START_APPROVAL,
                next_state=WorkflowState(
                    active_release=ReleaseContext(
                        release_version="5.104.0",
                        step=ReleaseStep.WAIT_START_APPROVAL,
                        slack_channel_id=None,
                    ),
                    previous_release_version=state.previous_release_version,
                    previous_release_completed_at=state.previous_release_completed_at,
                    processed_event_ids=["ev-1"],
                    completed_actions={},
                    checkpoints=[],
                ),
                audit_reason="manual_start_processed",
                tool_calls=[
                    {
                        "tool": "functions.slack_approve",
                        "reason": "request start approval",
                        "args": {
                            "channel_id": config.slack_channel_id,
                            "text": "Подтвердите старт релизного трейна 5.104.0.",
                            "approve_label": "Подтвердить",
                        },
                    }
                ],
            )
        return CrewDecision(next_step=state.step, next_state=state, audit_reason="no_change")


class ManualReleaseFlowCrewRuntime(_KickoffCompatibleRuntime):
    def decide(self, *, state: WorkflowState, events, config, now=None):  # noqa: ANN001
        if state.step == ReleaseStep.IDLE and any(e.event_type == "manual_start" for e in events):
            return CrewDecision(
                next_step=ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION,
                next_state=WorkflowState(
                    active_release=ReleaseContext(
                        release_version="5.104.0",
                        step=ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION,
                        slack_channel_id=None,
                    ),
                    previous_release_version=state.previous_release_version,
                    previous_release_completed_at=state.previous_release_completed_at,
                    processed_event_ids=["ev-1"],
                    completed_actions={},
                    checkpoints=[],
                ),
                audit_reason="waiting_manual_release_confirmation",
                tool_calls=[
                    {
                        "tool": "functions.slack_approve",
                        "reason": "manual Jira release confirmation",
                        "args": {
                            "channel_id": config.slack_channel_id,
                            "text": "Подтвердите, что релиз 5.104.0 создан вручную в Jira plan.",
                            "approve_label": "Релиз создан",
                        },
                    }
                ],
            )
        if state.step == ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION and any(
            e.event_type == "approval_confirmed" for e in events
        ):
            event_channel = events[0].channel_id if events else config.slack_channel_id
            event_message_ts = events[0].message_ts if events else ""
            active = state.active_release
            assert active is not None
            next_state = WorkflowState(
                active_release=ReleaseContext(
                    release_version=active.release_version,
                    step=ReleaseStep.JIRA_RELEASE_CREATED,
                    slack_channel_id=active.slack_channel_id,
                    message_ts=dict(active.message_ts),
                    thread_ts=dict(active.thread_ts),
                    readiness_map=dict(active.readiness_map),
                ),
                previous_release_version=state.previous_release_version,
                previous_release_completed_at=state.previous_release_completed_at,
                processed_event_ids=[*state.processed_event_ids, "ev-2"],
                completed_actions=dict(state.completed_actions),
                checkpoints=list(state.checkpoints),
            )
            return CrewDecision(
                next_step=ReleaseStep.JIRA_RELEASE_CREATED,
                next_state=next_state,
                audit_reason="manual_release_confirmed",
                tool_calls=[
                    {
                        "tool": "functions.slack_update",
                        "reason": "acknowledge confirmation",
                        "args": {
                            "channel_id": event_channel,
                            "message_ts": event_message_ts,
                            "text": "Подтверждение получено :white_check_mark:",
                        },
                    },
                    {
                        "tool": "functions.slack_message",
                        "reason": "send manual release instructions",
                        "args": {
                            "channel_id": event_channel,
                            "text": MANUAL_RELEASE_MESSAGE,
                            "thread_ts": event_message_ts,
                        },
                    },
                ],
            )
        return CrewDecision(next_step=state.step, next_state=state, audit_reason="no_change")


class InvalidSlackMessageCrewRuntime(_KickoffCompatibleRuntime):
    def decide(self, *, state: WorkflowState, events, config, now=None):  # noqa: ANN001
        if state.step == ReleaseStep.IDLE and any(e.event_type == "manual_start" for e in events):
            next_state = WorkflowState(
                active_release=None,
                previous_release_version=state.previous_release_version,
                previous_release_completed_at=state.previous_release_completed_at,
                processed_event_ids=[*state.processed_event_ids, "ev-invalid"],
                completed_actions=dict(state.completed_actions),
                checkpoints=list(state.checkpoints),
            )
            return CrewDecision(
                next_step=ReleaseStep.IDLE,
                next_state=next_state,
                audit_reason="manual_override_to_idle",
                tool_calls=[
                    {
                        "tool": "functions.slack_message",
                        "reason": "confirm manual override",
                    }
                ],
            )
        return CrewDecision(next_step=state.step, next_state=state, audit_reason="no_change")


class ReadinessAnnouncementCrewRuntime(_KickoffCompatibleRuntime):
    def decide(self, *, state: WorkflowState, events, config, now=None):  # noqa: ANN001, ARG002
        readiness_text = (
            "Релиз <https://instories.atlassian.net/issues/?jql=JQL|5.104.0>\n\n"
            "Статус готовности к срезу:\n"
            ":hourglass_flowing_sand: Growth @owner\n\n"
            "Напишите в треде по готовности своей части.\n"
            "**Важное напоминание** – все задачи, не влитые в ветку RC до 15:00 МСК "
            "едут в релиз только после одобрения QA"
        )
        if state.step == ReleaseStep.IDLE and any(e.event_type == "manual_start" for e in events):
            return CrewDecision(
                next_step=ReleaseStep.WAIT_READINESS_CONFIRMATIONS,
                next_state=WorkflowState(
                    active_release=ReleaseContext(
                        release_version="5.104.0",
                        step=ReleaseStep.WAIT_READINESS_CONFIRMATIONS,
                        slack_channel_id="C0AGLKF6KHD",
                    ),
                    previous_release_version=state.previous_release_version,
                    previous_release_completed_at=state.previous_release_completed_at,
                    processed_event_ids=["ev-1"],
                    completed_actions=dict(state.completed_actions),
                    checkpoints=list(state.checkpoints),
                ),
                audit_reason="move_to_readiness",
                tool_calls=[
                    {
                        "tool": "slack_message",
                        "reason": "announce readiness",
                        "args": {
                            "channel_id": "C0AGLKF6KHD",
                            "text": readiness_text,
                        },
                    }
                ],
            )
        if state.step == ReleaseStep.WAIT_READINESS_CONFIRMATIONS:
            return CrewDecision(
                next_step=ReleaseStep.WAIT_READINESS_CONFIRMATIONS,
                next_state=WorkflowState.from_dict(state.to_dict()),
                audit_reason="repeat_readiness_announcement",
                tool_calls=[
                    {
                        "tool": "slack_message",
                        "reason": "repeat readiness status",
                        "args": {
                            "channel_id": "C0AGLKF6KHD",
                            "text": readiness_text,
                        },
                    }
                ],
            )
        return CrewDecision(next_step=state.step, next_state=state, audit_reason="no_change")


def test_memory_state_persists_across_restarts(tmp_path: Path) -> None:
    memory_db = tmp_path / "crewai_memory.db"
    audit_log = tmp_path / "workflow_audit.jsonl"
    slack_events = tmp_path / "slack_events.jsonl"

    config = RuntimeConfig(
        slack_channel_id="C_RELEASE",
        slack_bot_token="xoxb-test-token",
        timezone="Europe/Moscow",
        heartbeat_active_minutes=15,
        heartbeat_idle_minutes=240,
        jira_project_keys=["IOS"],
        readiness_owners={"Growth": "owner"},
        memory_db_path=str(memory_db),
        audit_log_path=str(audit_log),
        slack_events_path=str(slack_events),
        agent_pid_path=str(tmp_path / "agent.pid"),
    )
    memory = CrewAIMemory(db_path=str(memory_db))
    slack_gateway = SlackGateway(bot_token="xoxb-test-token", events_path=slack_events)

    workflow = ReleaseWorkflow(config=config, memory=memory, slack_gateway=slack_gateway, crew_runtime=FakeCrewRuntime())

    _append_event(
        slack_events,
        {
            "event_id": "ev-1",
            "event_type": "manual_start",
            "channel_id": "C_RELEASE",
            "text": "start release",
        },
    )
    first_state = workflow.tick()

    restarted_workflow = ReleaseWorkflow(
        config=config,
        memory=CrewAIMemory(db_path=str(memory_db)),
        slack_gateway=slack_gateway,
        crew_runtime=FakeCrewRuntime(),
    )
    second_state = restarted_workflow.tick()

    assert first_state.step == ReleaseStep.WAIT_START_APPROVAL
    assert second_state.step == ReleaseStep.WAIT_START_APPROVAL
    assert second_state.active_release is not None
    assert second_state.active_release.release_version == "1.0.0"
    assert audit_log.exists()
    audit_rows = [json.loads(line) for line in audit_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert audit_rows
    assert "actor" in audit_rows[-1]


def test_runtime_executes_slack_approve_tool_call(tmp_path: Path, monkeypatch) -> None:
    memory_db = tmp_path / "crewai_memory.db"
    audit_log = tmp_path / "workflow_audit.jsonl"
    slack_events = tmp_path / "slack_events.jsonl"
    sent: list[tuple[str, str, str]] = []

    def _fake_send_approve(channel_id: str, text: str, approve_label: str) -> dict[str, str]:
        sent.append((channel_id, text, approve_label))
        return {"message_ts": "1700000000.123456"}

    config = RuntimeConfig(
        slack_channel_id="C_DEFAULT",
        slack_bot_token="xoxb-test-token",
        timezone="Europe/Moscow",
        heartbeat_active_minutes=15,
        heartbeat_idle_minutes=240,
        jira_project_keys=["IOS"],
        readiness_owners={"Growth": "owner"},
        memory_db_path=str(memory_db),
        audit_log_path=str(audit_log),
        slack_events_path=str(slack_events),
        agent_pid_path=str(tmp_path / "agent.pid"),
    )
    memory = CrewAIMemory(db_path=str(memory_db))
    slack_gateway = SlackGateway(bot_token="xoxb-test-token", events_path=slack_events)
    monkeypatch.setattr(slack_gateway, "send_approve", _fake_send_approve)
    workflow = ReleaseWorkflow(
        config=config,
        memory=memory,
        slack_gateway=slack_gateway,
        crew_runtime=StartApprovalCrewRuntime(),
    )

    _append_event(
        slack_events,
        {
            "event_id": "ev-1",
            "event_type": "manual_start",
            "channel_id": "C0AGLKF6KHD",
            "text": "5.104.0",
        },
    )
    state = workflow.tick()

    assert len(sent) == 1
    assert sent[0][0] == "C0AGLKF6KHD"
    assert state.active_release is not None
    assert state.active_release.message_ts["start_approval"] == "1700000000.123456"
    assert state.active_release.thread_ts["start_approval"] == "1700000000.123456"
    assert state.active_release.slack_channel_id == "C0AGLKF6KHD"
    assert state.completed_actions == {}


def test_runtime_executes_manual_release_confirmation_flow(tmp_path: Path, monkeypatch) -> None:
    memory_db = tmp_path / "crewai_memory.db"
    audit_log = tmp_path / "workflow_audit.jsonl"
    slack_events = tmp_path / "slack_events.jsonl"
    approvals: list[tuple[str, str, str]] = []
    messages: list[tuple[str, str, str | None]] = []
    updates: list[tuple[str, str, str]] = []

    def _fake_send_approve(channel_id: str, text: str, approve_label: str) -> dict[str, str]:
        approvals.append((channel_id, text, approve_label))
        return {"message_ts": "1700000000.200000"}

    def _fake_send_message(channel_id: str, text: str, thread_ts: str | None = None) -> dict[str, str]:
        messages.append((channel_id, text, thread_ts))
        return {"message_ts": "1700000000.300000"}

    def _fake_update_message(channel_id: str, message_ts: str, text: str) -> dict[str, str]:
        updates.append((channel_id, message_ts, text))
        return {"message_ts": message_ts}

    config = RuntimeConfig(
        slack_channel_id="C_DEFAULT",
        slack_bot_token="xoxb-test-token",
        timezone="Europe/Moscow",
        heartbeat_active_minutes=15,
        heartbeat_idle_minutes=240,
        jira_project_keys=["IOS"],
        readiness_owners={"Growth": "owner"},
        memory_db_path=str(memory_db),
        audit_log_path=str(audit_log),
        slack_events_path=str(slack_events),
        agent_pid_path=str(tmp_path / "agent.pid"),
    )
    memory = CrewAIMemory(db_path=str(memory_db))
    slack_gateway = SlackGateway(bot_token="xoxb-test-token", events_path=slack_events)
    monkeypatch.setattr(slack_gateway, "send_approve", _fake_send_approve)
    monkeypatch.setattr(slack_gateway, "send_message", _fake_send_message)
    monkeypatch.setattr(slack_gateway, "update_message", _fake_update_message)
    workflow = ReleaseWorkflow(
        config=config,
        memory=memory,
        slack_gateway=slack_gateway,
        crew_runtime=ManualReleaseFlowCrewRuntime(),
    )

    _append_event(
        slack_events,
        {
            "event_id": "ev-1",
            "event_type": "manual_start",
            "channel_id": "C0AGLKF6KHD",
            "text": "5.104.0",
        },
    )
    first_state = workflow.tick()

    _append_event(
        slack_events,
        {
            "event_id": "ev-2",
            "event_type": "approval_confirmed",
            "channel_id": "C0AGLKF6KHD",
            "text": "Релиз создан",
            "message_ts": "1700000000.200000",
            "thread_ts": "1700000000.200000",
            "metadata": {
                "trigger_message_text": (
                    "Подтвердите, что релиз 5.104.0 создан вручную в Jira plan, "
                    "и можно переходить к следующему шагу."
                )
            },
        },
    )
    second_state = workflow.tick()

    assert len(approvals) == 1
    assert approvals[0][0] == "C0AGLKF6KHD"
    assert approvals[0][2] == "Релиз создан"
    assert first_state.active_release is not None
    assert first_state.active_release.step == ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION
    assert first_state.active_release.message_ts["manual_release_confirmation"] == "1700000000.200000"
    assert first_state.completed_actions == {}

    assert len(messages) == 1
    assert messages[0][0] == "C0AGLKF6KHD"
    assert messages[0][1] == MANUAL_RELEASE_MESSAGE
    assert messages[0][2] == "1700000000.200000"
    assert len(updates) == 1
    assert updates[0][0] == "C0AGLKF6KHD"
    assert updates[0][1] == "1700000000.200000"
    assert updates[0][2].endswith(":white_check_mark:")
    assert second_state.active_release is not None
    assert second_state.active_release.step == ReleaseStep.JIRA_RELEASE_CREATED
    assert second_state.active_release.message_ts["manual_release_instruction"] == "1700000000.300000"
    assert second_state.completed_actions == {}


def test_runtime_failfast_on_slack_message_without_args_text(tmp_path: Path, monkeypatch) -> None:
    memory_db = tmp_path / "crewai_memory.db"
    audit_log = tmp_path / "workflow_audit.jsonl"
    slack_events = tmp_path / "slack_events.jsonl"
    sent_messages: list[tuple[str, str, str | None]] = []

    def _fake_send_message(channel_id: str, text: str, thread_ts: str | None = None) -> dict[str, str]:
        sent_messages.append((channel_id, text, thread_ts))
        return {"message_ts": "1700000000.400000"}

    config = RuntimeConfig(
        slack_channel_id="C_DEFAULT",
        slack_bot_token="xoxb-test-token",
        timezone="Europe/Moscow",
        heartbeat_active_minutes=15,
        heartbeat_idle_minutes=240,
        jira_project_keys=["IOS"],
        readiness_owners={"Growth": "owner"},
        memory_db_path=str(memory_db),
        audit_log_path=str(audit_log),
        slack_events_path=str(slack_events),
        agent_pid_path=str(tmp_path / "agent.pid"),
    )
    memory = CrewAIMemory(db_path=str(memory_db))
    slack_gateway = SlackGateway(bot_token="xoxb-test-token", events_path=slack_events)
    monkeypatch.setattr(slack_gateway, "send_message", _fake_send_message)
    workflow = ReleaseWorkflow(
        config=config,
        memory=memory,
        slack_gateway=slack_gateway,
        crew_runtime=InvalidSlackMessageCrewRuntime(),
    )

    _append_event(
        slack_events,
        {
            "event_id": "ev-1",
            "event_type": "manual_start",
            "channel_id": "C0AGLKF6KHD",
            "text": "override to idle",
        },
    )

    previous_state = memory.load_state()
    state = workflow.tick()
    persisted_state = memory.load_state()
    audit_rows = [json.loads(line) for line in audit_log.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert sent_messages == []
    assert state.step == previous_state.step
    assert persisted_state.step == previous_state.step
    assert audit_rows[-1]["audit_reason"].startswith("invalid_tool_call:")


def test_runtime_executes_readiness_message_without_deduplication(tmp_path: Path, monkeypatch) -> None:
    memory_db = tmp_path / "crewai_memory.db"
    audit_log = tmp_path / "workflow_audit.jsonl"
    slack_events = tmp_path / "slack_events.jsonl"
    sent_messages: list[tuple[str, str, str | None]] = []

    def _fake_send_message(channel_id: str, text: str, thread_ts: str | None = None) -> dict[str, str]:
        sent_messages.append((channel_id, text, thread_ts))
        return {"message_ts": f"1700000000.4{len(sent_messages)}0000"}

    config = RuntimeConfig(
        slack_channel_id="C_DEFAULT",
        slack_bot_token="xoxb-test-token",
        timezone="Europe/Moscow",
        heartbeat_active_minutes=15,
        heartbeat_idle_minutes=240,
        jira_project_keys=["IOS"],
        readiness_owners={"Growth": "owner"},
        memory_db_path=str(memory_db),
        audit_log_path=str(audit_log),
        slack_events_path=str(slack_events),
        agent_pid_path=str(tmp_path / "agent.pid"),
    )
    memory = CrewAIMemory(db_path=str(memory_db))
    slack_gateway = SlackGateway(bot_token="xoxb-test-token", events_path=slack_events)
    monkeypatch.setattr(slack_gateway, "send_message", _fake_send_message)
    workflow = ReleaseWorkflow(
        config=config,
        memory=memory,
        slack_gateway=slack_gateway,
        crew_runtime=ReadinessAnnouncementCrewRuntime(),
    )

    _append_event(
        slack_events,
        {
            "event_id": "ev-1",
            "event_type": "manual_start",
            "channel_id": "C0AGLKF6KHD",
            "text": "5.104.0",
        },
    )
    first_state = workflow.tick()
    second_state = workflow.tick()

    assert first_state.step == ReleaseStep.WAIT_READINESS_CONFIRMATIONS
    assert second_state.step == ReleaseStep.WAIT_READINESS_CONFIRMATIONS
    assert len(sent_messages) == 2
    assert sent_messages[0][0] == "C0AGLKF6KHD"
    assert sent_messages[0][2] is None
    assert "Статус готовности к срезу:" in sent_messages[0][1]
    assert ":hourglass_flowing_sand: Growth @owner" in sent_messages[0][1]
    assert "Напишите в треде по готовности своей части." in sent_messages[0][1]
    assert sent_messages[1][1] == sent_messages[0][1]


class CountingMemory:
    def __init__(self, state: WorkflowState | None = None) -> None:
        self._state = WorkflowState.from_dict((state or WorkflowState()).to_dict())
        self.load_count = 0

    def load_state(self) -> WorkflowState:
        self.load_count += 1
        return WorkflowState.from_dict(self._state.to_dict())

    def save_state(self, state: WorkflowState, *, reason: str) -> None:  # noqa: ARG002
        self._state = WorkflowState.from_dict(state.to_dict())


class NoopCrewRuntime(_KickoffCompatibleRuntime):
    def decide(self, *, state: WorkflowState, events, config, now=None):  # noqa: ANN001, ARG002
        return CrewDecision(
            next_step=state.step,
            next_state=state,
            audit_reason="noop",
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


def _state_with_release(version: str, previous: str | None = None) -> WorkflowState:
    return WorkflowState(
        active_release=ReleaseContext(
            release_version=version,
            step=ReleaseStep.WAIT_START_APPROVAL,
            slack_channel_id="C_RELEASE",
        ),
        previous_release_version=previous,
        previous_release_completed_at=None,
        processed_event_ids=[f"event-{version}"],
        completed_actions={},
        checkpoints=[],
    )


def test_workflow_loads_state_only_once_per_process(tmp_path: Path) -> None:
    config = _runtime_config(tmp_path)
    memory = CountingMemory()
    slack_gateway = SlackGateway(bot_token="xoxb-test-token", events_path=Path(config.slack_events_path))
    workflow = ReleaseWorkflow(
        config=config,
        memory=memory,
        slack_gateway=slack_gateway,
        crew_runtime=NoopCrewRuntime(),
    )

    workflow.tick()
    workflow.tick()

    assert memory.load_count == 1


def test_crewai_memory_keeps_latest_snapshot_per_release_and_last_two_releases(tmp_path: Path) -> None:
    memory_db = tmp_path / "crewai_memory.db"
    memory = CrewAIMemory(db_path=str(memory_db))

    release_100 = _state_with_release("1.0.0")
    memory.save_state(release_100, reason="first")

    release_100_updated = _state_with_release("1.0.0")
    release_100_updated.processed_event_ids = ["event-1.0.0", "event-1.0.0-updated"]
    memory.save_state(release_100_updated, reason="first-updated")

    release_110 = _state_with_release("1.1.0", previous="1.0.0")
    memory.save_state(release_110, reason="second")

    release_120 = _state_with_release("1.2.0", previous="1.1.0")
    memory.save_state(release_120, reason="third")

    keys = memory.snapshot_keys()
    assert keys == ["1.1.0", "1.2.0"]
    release_120_state = memory.snapshot_state("1.2.0")
    release_110_state = memory.snapshot_state("1.1.0")
    assert release_120_state is not None
    assert release_110_state is not None
    assert release_120_state["active_release"]["release_version"] == "1.2.0"
    assert release_110_state["active_release"]["release_version"] == "1.1.0"


def test_manual_transition_prunes_future_actions_and_replays_start_approval(tmp_path: Path, monkeypatch) -> None:
    class ReenterStartApprovalCrewRuntime(_KickoffCompatibleRuntime):
        def decide(self, *, state: WorkflowState, events, config, now=None):  # noqa: ANN001, ARG002
            if state.step == ReleaseStep.IDLE and any(e.event_type == "manual_start" for e in events):
                return CrewDecision(
                    next_step=ReleaseStep.WAIT_START_APPROVAL,
                    next_state=WorkflowState(
                        active_release=ReleaseContext(
                            release_version="5.104.0",
                            step=ReleaseStep.WAIT_START_APPROVAL,
                            slack_channel_id="C0AGLKF6KHD",
                        ),
                        previous_release_version=state.previous_release_version,
                        previous_release_completed_at=state.previous_release_completed_at,
                        processed_event_ids=[*state.processed_event_ids, "ev-manual-start"],
                        completed_actions=dict(state.completed_actions),
                        checkpoints=list(state.checkpoints),
                    ),
                    audit_reason="manual_start_reentered",
                    tool_calls=[
                        {
                            "tool": "slack_approve",
                            "reason": "request fresh approval after manual transition",
                            "args": {
                                "channel_id": "C0AGLKF6KHD",
                                "text": "Подтвердите старт релиза 5.104.0.",
                                "approve_label": "Подтвердить запуск",
                            },
                        }
                    ],
                )
            return CrewDecision(next_step=state.step, next_state=state, audit_reason="no_change")

    sent: list[tuple[str, str, str]] = []

    def _fake_send_approve(channel_id: str, text: str, approve_label: str) -> dict[str, str]:
        sent.append((channel_id, text, approve_label))
        return {"message_ts": "1772140000.123456"}

    config = _runtime_config(tmp_path)
    memory = CrewAIMemory(db_path=config.memory_db_path)
    stale_state = WorkflowState(
        active_release=None,
        previous_release_version="5.104.0",
        previous_release_completed_at=None,
        processed_event_ids=["ev-old"],
        completed_actions={
            "start-approval:5.104.0": "stale",
            "manual-release-instruction:5.104.0": "stale",
        },
        checkpoints=[],
    )
    memory.save_state(stale_state, reason="seed_stale_actions")
    slack_gateway = SlackGateway(bot_token="xoxb-test-token", events_path=Path(config.slack_events_path))
    monkeypatch.setattr(slack_gateway, "send_approve", _fake_send_approve)
    workflow = ReleaseWorkflow(
        config=config,
        memory=memory,
        slack_gateway=slack_gateway,
        crew_runtime=ReenterStartApprovalCrewRuntime(),
    )

    _append_event(
        Path(config.slack_events_path),
        {
            "event_id": "ev-manual-start",
            "event_type": "manual_start",
            "channel_id": "C0AGLKF6KHD",
            "text": "5.104.0",
            "metadata": {"source": "slash_command", "user_id": "U1"},
        },
    )
    state = workflow.tick()

    assert len(sent) == 1
    assert state.active_release is not None
    assert state.active_release.step == ReleaseStep.WAIT_START_APPROVAL
    assert "manual-release-instruction:5.104.0" not in state.completed_actions
    assert "start-approval:5.104.0" not in state.completed_actions
    cleanup_checkpoints = [cp for cp in state.checkpoints if cp.get("tool") == "state_cleanup"]
    assert cleanup_checkpoints
    assert "start-approval:5.104.0" in cleanup_checkpoints[-1]["removed_completed_actions"]


def test_non_user_transition_does_not_prune_completed_actions(tmp_path: Path) -> None:
    class AutoAdvanceCrewRuntime(_KickoffCompatibleRuntime):
        def decide(self, *, state: WorkflowState, events, config, now=None):  # noqa: ANN001, ARG002
            active = state.active_release
            assert active is not None
            return CrewDecision(
                next_step=ReleaseStep.JIRA_RELEASE_CREATED,
                next_state=WorkflowState(
                    active_release=ReleaseContext(
                        release_version=active.release_version,
                        step=ReleaseStep.JIRA_RELEASE_CREATED,
                        slack_channel_id=active.slack_channel_id,
                        message_ts=dict(active.message_ts),
                        thread_ts=dict(active.thread_ts),
                        readiness_map=dict(active.readiness_map),
                    ),
                    previous_release_version=state.previous_release_version,
                    previous_release_completed_at=state.previous_release_completed_at,
                    processed_event_ids=list(state.processed_event_ids),
                    completed_actions=dict(state.completed_actions),
                    checkpoints=list(state.checkpoints),
                ),
                audit_reason="auto_advance_without_user_event",
            )

    config = _runtime_config(tmp_path)
    memory = CrewAIMemory(db_path=config.memory_db_path)
    seeded_state = WorkflowState(
        active_release=ReleaseContext(
            release_version="5.104.0",
            step=ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION,
            slack_channel_id="C0AGLKF6KHD",
        ),
        previous_release_version="5.104.0",
        previous_release_completed_at=None,
        processed_event_ids=["ev-1"],
        completed_actions={
            "manual-release-instruction:5.104.0": "done",
            "approval-confirmed-update:5.104.0:1772140000.123456": "done",
        },
        checkpoints=[],
    )
    memory.save_state(seeded_state, reason="seed_without_user_event")
    slack_gateway = SlackGateway(bot_token="xoxb-test-token", events_path=Path(config.slack_events_path))
    workflow = ReleaseWorkflow(
        config=config,
        memory=memory,
        slack_gateway=slack_gateway,
        crew_runtime=AutoAdvanceCrewRuntime(),
    )

    state = workflow.tick()

    assert state.step == ReleaseStep.JIRA_RELEASE_CREATED
    assert "manual-release-instruction:5.104.0" in state.completed_actions
    assert "approval-confirmed-update:5.104.0:1772140000.123456" in state.completed_actions
    cleanup_checkpoints = [cp for cp in state.checkpoints if cp.get("tool") == "state_cleanup"]
    assert cleanup_checkpoints == []


