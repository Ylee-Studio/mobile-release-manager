import json
from pathlib import Path

from src.crew_memory import SQLiteMemory
from src.crew_runtime import CrewDecision
from src.release_workflow import MANUAL_RELEASE_MESSAGE, ReleaseWorkflow, RuntimeConfig
from src.tools.slack_tools import SlackGateway
from src.workflow_state import ReleaseContext, ReleaseStep, WorkflowState


def _append_event(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


class FakeCrewRuntime:
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
                tool_calls=[{"tool": "slack_approve", "reason": "start confirmation"}],
            )

        return CrewDecision(
            next_step=state.step,
            next_state=state,
            audit_reason="no_change",
        )


class RecordingCrewRuntime:
    def __init__(self) -> None:
        self.seen_event_ids: list[list[str]] = []

    def decide(self, *, state: WorkflowState, events, config, now=None):  # noqa: ANN001
        self.seen_event_ids.append([event.event_id for event in events])
        return CrewDecision(
            next_step=state.step,
            next_state=state,
            audit_reason="recorded",
        )


class StartApprovalCrewRuntime:
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
                tool_calls=[{"tool": "functions.slack_approve", "reason": "request start approval"}],
            )
        return CrewDecision(next_step=state.step, next_state=state, audit_reason="no_change")


class ManualReleaseFlowCrewRuntime:
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
                tool_calls=[{"tool": "functions.slack_approve", "reason": "manual Jira release confirmation"}],
            )
        if state.step == ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION and any(
            e.event_type == "approval_confirmed" for e in events
        ):
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
                    {"tool": "functions.slack_update", "reason": "acknowledge confirmation"},
                    {"tool": "functions.slack_message", "reason": "send manual release instructions"},
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
    memory = SQLiteMemory(db_path=str(memory_db))
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
        memory=SQLiteMemory(db_path=str(memory_db)),
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


def test_fifo_events_consumed_one_per_tick(tmp_path: Path) -> None:
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
    memory = SQLiteMemory(db_path=str(memory_db))
    slack_gateway = SlackGateway(bot_token="xoxb-test-token", events_path=slack_events)
    runtime = RecordingCrewRuntime()
    workflow = ReleaseWorkflow(config=config, memory=memory, slack_gateway=slack_gateway, crew_runtime=runtime)

    _append_event(
        slack_events,
        {
            "event_id": "ev-1",
            "event_type": "manual_start",
            "channel_id": "C_RELEASE",
            "text": "start release",
        },
    )
    _append_event(
        slack_events,
        {
            "event_id": "ev-2",
            "event_type": "approval_confirmed",
            "channel_id": "C_RELEASE",
            "text": "approve",
        },
    )

    workflow.tick()
    queue_after_first_tick = slack_events.read_text(encoding="utf-8").splitlines()
    workflow.tick()
    queue_after_second_tick = slack_events.read_text(encoding="utf-8").splitlines()

    assert runtime.seen_event_ids == [["ev-1"], ["ev-2"]]
    assert len(queue_after_first_tick) == 1
    assert queue_after_second_tick == []


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
    memory = SQLiteMemory(db_path=str(memory_db))
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
    assert "start-approval:5.104.0" in state.completed_actions


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
    memory = SQLiteMemory(db_path=str(memory_db))
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
    assert "manual-release-confirmation-request:5.104.0" in first_state.completed_actions

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
    assert "manual-release-instruction:5.104.0" in second_state.completed_actions
    assert "approval-confirmed-update:5.104.0:1700000000.200000" in second_state.completed_actions


