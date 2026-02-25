import json
from pathlib import Path

from src.crew_memory import SQLiteMemory
from src.crew_runtime import CrewDecision
from src.release_workflow import ReleaseWorkflow, RuntimeConfig
from src.tools.jira_tools import JiraGateway
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


def test_memory_state_persists_across_restarts(tmp_path: Path) -> None:
    memory_db = tmp_path / "crewai_memory.db"
    audit_log = tmp_path / "workflow_audit.jsonl"
    slack_outbox = tmp_path / "slack_outbox.jsonl"
    slack_events = tmp_path / "slack_events.jsonl"
    jira_outbox = tmp_path / "jira_outbox.jsonl"

    config = RuntimeConfig(
        slack_channel_id="C_RELEASE",
        timezone="Europe/Moscow",
        heartbeat_active_minutes=15,
        heartbeat_idle_minutes=240,
        jira_project_keys=["IOS"],
        readiness_owners={"Growth": "owner"},
        memory_db_path=str(memory_db),
        audit_log_path=str(audit_log),
        slack_outbox_path=str(slack_outbox),
        slack_events_path=str(slack_events),
        jira_outbox_path=str(jira_outbox),
        agent_pid_path=str(tmp_path / "agent.pid"),
    )
    memory = SQLiteMemory(db_path=str(memory_db))
    slack_gateway = SlackGateway(outbox_path=slack_outbox, events_path=slack_events)
    _jira_gateway = JiraGateway(outbox_path=jira_outbox)

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
