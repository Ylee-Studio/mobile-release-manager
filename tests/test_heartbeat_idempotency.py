import json
from pathlib import Path

from src.release_workflow import ReleaseWorkflow, RuntimeConfig
from src.state_store import StateStore
from src.tools.jira_tools import JiraGateway
from src.tools.slack_tools import SlackGateway
from src.workflow_state import ReleaseStep


def _append_event(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def test_manual_start_event_is_idempotent(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
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
        release_state_path=str(state_path),
        slack_outbox_path=str(slack_outbox),
        slack_events_path=str(slack_events),
        jira_outbox_path=str(jira_outbox),
    )
    workflow = ReleaseWorkflow(
        config=config,
        state_store=StateStore(state_path),
        slack_gateway=SlackGateway(outbox_path=slack_outbox, events_path=slack_events),
        jira_gateway=JiraGateway(outbox_path=jira_outbox),
    )

    _append_event(
        slack_events,
        {
            "event_id": "ev-1",
            "event_type": "manual_start",
            "channel_id": "C_RELEASE",
            "text": "start release",
        },
    )
    workflow.tick()
    first_outbox_size = len(slack_outbox.read_text(encoding="utf-8").splitlines())
    first_state = StateStore(state_path).load()

    workflow.tick()
    second_outbox_size = len(slack_outbox.read_text(encoding="utf-8").splitlines())
    second_state = StateStore(state_path).load()

    assert first_outbox_size == 1
    assert second_outbox_size == 1
    assert first_state.step == ReleaseStep.WAIT_START_APPROVAL
    assert second_state.step == ReleaseStep.WAIT_START_APPROVAL
