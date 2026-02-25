import json
from pathlib import Path

from src.tools.jira_tools import JiraGateway
from src.tools.slack_tools import SlackGateway


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_slack_outbox_format_is_compatible(tmp_path: Path) -> None:
    slack_outbox = tmp_path / "slack_outbox.jsonl"
    slack_events = tmp_path / "slack_events.jsonl"
    gateway = SlackGateway(outbox_path=slack_outbox, events_path=slack_events)

    gateway.send_message(channel_id="C_RELEASE", text="hello")
    gateway.send_approve(channel_id="C_RELEASE", text="approve", approve_label="OK")
    gateway.update_message(channel_id="C_RELEASE", message_ts="00000001.000001", text="updated")

    rows = _read_jsonl(slack_outbox)
    assert rows[0]["op"] == "slack_message"
    assert rows[1]["op"] == "slack_approve"
    assert rows[2]["op"] == "slack_update"
    assert "message_ts" in rows[0]
    assert "message_ts" in rows[1]
    assert "message_ts" in rows[2]


def test_jira_outbox_format_is_compatible(tmp_path: Path) -> None:
    jira_outbox = tmp_path / "jira_outbox.jsonl"
    gateway = JiraGateway(outbox_path=jira_outbox)
    result = gateway.create_crossspace_release(version="3.4.5", project_keys=["IOS", "CORE"])

    rows = _read_jsonl(jira_outbox)
    assert rows[0]["op"] == "create_crossspace_release"
    assert rows[0]["version"] == "3.4.5"
    assert rows[0]["project_keys"] == ["IOS", "CORE"]
    assert result["status"] == "created"
