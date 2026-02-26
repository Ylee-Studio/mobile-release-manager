import json
from pathlib import Path

from src.tools.slack_tools import SlackGateway


def _append_raw(path: Path, payload: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")


def _append_event(path: Path, event_id: str) -> None:
    _append_raw(
        path,
        json.dumps(
            {
                "event_id": event_id,
                "event_type": "message",
                "channel_id": "C_RELEASE",
                "text": f"text-{event_id}",
            },
            ensure_ascii=True,
        ),
    )


def test_poll_events_empty_queue(tmp_path: Path) -> None:
    gateway = SlackGateway(
        bot_token="xoxb-test-token",
        events_path=tmp_path / "slack_events.jsonl",
    )

    assert gateway.poll_events() == []


def test_poll_events_consumes_single_event(tmp_path: Path) -> None:
    events_path = tmp_path / "slack_events.jsonl"
    _append_event(events_path, "ev-1")
    gateway = SlackGateway(
        bot_token="xoxb-test-token",
        events_path=events_path,
    )

    events = gateway.poll_events()

    assert len(events) == 1
    assert events[0].event_id == "ev-1"
    assert events_path.read_text(encoding="utf-8") == ""


def test_poll_events_preserves_fifo_order(tmp_path: Path) -> None:
    events_path = tmp_path / "slack_events.jsonl"
    _append_event(events_path, "ev-1")
    _append_event(events_path, "ev-2")
    gateway = SlackGateway(
        bot_token="xoxb-test-token",
        events_path=events_path,
    )

    first = gateway.poll_events()
    second = gateway.poll_events()
    third = gateway.poll_events()

    assert [event.event_id for event in first] == ["ev-1"]
    assert [event.event_id for event in second] == ["ev-2"]
    assert third == []


def test_poll_events_drops_malformed_head(tmp_path: Path) -> None:
    events_path = tmp_path / "slack_events.jsonl"
    _append_raw(events_path, "{bad-json")
    _append_event(events_path, "ev-2")
    gateway = SlackGateway(
        bot_token="xoxb-test-token",
        events_path=events_path,
    )

    first = gateway.poll_events()
    second = gateway.poll_events()

    assert first == []
    assert [event.event_id for event in second] == ["ev-2"]


def test_pending_events_count_ignores_empty_lines(tmp_path: Path) -> None:
    events_path = tmp_path / "slack_events.jsonl"
    _append_raw(events_path, "")
    _append_event(events_path, "ev-1")
    _append_raw(events_path, "   ")
    _append_event(events_path, "ev-2")
    gateway = SlackGateway(
        bot_token="xoxb-test-token",
        events_path=events_path,
    )

    assert gateway.pending_events_count() == 2


def test_add_reaction_calls_reactions_add(tmp_path: Path, monkeypatch) -> None:
    gateway = SlackGateway(
        bot_token="xoxb-test-token",
        events_path=tmp_path / "slack_events.jsonl",
    )
    captured: list[tuple[str, dict[str, str]]] = []

    def _fake_request_json(*, method_name: str, payload: dict[str, str]) -> dict[str, bool]:
        captured.append((method_name, payload))
        return {"ok": True}

    monkeypatch.setattr(gateway, "_request_json", _fake_request_json)

    added = gateway.add_reaction("C_RELEASE", "1700000000.123456", "eyes")

    assert added is True
    assert captured == [
        (
            "reactions.add",
            {
                "channel": "C_RELEASE",
                "timestamp": "1700000000.123456",
                "name": "eyes",
            },
        )
    ]


def test_add_reaction_treats_already_reacted_as_success(tmp_path: Path, monkeypatch) -> None:
    gateway = SlackGateway(
        bot_token="xoxb-test-token",
        events_path=tmp_path / "slack_events.jsonl",
    )

    def _fake_request_json(*, method_name: str, payload: dict[str, str]) -> dict[str, bool]:  # noqa: ARG001
        raise RuntimeError("Slack API reactions.add failed: already_reacted")

    monkeypatch.setattr(gateway, "_request_json", _fake_request_json)

    added = gateway.add_reaction("C_RELEASE", "1700000000.123456", "eyes")

    assert added is False


def test_add_reaction_treats_missing_scope_as_non_fatal(tmp_path: Path, monkeypatch) -> None:
    gateway = SlackGateway(
        bot_token="xoxb-test-token",
        events_path=tmp_path / "slack_events.jsonl",
    )

    def _fake_request_json(*, method_name: str, payload: dict[str, str]) -> dict[str, bool]:  # noqa: ARG001
        raise RuntimeError("Slack API reactions.add failed: missing_scope")

    monkeypatch.setattr(gateway, "_request_json", _fake_request_json)

    added = gateway.add_reaction("C_RELEASE", "1700000000.123456", "eyes")

    assert added is False
