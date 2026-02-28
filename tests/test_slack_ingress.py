import json
from types import SimpleNamespace

from src.slack_ingress import SlackRequestHandler, _extract_bot_user_id, _should_ignore_message_event


def test_extract_bot_user_id_from_authorizations() -> None:
    raw_event = {
        "authorizations": [
            {"enterprise_id": None, "team_id": "T1", "user_id": "U_BOT", "is_bot": True}
        ]
    }

    assert _extract_bot_user_id(raw_event) == "U_BOT"


def test_ignore_message_from_same_bot_user() -> None:
    raw_event = {
        "authorizations": [
            {"enterprise_id": None, "team_id": "T1", "user_id": "U_BOT", "is_bot": True}
        ]
    }
    message_event = {"type": "message", "user": "U_BOT", "text": "self echo"}

    assert _should_ignore_message_event(raw_event, message_event) is True


def test_allow_message_from_human_user() -> None:
    raw_event = {
        "authorizations": [
            {"enterprise_id": None, "team_id": "T1", "user_id": "U_BOT", "is_bot": True}
        ]
    }
    message_event = {"type": "message", "user": "U_HUMAN", "text": "please <@U_BOT> release"}

    assert _should_ignore_message_event(raw_event, message_event) is False


def test_ignore_human_message_without_bot_mention() -> None:
    raw_event = {
        "authorizations": [
            {"enterprise_id": None, "team_id": "T1", "user_id": "U_BOT", "is_bot": True}
        ]
    }
    message_event = {"type": "message", "user": "U_HUMAN", "text": "release please"}

    assert _should_ignore_message_event(raw_event, message_event) is True


def test_allow_human_message_with_mention_alias_form() -> None:
    raw_event = {
        "authorizations": [
            {"enterprise_id": None, "team_id": "T1", "user_id": "U_BOT", "is_bot": True}
        ]
    }
    message_event = {"type": "message", "user": "U_HUMAN", "text": "please <@U_BOT|bot> check"}

    assert _should_ignore_message_event(raw_event, message_event) is False


def test_interactivity_maps_release_reject_to_approval_rejected() -> None:
    written: list[dict] = []
    replies: list[tuple[int, dict]] = []

    def _write(**kwargs):  # noqa: ANN003
        written.append(kwargs)

    handler = SimpleNamespace(
        writer=SimpleNamespace(write=_write),
        _trigger_event_processing=lambda: None,
        _send_json=lambda code, payload: replies.append((code, payload)),
    )
    payload = {
        "type": "block_actions",
        "trigger_id": "trigger-1",
        "channel": {"id": "C_RELEASE"},
        "actions": [
            {
                "action_id": "release_reject",
                "value": "Отклонить",
                "block_id": "b1",
            }
        ],
        "message": {
            "ts": "171.222",
            "text": "Подтвердите старт релизного трейна 5.105.0",
        },
        "container": {},
        "user": {"id": "U123"},
    }
    body = f"payload={json.dumps(payload, ensure_ascii=True)}".encode("utf-8")

    SlackRequestHandler._handle_interactivity(handler, body)  # type: ignore[arg-type]

    assert written[0]["event_type"] == "approval_rejected"
    assert written[0]["metadata"]["action_id"] == "release_reject"
    assert replies[-1] == (200, {"ok": True})
