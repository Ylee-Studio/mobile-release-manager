import json
from types import SimpleNamespace
from urllib.parse import urlencode

from src.slack_ingress import SlackRequestHandler, _extract_bot_user_id, _should_ignore_message_event
from src.workflow_state import ReleaseContext, ReleaseStep, WorkflowState


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


def test_handle_events_persists_message_with_mention_without_conversion() -> None:
    written: list[dict] = []
    replies: list[tuple[int, dict]] = []
    triggered = {"called": 0}

    def _write(**kwargs):  # noqa: ANN003
        written.append(kwargs)

    handler = SimpleNamespace(
        writer=SimpleNamespace(write=_write),
        _trigger_event_processing=lambda: triggered.__setitem__("called", triggered["called"] + 1),
        _send_json=lambda code, payload: replies.append((code, payload)),
    )
    raw = {
        "type": "event_callback",
        "event_id": "Ev-message-1",
        "authorizations": [{"user_id": "U_BOT", "is_bot": True}],
        "event": {
            "type": "message",
            "user": "U_HUMAN",
            "channel": "C_RELEASE",
            "text": "approve <@U_BOT>",
            "ts": "171.111",
        },
    }

    SlackRequestHandler._handle_events(handler, json.dumps(raw, ensure_ascii=True).encode("utf-8"))  # type: ignore[arg-type]

    assert replies[-1] == (200, {"ok": True})
    assert triggered["called"] == 1
    assert written == [
        {
            "event_id": "Ev-message-1",
            "event_type": "message",
            "channel_id": "C_RELEASE",
            "text": "approve <@U_BOT>",
            "thread_ts": "171.111",
            "message_ts": "171.111",
            "metadata": {
                "source": "events_api",
                "user_id": "U_HUMAN",
            },
        }
    ]


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
            "text": "Подтвердите старт релизного трейна 5.105.0.",
        },
        "container": {},
        "user": {"id": "U123"},
    }
    body = f"payload={json.dumps(payload, ensure_ascii=True)}".encode("utf-8")

    SlackRequestHandler._handle_interactivity(handler, body)  # type: ignore[arg-type]

    assert written[0]["event_type"] == "approval_rejected"
    assert written[0]["metadata"]["action_id"] == "release_reject"
    assert replies[-1] == (200, {"ok": True})


def test_release_status_returns_public_workflow_state_json() -> None:
    written: list[dict] = []
    replies: list[tuple[int, dict]] = []

    state = WorkflowState(
        active_release=ReleaseContext(
            release_version="5.105.0",
            step=ReleaseStep.WAIT_START_APPROVAL,
            slack_channel_id="C_RELEASE",
        ),
        previous_release_version="5.104.0",
        previous_release_completed_at="2026-01-01T00:00:00+00:00",
        flow_execution_id="flow-123",
        flow_paused_at="2026-01-01T01:00:00+00:00",
        pause_reason="manual_gate",
        checkpoints=[{"name": "internal_only"}],
    )

    class _Handler:
        get_workflow_state = staticmethod(lambda: state)
        _build_public_status_payload = SlackRequestHandler._build_public_status_payload

        def __init__(self) -> None:
            self.writer = SimpleNamespace(write=lambda **kwargs: written.append(kwargs))
            self.cfg = SimpleNamespace(announce_channel_id="C_RELEASE")

        def _send_json(self, code: int, payload: dict) -> None:
            replies.append((code, payload))

    handler = _Handler()
    body = urlencode(
        {
            "command": "/release-status",
            "channel_id": "C_RELEASE",
            "user_id": "U123",
        }
    ).encode("utf-8")

    SlackRequestHandler._handle_commands(handler, body)  # type: ignore[arg-type]

    assert written == []
    assert replies
    status, payload = replies[-1]
    assert status == 200
    assert payload["response_type"] == "ephemeral"
    parsed = json.loads(payload["text"])
    assert parsed == {
        "active_release": {
            "release_version": "5.105.0",
            "step": "WAIT_START_APPROVAL",
            "updated_at": state.active_release.updated_at,  # type: ignore[union-attr]
            "slack_channel_id": "C_RELEASE",
            "message_ts": {},
            "thread_ts": {},
            "readiness_map": {},
            "github_action_run_id": None,
            "github_action_status": None,
            "github_action_conclusion": None,
            "github_action_last_polled_at": None,
        },
        "previous_release_version": "5.104.0",
        "previous_release_completed_at": "2026-01-01T00:00:00+00:00",
    }
