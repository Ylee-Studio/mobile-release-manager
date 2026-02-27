import pytest
from slack_sdk.errors import SlackApiError

from src.tools.slack_tools import SlackGateway


class _FakeSlackResponse(dict):
    def __init__(self, *, status_code: int, payload: dict[str, object]):
        super().__init__(payload)
        self.status_code = status_code


def test_slack_gateway_calls_real_api(tmp_path) -> None:
    captured: list[tuple[str, dict[str, object]]] = []

    class _FakeClient:
        def chat_postMessage(self, **kwargs):  # noqa: ANN003
            captured.append(("chat.postMessage", kwargs))
            return {"ok": True, "ts": "1700000000.123456"}

        def chat_update(self, **kwargs):  # noqa: ANN003
            captured.append(("chat.update", kwargs))
            return {"ok": True, "ts": "1700000000.123456"}

    gateway = SlackGateway(bot_token="xoxb-test-token", events_path=tmp_path / "slack_events.jsonl")
    gateway.client = _FakeClient()

    sent = gateway.send_message(channel_id="C_RELEASE", text="hello", thread_ts="123.456")
    approved = gateway.send_approve(channel_id="C_RELEASE", text="please approve", approve_label="Approve")
    updated = gateway.update_message(channel_id="C_RELEASE", message_ts="123.456", text="updated")

    assert sent["message_ts"] == "1700000000.123456"
    assert approved["message_ts"] == "1700000000.123456"
    assert updated["message_ts"] == "1700000000.123456"
    assert [method for method, _payload in captured] == [
        "chat.postMessage",
        "chat.postMessage",
        "chat.update",
    ]
    approve_payload = captured[1][1]
    assert approve_payload["blocks"][1]["elements"][0]["action_id"] == "release_approve"


def test_slack_gateway_surfaces_api_errors(tmp_path) -> None:
    class _FailingClient:
        def chat_postMessage(self, **kwargs):  # noqa: ANN003, ARG002
            response = _FakeSlackResponse(
                status_code=400,
                payload={"ok": False, "error": "invalid_auth"},
            )
            raise SlackApiError(message="invalid_auth", response=response)

    gateway = SlackGateway(bot_token="xoxb-test-token", events_path=tmp_path / "slack_events.jsonl")
    gateway.client = _FailingClient()

    with pytest.raises(RuntimeError, match="Slack API chat.postMessage failed with HTTP 400"):
        gateway.send_message(channel_id="C_RELEASE", text="hello")
