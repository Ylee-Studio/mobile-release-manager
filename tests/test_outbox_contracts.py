import io
import json
from urllib.error import HTTPError, URLError

import pytest

from src.tools.jira_tools import JiraGateway
from src.tools.slack_tools import SlackGateway


class _FakeResponse:
    def __init__(self, body: dict | list):
        self._raw = json.dumps(body, ensure_ascii=True).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_slack_gateway_calls_real_api(monkeypatch, tmp_path) -> None:
    captured: list[tuple[str, dict]] = []

    def _fake_urlopen(request, timeout):  # noqa: ANN001
        payload = json.loads(request.data.decode("utf-8"))
        captured.append((request.full_url, payload))
        return _FakeResponse({"ok": True, "ts": "1700000000.123456"})

    monkeypatch.setattr("src.tools.slack_tools.urlopen", _fake_urlopen)
    gateway = SlackGateway(bot_token="xoxb-test-token", events_path=tmp_path / "slack_events.jsonl")

    sent = gateway.send_message(channel_id="C_RELEASE", text="hello", thread_ts="123.456")
    approved = gateway.send_approve(channel_id="C_RELEASE", text="please approve", approve_label="Approve")
    updated = gateway.update_message(channel_id="C_RELEASE", message_ts="123.456", text="updated")

    assert sent["message_ts"] == "1700000000.123456"
    assert approved["message_ts"] == "1700000000.123456"
    assert updated["message_ts"] == "1700000000.123456"
    assert [url for url, _payload in captured] == [
        "https://slack.com/api/chat.postMessage",
        "https://slack.com/api/chat.postMessage",
        "https://slack.com/api/chat.update",
    ]
    approve_payload = captured[1][1]
    assert approve_payload["blocks"][1]["elements"][0]["action_id"] == "release_approve"


def test_slack_gateway_surfaces_api_errors(monkeypatch, tmp_path) -> None:
    def _http_error(_request, timeout):  # noqa: ANN001, ARG001
        raise HTTPError(
            url="https://slack.com/api/chat.postMessage",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=io.BytesIO(b'{"ok":false,"error":"invalid_auth"}'),
        )

    monkeypatch.setattr("src.tools.slack_tools.urlopen", _http_error)
    gateway = SlackGateway(bot_token="xoxb-test-token", events_path=tmp_path / "slack_events.jsonl")

    with pytest.raises(RuntimeError, match="Slack API chat.postMessage failed with HTTP 400"):
        gateway.send_message(channel_id="C_RELEASE", text="hello")


def test_jira_gateway_calls_real_api(monkeypatch) -> None:
    requests_seen: list[tuple[str, str, dict | None]] = []

    def _fake_urlopen(request, timeout):  # noqa: ANN001
        payload = json.loads(request.data.decode("utf-8")) if request.data else None
        requests_seen.append((request.method, request.full_url, payload))
        if request.full_url.endswith("/rest/api/3/project/IOS/versions"):
            return _FakeResponse([])
        if request.full_url.endswith("/rest/api/3/project/IOS"):
            return _FakeResponse({"id": "10001"})
        if request.full_url.endswith("/rest/api/3/version"):
            return _FakeResponse({"id": "20001"})
        raise AssertionError(f"Unexpected URL {request.full_url}")

    monkeypatch.setattr("src.tools.jira_tools.urlopen", _fake_urlopen)
    gateway = JiraGateway(
        base_url="https://jira.example.com",
        email="bot@example.com",
        api_token="token",
    )
    result = gateway.create_crossspace_release(version="3.4.5", project_keys=["IOS"])

    assert result["status"] == "created"
    assert requests_seen[0][0] == "GET"
    assert requests_seen[2][0] == "POST"
    assert requests_seen[2][2] == {"name": "3.4.5", "projectId": 10001}


def test_jira_gateway_surfaces_network_errors(monkeypatch) -> None:
    def _url_error(_request, timeout):  # noqa: ANN001, ARG001
        raise URLError("connection refused")

    monkeypatch.setattr("src.tools.jira_tools.urlopen", _url_error)
    gateway = JiraGateway(
        base_url="https://jira.example.com",
        email="bot@example.com",
        api_token="token",
    )

    with pytest.raises(RuntimeError, match="Jira API GET /rest/api/3/project/IOS/versions failed"):
        gateway.create_crossspace_release(version="3.4.5", project_keys=["IOS"])
