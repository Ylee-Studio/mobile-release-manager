from __future__ import annotations

from types import SimpleNamespace

from src.runtime_engine import RuntimeCoordinator, RuntimeDecision
from src.policies import PolicyConfig
from src.tools.slack_tools import SlackEvent, SlackGateway
from src.workflow_state import ReleaseContext, ReleaseStep, WorkflowState


def _policy() -> PolicyConfig:
    return PolicyConfig.from_dict({})


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        slack_channel_id="C_RELEASE",
        timezone="Europe/Moscow",
        jira_project_keys=["IOS"],
        readiness_owners={"Growth": "owner"},
    )


def _build_coordinator(monkeypatch, tmp_path) -> RuntimeCoordinator:
    _ = monkeypatch
    gateway = SlackGateway(bot_token="xoxb-test-token", events_path=tmp_path / "events.jsonl")
    return RuntimeCoordinator(
        policy=_policy(),
        slack_gateway=gateway,
        memory_db_path=str(tmp_path / "memory.db"),
    )


def test_paused_flow_without_confirmation_event_does_not_execute(monkeypatch, tmp_path) -> None:
    coordinator = _build_coordinator(monkeypatch, tmp_path)
    flow_calls = {"kickoff": 0}

    class _FlowStub:
        def kickoff(self, *, inputs):  # noqa: ANN001
            flow_calls["kickoff"] += 1
            return RuntimeDecision(next_step=ReleaseStep.IDLE, next_state=WorkflowState())

    coordinator.flow = _FlowStub()
    state = WorkflowState(
        active_release=ReleaseContext(release_version="5.104.0", step=ReleaseStep.WAIT_MEETING_CONFIRMATION),
        flow_execution_id="flow-1",
        flow_paused_at="2026-01-01T00:00:00+00:00",
        pause_reason="awaiting_confirmation:WAIT_MEETING_CONFIRMATION",
    )
    decision = coordinator.kickoff(state=state, events=[], config=_config())

    assert flow_calls["kickoff"] == 1
    assert decision.flow_lifecycle == "completed"
    assert decision.audit_reason == "agent_decision"


def test_paused_flow_with_confirmation_event_runs_kickoff(monkeypatch, tmp_path) -> None:
    coordinator = _build_coordinator(monkeypatch, tmp_path)
    flow_calls = {"kickoff": 0}

    class _FlowStub:
        def kickoff(self, *, inputs):  # noqa: ANN001
            _ = inputs
            flow_calls["kickoff"] += 1
            return RuntimeDecision(
                next_step=ReleaseStep.WAIT_READINESS_CONFIRMATIONS,
                next_state=WorkflowState(
                    active_release=ReleaseContext(
                        release_version="5.104.0",
                        step=ReleaseStep.WAIT_READINESS_CONFIRMATIONS,
                    )
                ),
                flow_lifecycle="running",
            )

    coordinator.flow = _FlowStub()
    state = WorkflowState(
        active_release=ReleaseContext(release_version="5.104.0", step=ReleaseStep.WAIT_MEETING_CONFIRMATION),
        flow_execution_id="flow-1",
        flow_paused_at="2026-01-01T00:00:00+00:00",
        pause_reason="awaiting_confirmation:WAIT_MEETING_CONFIRMATION",
    )
    decision = coordinator.kickoff(
        state=state,
        events=[
            SlackEvent(
                event_id="ev-1",
                event_type="approval_confirmed",
                channel_id="C_RELEASE",
            )
        ],
        config=_config(),
    )

    assert flow_calls["kickoff"] == 1
    assert decision.next_step == ReleaseStep.WAIT_READINESS_CONFIRMATIONS
    assert decision.flow_lifecycle == "running"


def test_message_with_bot_mention_can_confirm_approval_via_flow(monkeypatch, tmp_path) -> None:
    coordinator = _build_coordinator(monkeypatch, tmp_path)
    flow_calls = {"kickoff": 0}

    class _FlowStub:
        def kickoff(self, *, inputs):  # noqa: ANN001
            flow_calls["kickoff"] += 1
            events = inputs["events"]
            assert len(events) == 1
            assert events[0]["event_type"] == "message"
            assert "<@U_BOT>" in events[0]["text"]
            return RuntimeDecision(
                next_step=ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION,
                next_state=WorkflowState(
                    active_release=ReleaseContext(
                        release_version="5.104.0",
                        step=ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION,
                    )
                ),
                audit_reason="message_approval_intent",
                flow_lifecycle="running",
            )

    coordinator.flow = _FlowStub()
    state = WorkflowState(
        active_release=ReleaseContext(release_version="5.104.0", step=ReleaseStep.WAIT_START_APPROVAL),
        flow_execution_id="flow-1",
        flow_paused_at="2026-01-01T00:00:00+00:00",
        pause_reason="awaiting_confirmation:WAIT_START_APPROVAL",
    )
    decision = coordinator.kickoff(
        state=state,
        events=[
            SlackEvent(
                event_id="ev-message-1",
                event_type="message",
                channel_id="C_RELEASE",
                text="<@U_BOT> подтверждаю",
                thread_ts="171.100",
                message_ts="171.101",
                metadata={"source": "events_api", "user_id": "U123"},
            )
        ],
        config=_config(),
    )

    assert flow_calls["kickoff"] == 1
    assert decision.next_step == ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION
    assert decision.audit_reason == "message_approval_intent"


def test_neutral_message_with_bot_mention_can_result_in_noop(monkeypatch, tmp_path) -> None:
    coordinator = _build_coordinator(monkeypatch, tmp_path)
    flow_calls = {"kickoff": 0}

    class _FlowStub:
        def kickoff(self, *, inputs):  # noqa: ANN001
            flow_calls["kickoff"] += 1
            events = inputs["events"]
            assert len(events) == 1
            assert events[0]["event_type"] == "message"
            assert events[0]["text"] == "<@U_BOT> посмотрел"
            return RuntimeDecision(
                next_step=ReleaseStep.WAIT_START_APPROVAL,
                next_state=WorkflowState(
                    active_release=ReleaseContext(
                        release_version="5.104.0",
                        step=ReleaseStep.WAIT_START_APPROVAL,
                    )
                ),
                audit_reason="message_intent_ambiguous_noop",
                flow_lifecycle="running",
            )

    coordinator.flow = _FlowStub()
    state = WorkflowState(
        active_release=ReleaseContext(release_version="5.104.0", step=ReleaseStep.WAIT_START_APPROVAL),
        flow_execution_id="flow-1",
        flow_paused_at="2026-01-01T00:00:00+00:00",
        pause_reason="awaiting_confirmation:WAIT_START_APPROVAL",
    )
    decision = coordinator.kickoff(
        state=state,
        events=[
            SlackEvent(
                event_id="ev-message-2",
                event_type="message",
                channel_id="C_RELEASE",
                text="<@U_BOT> посмотрел",
                thread_ts="171.100",
                message_ts="171.102",
                metadata={"source": "events_api", "user_id": "U123"},
            )
        ],
        config=_config(),
    )

    assert flow_calls["kickoff"] == 1
    assert decision.next_step == ReleaseStep.WAIT_START_APPROVAL
    assert decision.audit_reason == "message_intent_ambiguous_noop"


def test_resume_from_pending_uses_from_pending_flow(monkeypatch, tmp_path) -> None:
    coordinator = _build_coordinator(monkeypatch, tmp_path)
    called = {"from_pending": 0, "resume": 0}

    class _RestoredFlow:
        def resume(self, feedback=""):  # noqa: ANN001
            _ = feedback
            called["resume"] += 1
            return "ok"

    def _from_pending(cls, flow_id, persistence=None, **kwargs):  # noqa: ANN001, ANN003
        _ = (cls, flow_id, persistence, kwargs)
        called["from_pending"] += 1
        return _RestoredFlow()

    monkeypatch.setattr(type(coordinator.flow), "from_pending", classmethod(_from_pending), raising=False)
    state = WorkflowState(
        active_release=ReleaseContext(release_version="5.104.0", step=ReleaseStep.WAIT_MEETING_CONFIRMATION),
        flow_execution_id="flow-1",
        flow_paused_at="2026-01-01T00:00:00+00:00",
        pause_reason="awaiting_confirmation:WAIT_MEETING_CONFIRMATION",
    )

    decision = coordinator.resume_from_pending(flow_id="flow-1", feedback="Подтвердить", state=state)

    assert called["from_pending"] == 1
    assert called["resume"] == 1
    assert decision.flow_lifecycle == "running"
    assert decision.audit_reason == "resume_dispatched"


def test_resolve_tool_calls_prefers_payload_list_over_partial_native(monkeypatch, tmp_path) -> None:
    coordinator = _build_coordinator(monkeypatch, tmp_path)
    payload = {
        "tool_calls": [
            {
                "tool": "slack_update",
                "reason": "update trigger message",
                "args": {
                    "channel_id": "C0AGLKF6KHD",
                    "message_ts": "1772178090.522069",
                    "text": "Подтвердите создание релиза 5.104.0 в JIRA. <https://instories.atlassian.net/jira/plans/1/scenarios/1/releases|JIRA>",
                },
            },
            {
                "tool": "slack_approve",
                "reason": "request meeting confirmation",
                "args": {
                    "channel_id": "C0AGLKF6KHD",
                    "text": "Подтвердите, что встреча фиксации <https://instories.atlassian.net/issues/?jql=fixVersion%3D5.104.0|релиза> уже прошла.",
                    "approve_label": "Подтвердить",
                },
            },
        ]
    }
    native_tool_calls = [
        {
            "tool": "slack_update",
            "reason": "native partial",
            "args": {
                "channel_id": "C0AGLKF6KHD",
                "message_ts": "1772178090.522069",
                "text": "Подтвердите создание релиза 5.104.0 в JIRA. <https://instories.atlassian.net/jira/plans/1/scenarios/1/releases|JIRA>",
            },
        }
    ]

    resolved = coordinator._resolve_tool_calls(  # noqa: SLF001 - explicit regression test for internal policy
        payload=payload,
        native_tool_calls=native_tool_calls,
    )

    assert [call["tool"] for call in resolved] == ["slack_update", "slack_approve"]


def test_resolve_tool_calls_fallbacks_to_native_when_payload_invalid(monkeypatch, tmp_path) -> None:
    coordinator = _build_coordinator(monkeypatch, tmp_path)
    payload = {
        "tool_calls": [
            {
                "tool": "slack_update",
                "reason": "invalid payload call",
                "args": {
                    "channel_id": "C0AGLKF6KHD",
                    "message_ts": "1772178090.522069",
                    # Missing required `text` to reproduce schema mismatch.
                },
            }
        ]
    }
    native_tool_calls = [
        {
            "tool": "slack_update",
            "reason": "native fallback",
            "args": {
                "channel_id": "C0AGLKF6KHD",
                "message_ts": "1772178090.522069",
                "text": "Подтверждено :white_check_mark:",
            },
        }
    ]

    resolved = coordinator._resolve_tool_calls(  # noqa: SLF001 - explicit regression test for internal policy
        payload=payload,
        native_tool_calls=native_tool_calls,
    )

    assert resolved == [
        {
            "tool": "slack_update",
            "reason": "native fallback",
            "args": {
                "channel_id": "C0AGLKF6KHD",
                "message_ts": "1772178090.522069",
                "text": "Подтверждено :white_check_mark:",
            },
        }
    ]


def test_resolve_tool_calls_returns_empty_when_all_sources_invalid(monkeypatch, tmp_path) -> None:
    coordinator = _build_coordinator(monkeypatch, tmp_path)
    payload = {
        "tool_calls": [
            {
                "tool": "slack_update",
                "reason": "invalid payload call",
                "args": {"channel_id": "C0AGLKF6KHD"},
            }
        ]
    }
    native_tool_calls = [
        {
            "tool": "slack_update",
            "reason": "invalid native call",
            "args": {"message_ts": "1772178090.522069"},
        }
    ]

    resolved = coordinator._resolve_tool_calls(  # noqa: SLF001 - explicit regression test for internal policy
        payload=payload,
        native_tool_calls=native_tool_calls,
    )

    assert resolved == []
