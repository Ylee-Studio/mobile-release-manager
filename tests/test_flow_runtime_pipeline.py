from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from src.runtime_engine import DirectLLMRuntime, RuntimeCoordinator, RuntimeDecision
from src.policies import PolicyConfig
from src.release_workflow import (
    ReleaseWorkflow,
    RuntimeConfig,
    _normalize_readiness_message_text,
)
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


@dataclass
class _KickoffProbeRuntime:
    trigger_reason: str | None = None

    def kickoff(self, *, state, events, config, now=None, trigger_reason="event_trigger"):  # noqa: ANN001
        self.trigger_reason = trigger_reason
        return RuntimeDecision(
            next_step=state.step,
            next_state=state,
            audit_reason="probe",
            actor="flow_agent",
            flow_lifecycle="running",
        )


def _runtime_config(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        slack_channel_id="C_RELEASE",
        slack_bot_token="xoxb-test-token",
        timezone="Europe/Moscow",
        jira_project_keys=["IOS"],
        readiness_owners={"Growth": "owner"},
        memory_db_path=str(tmp_path / "workflow_state.jsonl"),
        audit_log_path=str(tmp_path / "workflow_audit.jsonl"),
        slack_events_path=str(tmp_path / "slack_events.jsonl"),
    )


@dataclass
class _MemoryRecord:
    content: str
    metadata: dict


@dataclass
class _MemoryMatch:
    record: _MemoryRecord


@dataclass
class _InMemoryNativeMemory:
    entries: list[dict] = field(default_factory=list)

    def remember(self, *, content: str, metadata: dict, **_: object) -> None:
        self.entries.append({"content": content, "metadata": metadata})

    def recall(self, **_: object) -> list[_MemoryMatch]:
        return [
            _MemoryMatch(record=_MemoryRecord(content=entry["content"], metadata=entry["metadata"]))
            for entry in reversed(self.entries)
        ]


@dataclass
class _NoopStateStore:
    state: WorkflowState

    def load(self) -> WorkflowState:
        return self.state

    def save(self, *, state: WorkflowState, reason: str) -> None:  # noqa: ARG002
        self.state = state


def _seed_state(memory: _InMemoryNativeMemory, state: WorkflowState, *, reason: str = "seed") -> None:
    memory.remember(
        content="{}",
        metadata={"state": state.to_dict(), "reason": reason},
    )


def test_release_workflow_propagates_trigger_reason_to_runtime(tmp_path: Path) -> None:
    config = _runtime_config(tmp_path)
    memory = _InMemoryNativeMemory()
    gateway = SlackGateway(bot_token="xoxb-test-token", events_path=Path(config.slack_events_path))
    runtime = _KickoffProbeRuntime()
    workflow = ReleaseWorkflow(
        config=config,
        memory=memory,
        slack_gateway=gateway,
        legacy_runtime=runtime,
    )

    state = workflow.tick(trigger_reason="signal_trigger")

    assert state.step == ReleaseStep.IDLE
    assert runtime.trigger_reason == "signal_trigger"


def test_release_workflow_calls_decision_before_processing_reaction(tmp_path: Path) -> None:
    call_order: list[str] = []

    class _OrderedSlackGateway:
        def poll_events(self) -> list[SlackEvent]:
            call_order.append("poll_events")
            return [
                SlackEvent(
                    event_id="evt-1",
                    event_type="message",
                    channel_id="C_RELEASE",
                    text="start",
                    message_ts="1700000000.123456",
                )
            ]

        def add_reaction(self, channel_id: str, message_ts: str, emoji: str = "eyes") -> bool:  # noqa: ARG002
            call_order.append("add_reaction")
            return True

        def pending_events_count(self) -> int:
            call_order.append("pending_events_count")
            return 0

    class _OrderedRuntime:
        def kickoff(self, *, state, events, config, now=None, trigger_reason="event_trigger"):  # noqa: ANN001
            _ = (events, config, now, trigger_reason)
            call_order.append("kickoff")
            return RuntimeDecision(
                next_step=state.step,
                next_state=state,
                audit_reason="ordered",
                actor="flow_agent",
                flow_lifecycle="running",
            )

    config = _runtime_config(tmp_path)
    workflow = ReleaseWorkflow(
        config=config,
        slack_gateway=_OrderedSlackGateway(),  # type: ignore[arg-type]
        state_store=_NoopStateStore(WorkflowState()),
        runtime_engine=_OrderedRuntime(),
    )

    workflow.tick(trigger_reason="event_trigger")

    assert "kickoff" in call_order
    assert "add_reaction" in call_order
    assert call_order.index("kickoff") < call_order.index("pending_events_count")


def test_release_workflow_keeps_paused_state_without_approval_event(tmp_path: Path) -> None:
    class _PausedRuntime:
        def kickoff(self, *, state, events, config, now=None, trigger_reason="event_trigger"):  # noqa: ANN001
            _ = (events, config, now, trigger_reason)
            return RuntimeDecision(
                next_step=state.step,
                next_state=state,
                audit_reason="paused",
                actor="flow_runtime",
                flow_lifecycle="paused",
            )

    config = _runtime_config(tmp_path)
    memory = _InMemoryNativeMemory()
    paused_state = WorkflowState(
        active_release=ReleaseContext(
            release_version="5.104.0",
            step=ReleaseStep.WAIT_MEETING_CONFIRMATION,
        ),
        flow_execution_id="flow-1",
        flow_paused_at="2026-01-01T00:00:00+00:00",
        pause_reason="awaiting_confirmation:WAIT_MEETING_CONFIRMATION",
    )
    _seed_state(memory, paused_state)
    gateway = SlackGateway(bot_token="xoxb-test-token", events_path=Path(config.slack_events_path))
    workflow = ReleaseWorkflow(
        config=config,
        memory=memory,
        slack_gateway=gateway,
        legacy_runtime=_PausedRuntime(),
    )

    state = workflow.tick(trigger_reason="event_trigger")

    assert state.is_paused is True
    assert state.flow_execution_id == "flow-1"


def test_kickoff_returns_safe_decision_on_invalid_structured_output(monkeypatch, tmp_path: Path) -> None:
    class _InvalidCrewResult:
        raw = "not-a-json-payload"
        tasks_output = []

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    gateway = SlackGateway(bot_token="xoxb-test-token", events_path=tmp_path / "events.jsonl")
    coordinator = RuntimeCoordinator(
        policy=_policy(),
        slack_gateway=gateway,
        memory_db_path=str(tmp_path / "memory.db"),
    )

    monkeypatch.setattr(
        coordinator,
        "_execute_runtime",
        lambda **kwargs: _InvalidCrewResult(),  # noqa: ARG005
    )
    initial_state = WorkflowState()
    decision = coordinator.kickoff(
        state=initial_state,
        events=[],
        config=_config(),
    )

    assert decision.next_step == initial_state.step
    assert decision.next_state == initial_state
    assert decision.audit_reason.startswith("runtime_error:")


def test_direct_llm_runtime_logs_usage_tokens(monkeypatch, caplog) -> None:  # noqa: ANN001
    runtime = object.__new__(DirectLLMRuntime)
    runtime._agent_retries = 3
    expected_output = SimpleNamespace(ok=True)

    class _Usage:
        request_tokens = 42
        response_tokens = 9
        total_tokens = 51
        requests = 1

    class _RunResult:
        output = expected_output

        def usage(self) -> _Usage:
            return _Usage()

    async def _fake_run_agent_async(
        self,
        *,
        prompt: str,
        model_name: str,
        trigger_reason: str,
        **_: object,
    ) -> tuple[_RunResult, dict[str, int]]:  # noqa: ANN001, ARG001
        return _RunResult(), {"model_wait_ms": 0, "result_parse_ms": 0}

    monkeypatch.setattr(
        DirectLLMRuntime,
        "_build_user_prompt",
        lambda self, **kwargs: "test-prompt",  # noqa: ARG005
    )
    monkeypatch.setattr(DirectLLMRuntime, "_run_agent_async", _fake_run_agent_async)

    with caplog.at_level(logging.INFO, logger="runtime_engine"):
        output = DirectLLMRuntime._run_agent_sync(runtime, payload={"trigger_reason": "webhook_event"})

    assert output is expected_output
    assert "llm_usage" in caplog.text
    assert "request_tokens=42" in caplog.text
    assert "response_tokens=9" in caplog.text
    assert "total_tokens=51" in caplog.text
    assert "requests=1" in caplog.text
    assert "usage_unavailable=False" in caplog.text


def test_direct_llm_runtime_logs_when_usage_unavailable(monkeypatch, caplog) -> None:  # noqa: ANN001
    runtime = object.__new__(DirectLLMRuntime)
    runtime._agent_retries = 3
    expected_output = SimpleNamespace(ok=True)

    class _RunResult:
        output = expected_output

        def usage(self) -> None:
            return None

    async def _fake_run_agent_async(
        self,
        *,
        prompt: str,
        model_name: str,
        trigger_reason: str,
        **_: object,
    ) -> tuple[_RunResult, dict[str, int]]:  # noqa: ANN001, ARG001
        return _RunResult(), {"model_wait_ms": 0, "result_parse_ms": 0}

    monkeypatch.setattr(
        DirectLLMRuntime,
        "_build_user_prompt",
        lambda self, **kwargs: "test-prompt",  # noqa: ARG005
    )
    monkeypatch.setattr(DirectLLMRuntime, "_run_agent_async", _fake_run_agent_async)

    with caplog.at_level(logging.INFO, logger="runtime_engine"):
        output = DirectLLMRuntime._run_agent_sync(runtime, payload={"trigger_reason": "webhook_event"})

    assert output is expected_output
    assert "llm_usage" in caplog.text
    assert "usage_unavailable=True" in caplog.text


def test_runtime_coordinator_logs_kickoff_timing(monkeypatch, tmp_path: Path, caplog) -> None:  # noqa: ANN001
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    gateway = SlackGateway(bot_token="xoxb-test-token", events_path=tmp_path / "events.jsonl")
    coordinator = RuntimeCoordinator(
        policy=_policy(),
        slack_gateway=gateway,
        memory_db_path=str(tmp_path / "memory.db"),
    )

    monkeypatch.setattr(
        coordinator,
        "_execute_runtime",
        lambda **kwargs: RuntimeDecision(  # noqa: ARG005
            next_step=ReleaseStep.IDLE,
            next_state=WorkflowState(),
            audit_reason="timing_probe",
            actor="flow_agent",
            flow_lifecycle="completed",
        ).to_dict(),
    )

    with caplog.at_level(logging.INFO, logger="runtime_engine"):
        decision = coordinator.kickoff(state=WorkflowState(), events=[], config=_config())

    assert decision.next_step == ReleaseStep.IDLE
    assert "kickoff_decision_start" in caplog.text
    assert "kickoff_decision_done" in caplog.text


def test_kickoff_routes_paused_message_to_agent_for_override_with_mention(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    gateway = SlackGateway(bot_token="xoxb-test-token", events_path=tmp_path / "events.jsonl")
    coordinator = RuntimeCoordinator(
        policy=_policy(),
        slack_gateway=gateway,
        memory_db_path=str(tmp_path / "memory.db"),
    )
    captured_events: list[dict[str, object]] = []

    def _agent_override_decision(*, payload: dict[str, object]) -> dict[str, object]:
        captured_events.extend(payload.get("events", []))  # type: ignore[arg-type]
        return {
            "next_step": "IDLE",
            "next_state": {
                "active_release": None,
                "previous_release_version": "5.105.0",
                "checkpoints": [],
            },
            "tool_calls": [],
            "audit_reason": "manual_status_override:5.105.0->IDLE",
            "flow_lifecycle": "completed",
        }

    monkeypatch.setattr(coordinator, "_execute_runtime", _agent_override_decision)

    initial_state = WorkflowState(
        active_release=ReleaseContext(
            release_version="5.105.0",
            step=ReleaseStep.WAIT_MEETING_CONFIRMATION,
            slack_channel_id="C_RELEASE",
        ),
        flow_execution_id="flow-1",
        flow_paused_at="2026-01-01T00:00:00+00:00",
        pause_reason="awaiting_confirmation",
    )
    decision = coordinator.kickoff(
        state=initial_state,
        events=[
            SlackEvent(
                event_id="evt-1",
                event_type="message",
                channel_id="C_RELEASE",
                text="<@U0AHX8XBZBJ> Переведи 5.105.0 в IDLE",
                message_ts="123.456",
            )
        ],
        config=_config(),
    )

    assert len(captured_events) == 1
    assert captured_events[0]["text"] == "<@U0AHX8XBZBJ> Переведи 5.105.0 в IDLE"
    assert decision.next_step == ReleaseStep.IDLE
    assert decision.next_state.active_release is None
    assert decision.next_state.previous_release_version == "5.105.0"
    assert decision.flow_lifecycle == "completed"
    assert decision.audit_reason == "manual_status_override:5.105.0->IDLE"


def test_kickoff_routes_non_message_paused_events_to_agent(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    gateway = SlackGateway(bot_token="xoxb-test-token", events_path=tmp_path / "events.jsonl")
    coordinator = RuntimeCoordinator(
        policy=_policy(),
        slack_gateway=gateway,
        memory_db_path=str(tmp_path / "memory.db"),
    )
    called = {"value": False}

    def _agent_decision(*, payload: dict[str, object]) -> dict[str, object]:
        called["value"] = True
        events = payload.get("events", [])
        assert isinstance(events, list)
        assert len(events) == 1
        assert events[0]["event_type"] == "approval_confirmed"
        return {
            "next_step": "WAIT_MANUAL_RELEASE_CONFIRMATION",
            "next_state": {
                "active_release": {
                    "release_version": "5.105.0",
                    "step": "WAIT_MANUAL_RELEASE_CONFIRMATION",
                    "slack_channel_id": "C_RELEASE",
                    "message_ts": {},
                    "thread_ts": {},
                    "readiness_map": {},
                },
                "flow_execution_id": "flow-1",
                "flow_paused_at": "2026-01-01T00:00:00+00:00",
                "pause_reason": "awaiting_confirmation",
                "checkpoints": [],
            },
            "tool_calls": [],
            "audit_reason": "agent_transition_after_approval_event",
            "flow_lifecycle": "paused",
        }

    monkeypatch.setattr(coordinator, "_execute_runtime", _agent_decision)

    initial_state = WorkflowState(
        active_release=ReleaseContext(
            release_version="5.105.0",
            step=ReleaseStep.WAIT_START_APPROVAL,
            slack_channel_id="C_RELEASE",
        ),
        flow_execution_id="flow-1",
        flow_paused_at="2026-01-01T00:00:00+00:00",
        pause_reason="awaiting_confirmation",
    )
    decision = coordinator.kickoff(
        state=initial_state,
        events=[
            SlackEvent(
                event_id="evt-2",
                event_type="approval_confirmed",
                channel_id="C_RELEASE",
                text="approve",
                message_ts="123.457",
            )
        ],
        config=_config(),
    )

    assert called["value"] is True
    assert decision.next_step == ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION
    assert decision.flow_lifecycle == "paused"
    assert decision.audit_reason == "agent_transition_after_approval_event"


def test_kickoff_routes_reject_event_and_allows_reset_to_idle(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    gateway = SlackGateway(bot_token="xoxb-test-token", events_path=tmp_path / "events.jsonl")
    coordinator = RuntimeCoordinator(
        policy=_policy(),
        slack_gateway=gateway,
        memory_db_path=str(tmp_path / "memory.db"),
    )
    called = {"value": False}

    def _agent_decision(*, payload: dict[str, object]) -> dict[str, object]:
        called["value"] = True
        events = payload.get("events", [])
        assert isinstance(events, list)
        assert len(events) == 1
        assert events[0]["event_type"] == "approval_rejected"
        return {
            "next_step": "IDLE",
            "next_state": {
                "active_release": None,
                "checkpoints": [],
            },
            "tool_calls": [],
            "audit_reason": "start_rejected",
            "flow_lifecycle": "completed",
        }

    monkeypatch.setattr(coordinator, "_execute_runtime", _agent_decision)

    initial_state = WorkflowState(
        active_release=ReleaseContext(
            release_version="5.105.0",
            step=ReleaseStep.WAIT_START_APPROVAL,
            slack_channel_id="C_RELEASE",
        ),
        flow_execution_id="flow-1",
        flow_paused_at="2026-01-01T00:00:00+00:00",
        pause_reason="awaiting_confirmation",
    )
    decision = coordinator.kickoff(
        state=initial_state,
        events=[
            SlackEvent(
                event_id="evt-3",
                event_type="approval_rejected",
                channel_id="C_RELEASE",
                text="reject",
                message_ts="123.458",
            )
        ],
        config=_config(),
    )

    assert called["value"] is True
    assert decision.next_step == ReleaseStep.IDLE
    assert decision.next_state.active_release is None
    assert decision.flow_lifecycle == "completed"
    assert decision.audit_reason == "start_rejected"


def test_normalize_readiness_message_adds_mentions_and_keeps_reminder_bold() -> None:
    source = (
        "Релиз 5.104.0\n\n"
        "Статус готовности к срезу:\n"
        ":hourglass_flowing_sand: Growth Мурат Камалов\n"
        ":hourglass_flowing_sand: Core Антон Давыдов\n\n"
        "Напишите в треде по готовности своей части.\n"
        "Важное напоминание – все задачи, не влитые в ветку RC до 15:00 МСК "
        "едут в релиз только после одобрения QA"
    )
    normalized = _normalize_readiness_message_text(
        text=source,
        release_version="5.104.0",
        readiness_owners={
            "Growth": "Мурат Камалов",
            "Core": "Антон Давыдов",
        },
    )

    assert ":hourglass_flowing_sand: Growth @Мурат Камалов" in normalized
    assert ":hourglass_flowing_sand: Core @Антон Давыдов" in normalized
    assert "*Важное напоминание* – все задачи" in normalized


def test_normalize_readiness_message_preserves_existing_slack_id_mentions() -> None:
    source = (
        "Релиз 5.104.0\n\n"
        "Статус готовности к срезу:\n"
        ":hourglass_flowing_sand: Growth Someone\n\n"
        "Напишите в треде по готовности своей части.\n"
        "*Важное напоминание* – все задачи, не влитые в ветку RC до 15:00 МСК "
        "едут в релиз только после одобрения QA"
    )
    normalized = _normalize_readiness_message_text(
        text=source,
        release_version="5.104.0",
        readiness_owners={"Growth": "<@U12345>"},
    )

    assert ":hourglass_flowing_sand: Growth <@U12345>" in normalized


def test_normalize_readiness_message_converts_channel_like_mentions() -> None:
    source = (
        "Релиз 5.104.0\n\n"
        "Статус готовности к срезу:\n"
        ":hourglass_flowing_sand: Growth Someone\n\n"
        "Напишите в треде по готовности своей части.\n"
        "Важное напоминание – все задачи, не влитые в ветку RC до 15:00 МСК "
        "едут в релиз только после одобрения QA"
    )
    normalized = _normalize_readiness_message_text(
        text=source,
        release_version="5.104.0",
        readiness_owners={"Growth": "<@D09TKUA7R9U>"},
    )

    assert ":hourglass_flowing_sand: Growth <#D09TKUA7R9U>" in normalized
