from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from src.release_workflow import ReleaseWorkflow, RuntimeConfig
from src.runtime_engine import RuntimeDecision
from src.workflow_state import ReleaseContext, ReleaseStep, WorkflowState


@dataclass
class _DummySlackGateway:
    updated_payloads: list[dict[str, str]]
    sent_payloads: list[dict[str, str]]

    def update_message(self, channel_id: str, message_ts: str, text: str) -> dict[str, str]:
        self.updated_payloads.append(
            {
                "channel_id": channel_id,
                "message_ts": message_ts,
                "text": text,
            }
        )
        return {"message_ts": message_ts}

    def send_message(self, channel_id: str, text: str, *, thread_ts: str | None = None) -> dict[str, str]:
        self.sent_payloads.append(
            {
                "channel_id": channel_id,
                "text": text,
                "thread_ts": thread_ts or "",
            }
        )
        return {"message_ts": "200.300"}


@dataclass
class _NoopStateStore:
    state: WorkflowState

    def load(self) -> WorkflowState:
        return self.state

    def save(self, *, state: WorkflowState, reason: str) -> None:  # noqa: ARG002
        self.state = state


@dataclass
class _NoopRuntime:
    def kickoff(self, *, state, events, config, now=None, trigger_reason="event_trigger"):  # noqa: ANN001
        _ = (events, config, now, trigger_reason)
        return RuntimeDecision(
            next_step=state.step,
            next_state=state,
            audit_reason="noop",
            flow_lifecycle="running",
        )


def _config(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        slack_channel_id="C_RELEASE",
        slack_bot_token="xoxb-test-token",
        timezone="Europe/Moscow",
        jira_project_keys=["IOS"],
        readiness_owners={"Growth": "@owner", "Core": "@owner-core"},
        memory_db_path=str(tmp_path / "workflow_state.jsonl"),
        audit_log_path=str(tmp_path / "workflow_audit.jsonl"),
        slack_events_path=str(tmp_path / "slack_events.jsonl"),
    )


def _build_workflow(tmp_path: Path) -> tuple[ReleaseWorkflow, _DummySlackGateway]:
    state = WorkflowState(
        active_release=ReleaseContext(
            release_version="5.105.0",
            step=ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION,
            slack_channel_id="C_RELEASE",
        )
    )
    slack_gateway = _DummySlackGateway(updated_payloads=[], sent_payloads=[])
    workflow = ReleaseWorkflow(
        config=_config(tmp_path),
        slack_gateway=slack_gateway,  # type: ignore[arg-type]
        state_store=_NoopStateStore(state=state),
        runtime_engine=_NoopRuntime(),
    )
    return workflow, slack_gateway


def _decision() -> RuntimeDecision:
    return RuntimeDecision(
        next_step=ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION,
        next_state=WorkflowState(
            active_release=ReleaseContext(
                release_version="5.105.0",
                step=ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION,
                slack_channel_id="C_RELEASE",
            )
        ),
        tool_calls=[],
        audit_reason="test",
        flow_lifecycle="paused",
    )


def test_slack_update_uses_trigger_message_text_and_appends_confirmation(tmp_path: Path) -> None:
    workflow, gateway = _build_workflow(tmp_path)
    decision = _decision()
    call = {"args": {"channel_id": "C_RELEASE", "message_ts": "111.222", "text": "Подтверждено :white_check_mark:"}}
    events = [
        SimpleNamespace(
            event_type="approval_confirmed",
            channel_id="C_RELEASE",
            message_ts="111.222",
            metadata={"trigger_message_text": "Подтвердите старт релизного трейна 5.105.0"},
        )
    ]

    workflow._execute_slack_update(call=call, decision=decision, events=events)  # noqa: SLF001

    assert gateway.updated_payloads == [
        {
            "channel_id": "C_RELEASE",
            "message_ts": "111.222",
            "text": "Подтвердите старт релизного трейна 5.105.0\n\nПодтверждено :white_check_mark:",
        }
    ]


def test_slack_update_append_confirmation_is_idempotent(tmp_path: Path) -> None:
    workflow, gateway = _build_workflow(tmp_path)
    decision = _decision()
    call = {"args": {"channel_id": "C_RELEASE", "message_ts": "111.222", "text": "Подтверждено :white_check_mark:"}}
    events = [
        SimpleNamespace(
            event_type="approval_confirmed",
            channel_id="C_RELEASE",
            message_ts="111.222",
            metadata={"trigger_message_text": "Текст\n\nПодтверждено :white_check_mark:"},
        )
    ]

    workflow._execute_slack_update(call=call, decision=decision, events=events)  # noqa: SLF001

    assert gateway.updated_payloads[0]["text"] == "Текст\n\nПодтверждено :white_check_mark:"


def test_slack_update_falls_back_to_args_text_when_trigger_message_missing(tmp_path: Path) -> None:
    workflow, gateway = _build_workflow(tmp_path)
    decision = _decision()
    call = {"args": {"channel_id": "C_RELEASE", "message_ts": "111.222", "text": "Базовый текст"}}
    events = [
        SimpleNamespace(
            event_type="approval_confirmed",
            channel_id="C_RELEASE",
            message_ts="111.222",
            metadata={},
        )
    ]

    workflow._execute_slack_update(call=call, decision=decision, events=events)  # noqa: SLF001

    assert gateway.updated_payloads[0]["text"] == "Базовый текст\n\nПодтверждено :white_check_mark:"


def test_readiness_confirmation_updates_original_checklist_without_new_message(tmp_path: Path) -> None:
    workflow, gateway = _build_workflow(tmp_path)
    decision = RuntimeDecision(
        next_step=ReleaseStep.WAIT_READINESS_CONFIRMATIONS,
        next_state=WorkflowState(
            active_release=ReleaseContext(
                release_version="5.105.0",
                step=ReleaseStep.WAIT_READINESS_CONFIRMATIONS,
                slack_channel_id="C_RELEASE",
                message_ts={"generic_message": "111.100"},
                thread_ts={"readiness": "111.100"},
                readiness_map={"Growth": True, "Core": False},
            )
        ),
        tool_calls=[],
        audit_reason="readiness_confirmation:Growth",
        flow_lifecycle="paused",
    )
    call = {
        "args": {
            "channel_id": "C_RELEASE",
            "text": "Принято — Growth отмечен как готово.",
        }
    }

    workflow._execute_slack_message(call=call, decision=decision, events=[])  # noqa: SLF001

    assert gateway.sent_payloads == []
    assert gateway.updated_payloads == [
        {
            "channel_id": "C_RELEASE",
            "message_ts": "111.100",
            "text": (
                "Релиз <https://instories.atlassian.net/issues/?jql=fixVersion%20%3D%205.105.0%20ORDER%20BY%20project%20ASC%2C%20type%20DESC%2C%20key%20ASC|5.105.0>\n\n"
                "Статус готовности к срезу:\n"
                ":white_check_mark: Growth @owner\n"
                ":hourglass_flowing_sand: Core @owner-core\n\n"
                "Напишите в треде по готовности своей части.\n\n"
                "*Важное напоминание* – все задачи, не влитые в ветку RC до 15:00 МСК едут в релиз только после одобрения QA"
            ),
        }
    ]


def test_readiness_confirmation_sequentially_updates_only_target_lines(tmp_path: Path) -> None:
    workflow, gateway = _build_workflow(tmp_path)
    decision = RuntimeDecision(
        next_step=ReleaseStep.WAIT_READINESS_CONFIRMATIONS,
        next_state=WorkflowState(
            active_release=ReleaseContext(
                release_version="5.105.0",
                step=ReleaseStep.WAIT_READINESS_CONFIRMATIONS,
                slack_channel_id="C_RELEASE",
                message_ts={"generic_message": "111.100"},
                thread_ts={"readiness": "111.100"},
                readiness_map={"Growth": True, "Core": False},
            )
        ),
        tool_calls=[],
        audit_reason="readiness_confirmation:Growth",
        flow_lifecycle="paused",
    )
    call = {"args": {"channel_id": "C_RELEASE", "text": "ack"}}
    workflow._execute_slack_message(call=call, decision=decision, events=[])  # noqa: SLF001

    release = decision.next_state.active_release
    assert release is not None
    release.readiness_map["Core"] = True
    decision.audit_reason = "readiness_confirmation:Core by U00000002"
    workflow._execute_slack_message(call=call, decision=decision, events=[])  # noqa: SLF001

    assert gateway.sent_payloads == []
    assert len(gateway.updated_payloads) == 2
    first_text = gateway.updated_payloads[0]["text"]
    second_text = gateway.updated_payloads[1]["text"]
    assert ":white_check_mark: Growth @owner" in first_text
    assert ":hourglass_flowing_sand: Core @owner-core" in first_text
    assert ":white_check_mark: Growth @owner" in second_text
    assert ":white_check_mark: Core @owner-core" in second_text
