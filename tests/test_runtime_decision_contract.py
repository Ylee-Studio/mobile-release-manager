from src.runtime_engine import RuntimeDecision
from src.tool_calling import extract_native_tool_calls, validate_and_normalize_tool_calls
from src.tools.slack_tools import SlackApproveInput, SlackMessageInput
from src.workflow_state import ReleaseStep, WorkflowState


def test_decision_accepts_full_next_state() -> None:
    payload = {
        "next_step": "WAIT_START_APPROVAL",
        "next_state": {
            "active_release": {
                "release_version": "2.0.0",
                "step": "WAIT_START_APPROVAL",
                "message_ts": {},
                "thread_ts": {},
                "readiness_map": {"Growth": False},
            },
            "checkpoints": [],
        },
        "tool_calls": [{"tool": "slack_approve", "reason": "request approval"}],
        "audit_reason": "new_release_started",
    }
    decision = RuntimeDecision.from_payload(payload, current_state=WorkflowState())
    assert decision.next_step == ReleaseStep.WAIT_START_APPROVAL
    assert decision.next_state.active_release is not None
    assert decision.next_state.active_release.release_version == "2.0.0"
    assert decision.tool_calls[0]["tool"] == "slack_approve"


def test_decision_supports_state_patch_merge() -> None:
    current = WorkflowState(
        active_release=None,
        previous_release_version="1.0.0",
        checkpoints=[],
    )
    payload = {
        "next_step": "IDLE",
        "state_patch": {"pause_reason": "patched"},
        "audit_reason": "mark_event_processed",
    }
    decision = RuntimeDecision.from_payload(payload, current_state=current)
    assert decision.next_step == ReleaseStep.IDLE
    assert decision.next_state.pause_reason == "patched"


def test_decision_keeps_single_agent_contract_fields() -> None:
    payload = {
        "next_step": "WAIT_MANUAL_RELEASE_CONFIRMATION",
        "next_state": {
            "active_release": {
                "release_version": "3.1.0",
                "step": "WAIT_MANUAL_RELEASE_CONFIRMATION",
                "message_ts": {},
                "thread_ts": {},
                "readiness_map": {"Core": False},
            },
            "checkpoints": [],
        },
        "tool_calls": [{"tool": "slack_update", "reason": "prepare release manager phase"}],
        "audit_reason": "flow_agent_transition",
    }

    decision = RuntimeDecision.from_payload(payload, current_state=WorkflowState(), actor="flow_agent")
    assert decision.actor == "flow_agent"
    assert decision.next_step == ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION
    assert decision.next_state.active_release is not None
    assert decision.next_state.active_release.release_version == "3.1.0"


def test_decision_accepts_active_release_aliases_from_agent() -> None:
    payload = {
        "next_step": "WAIT_START_APPROVAL",
        "next_state": {
            "active_release": {
                "version": "5.104.0",
                "status": "AWAITING_START_APPROVAL",
                "channel_id": "C0AGLKF6KHD",
            },
            "checkpoints": [],
        },
        "tool_calls": [{"tool": "slack_approve", "reason": "request approval"}],
        "audit_reason": "manual_start_processed",
    }

    decision = RuntimeDecision.from_payload(payload, current_state=WorkflowState())
    assert decision.next_step == ReleaseStep.WAIT_START_APPROVAL
    assert decision.next_state.active_release is not None
    assert decision.next_state.active_release.release_version == "5.104.0"
    assert decision.next_state.active_release.step == ReleaseStep.WAIT_START_APPROVAL


def test_decision_accepts_manual_release_confirmation_alias_from_agent() -> None:
    payload = {
        "next_step": "WAIT_MANUAL_RELEASE_CONFIRMATION",
        "next_state": {
            "active_release": {
                "version": "5.104.0",
                "status": "AWAITING_MANUAL_RELEASE_CONFIRMATION",
                "channel_id": "C0AGLKF6KHD",
            },
            "checkpoints": [],
        },
        "tool_calls": [{"tool": "slack_approve", "reason": "request manual release confirmation"}],
        "audit_reason": "manual_release_confirmation_requested",
    }

    decision = RuntimeDecision.from_payload(payload, current_state=WorkflowState())
    assert decision.next_step == ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION
    assert decision.next_state.active_release is not None
    assert decision.next_state.active_release.release_version == "5.104.0"
    assert decision.next_state.active_release.step == ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION


def test_decision_normalizes_string_active_release_in_state_patch() -> None:
    payload = {
        "next_step": "WAIT_MANUAL_RELEASE_CONFIRMATION",
        "state_patch": {
            "active_release": "5.105.0",
            "flow_execution_id": "exec-5.105.0",
            "flow_paused_at": "2026-02-27T06:51:41.471780+00:00",
            "pause_reason": "Waiting for manual release confirmation",
        },
        "tool_calls": [],
        "audit_reason": "Manual start event received, transition to wait for manual release confirmation",
    }

    decision = RuntimeDecision.from_payload(payload, current_state=WorkflowState())
    assert decision.next_step == ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION
    assert decision.next_state.active_release is not None
    assert decision.next_state.active_release.release_version == "5.105.0"
    assert decision.next_state.active_release.step == ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION
    assert decision.next_state.flow_execution_id == "exec-5.105.0"
    assert decision.tool_calls == []


def test_decision_normalizes_string_active_release_in_next_state_wait_manual() -> None:
    payload = {
        "next_step": "WAIT_MANUAL_RELEASE_CONFIRMATION",
        "next_state": {
            "active_release": "5.105.0",
            "flow_execution_id": "exec-5.105.0",
            "flow_paused_at": "2026-02-27T07:04:16.584921+00:00",
            "pause_reason": "Waiting for manual release confirmation.",
            "checkpoints": [],
        },
        "tool_calls": [],
        "audit_reason": "Received manual_start event for release 5.105.0, starting release flow and pausing for manual confirmation",
    }

    decision = RuntimeDecision.from_payload(payload, current_state=WorkflowState())
    assert decision.next_step == ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION
    assert decision.next_state.active_release is not None
    assert decision.next_state.active_release.release_version == "5.105.0"
    assert decision.next_state.active_release.step == ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION
    assert decision.next_state.flow_execution_id == "exec-5.105.0"
    assert decision.tool_calls == []


def test_decision_normalizes_string_active_release_in_next_state_idle() -> None:
    payload = {
        "next_step": "IDLE",
        "next_state": {
            "active_release": "5.105.0",
            "checkpoints": [],
        },
        "audit_reason": "release_completed",
    }

    decision = RuntimeDecision.from_payload(payload, current_state=WorkflowState())
    assert decision.next_step == ReleaseStep.IDLE
    assert decision.next_state.active_release is None


def test_native_tool_call_extraction_from_canonical_payload() -> None:
    result = {
        "tasks_output": [
            {
                "tool_calls": [
                    {
                        "tool": "slack_approve",
                        "args": {
                            "channel_id": "C0AGLKF6KHD",
                            "text": "start",
                            "approve_label": "Ok",
                        },
                    }
                ]
            }
        ]
    }

    extracted = extract_native_tool_calls(result)

    assert extracted == [
        {
            "tool": "slack_approve",
            "reason": "",
            "args": {"channel_id": "C0AGLKF6KHD", "text": "start", "approve_label": "Ok"},
        }
    ]


def test_validation_normalizes_functions_alias_and_validates_args_schema() -> None:
    tool_calls = [
        {
            "tool": "functions.slack_approve",
            "reason": "request approval",
            "args": {
                "channel_id": " C0AGLKF6KHD ",
                "text": " Подтвердите старт ",
                "approve_label": " Подтвердить ",
            },
        }
    ]

    validated = validate_and_normalize_tool_calls(
        tool_calls,
        schema_by_tool={"slack_approve": SlackApproveInput},
    )

    assert validated == [
        {
            "tool": "slack_approve",
            "reason": "request approval",
            "args": {
                "channel_id": "C0AGLKF6KHD",
                "text": "Подтвердите старт",
                "approve_label": "Подтвердить",
            },
        }
    ]


def test_validation_accepts_readiness_slack_message_with_required_sections() -> None:
    readiness_text = (
        "Релиз <https://instories.atlassian.net/issues/?jql=JQL|5.104.0>\n\n"
        "Статус готовности к срезу:\n"
        ":hourglass_flowing_sand: Growth @owner\n\n"
        "Напишите в треде по готовности своей части.\n"
        "**Важное напоминание** – все задачи, не влитые в ветку RC до 15:00 МСК "
        "едут в релиз только после одобрения QA"
    )
    tool_calls = [
        {
            "tool": "slack_message",
            "reason": "announce readiness confirmation",
            "args": {
                "channel_id": " C0AGLKF6KHD ",
                "text": f" {readiness_text} ",
            },
        }
    ]

    validated = validate_and_normalize_tool_calls(
        tool_calls,
        schema_by_tool={"slack_message": SlackMessageInput},
    )

    assert validated == [
        {
            "tool": "slack_message",
            "reason": "announce readiness confirmation",
            "args": {
                "channel_id": "C0AGLKF6KHD",
                "text": readiness_text,
            },
        }
    ]
