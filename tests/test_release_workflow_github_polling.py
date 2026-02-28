from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.release_workflow import ReleaseWorkflow, RuntimeConfig
from src.runtime_engine import RuntimeDecision
from src.tools.slack_tools import SlackEvent
from src.workflow_state import ReleaseContext, ReleaseStep, WorkflowState


@dataclass
class _NoopSlackGateway:
    def poll_events(self) -> list[Any]:
        return []

    def pending_events_count(self) -> int:
        return 0

    def add_reaction(self, channel_id: str, message_ts: str, emoji: str = "eyes") -> bool:  # noqa: ARG002
        return True


@dataclass
class _EventSlackGateway:
    events: list[SlackEvent]
    reaction_calls: list[dict[str, str]] | None = None
    sent_messages: list[dict[str, str]] | None = None
    updated_payloads: list[dict[str, str]] | None = None

    def poll_events(self) -> list[Any]:
        return list(self.events)

    def pending_events_count(self) -> int:
        return 0

    def add_reaction(self, channel_id: str, message_ts: str, emoji: str = "eyes") -> bool:
        if self.reaction_calls is None:
            self.reaction_calls = []
        self.reaction_calls.append(
            {
                "channel_id": channel_id,
                "message_ts": message_ts,
                "emoji": emoji,
            }
        )
        return True

    def send_message(self, channel_id: str, text: str, *, thread_ts: str | None = None) -> dict[str, str]:
        if self.sent_messages is None:
            self.sent_messages = []
        self.sent_messages.append(
            {
                "channel_id": channel_id,
                "text": text,
                "thread_ts": thread_ts or "",
            }
        )
        return {"message_ts": "1700000000.555555"}

    def update_message(self, channel_id: str, message_ts: str, text: str) -> dict[str, str]:
        if self.updated_payloads is None:
            self.updated_payloads = []
        self.updated_payloads.append(
            {
                "channel_id": channel_id,
                "message_ts": message_ts,
                "text": text,
            }
        )
        return {"message_ts": message_ts}

@dataclass
class _StateStore:
    state: WorkflowState

    def load(self) -> WorkflowState:
        return self.state

    def save(self, *, state: WorkflowState, reason: str) -> None:  # noqa: ARG002
        self.state = state


@dataclass
class _RuntimeStub:
    decision: RuntimeDecision

    def kickoff(self, *, state, events, config, now=None, trigger_reason="event_trigger"):  # noqa: ANN001
        _ = (state, events, config, now, trigger_reason)
        return self.decision


@dataclass
class _GitHubGatewayStub:
    trigger_calls: int = 0
    poll_calls: int = 0
    trigger_payloads: list[dict[str, Any]] | None = None

    def trigger_workflow(self, *, workflow_file: str, ref: str, inputs: dict[str, Any]) -> Any:
        self.trigger_calls += 1
        if self.trigger_payloads is None:
            self.trigger_payloads = []
        self.trigger_payloads.append(
            {
                "workflow_file": workflow_file,
                "ref": ref,
                "inputs": dict(inputs),
            }
        )
        return type("Run", (), {"run_id": 101, "status": "queued", "conclusion": None})()

    def get_workflow_run(self, *, run_id: int) -> Any:
        _ = run_id
        self.poll_calls += 1
        return type("Run", (), {"run_id": 101, "status": "completed", "conclusion": "success"})()


def _config(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        slack_channel_id="C_RELEASE",
        slack_bot_token="xoxb-test-token",
        timezone="Europe/Moscow",
        jira_project_keys=["IOS"],
        readiness_owners={"Growth": "<@U1>"},
        memory_db_path=str(tmp_path / "workflow_state.jsonl"),
        audit_log_path=str(tmp_path / "workflow_audit.jsonl"),
        slack_events_path=str(tmp_path / "slack_events.jsonl"),
        github_repo="Ylee-Studio/instories-ios",
        github_token="test-token",
        github_workflow_file="create_release_branch.yml",
        github_ref="dev",
        github_poll_interval_seconds=30,
    )


def test_github_action_dispatch_sets_run_context(tmp_path: Path) -> None:
    state = WorkflowState(
        active_release=ReleaseContext(
            release_version="5.105.0",
            step=ReleaseStep.WAIT_READINESS_CONFIRMATIONS,
            slack_channel_id="C_RELEASE",
        )
    )
    decision_state = WorkflowState.from_dict(state.to_dict())
    assert decision_state.active_release is not None
    decision_state.active_release.set_step(ReleaseStep.WAIT_BRANCH_CUT_APPROVAL)
    runtime = _RuntimeStub(
        decision=RuntimeDecision(
            next_step=ReleaseStep.WAIT_BRANCH_CUT_APPROVAL,
            next_state=decision_state,
            tool_calls=[
                {
                    "tool": "github_action",
                    "reason": "dispatch branch cut workflow",
                    "args": {
                        "workflow_file": "create_release_branch.yml",
                        "ref": "dev",
                        "inputs": {"version": "5.105.0"},
                    },
                }
            ],
            audit_reason="readiness_complete",
            flow_lifecycle="paused",
        )
    )
    github = _GitHubGatewayStub()
    workflow = ReleaseWorkflow(
        config=_config(tmp_path),
        slack_gateway=_NoopSlackGateway(),  # type: ignore[arg-type]
        state_store=_StateStore(state=state),
        runtime_engine=runtime,
        github_gateway=github,  # type: ignore[arg-type]
    )

    next_state = workflow.tick(trigger_reason="manual_tick")

    assert github.trigger_calls == 1
    assert github.poll_calls == 0
    assert next_state.active_release is not None
    assert next_state.active_release.step == ReleaseStep.WAIT_BRANCH_CUT_APPROVAL
    assert next_state.active_release.github_action_run_id == 101
    assert next_state.active_release.github_action_status == "queued"


def test_wait_branch_cut_approval_does_not_poll_github_after_dispatch(tmp_path: Path) -> None:
    state = WorkflowState(
        active_release=ReleaseContext(
            release_version="5.105.0",
            step=ReleaseStep.WAIT_BRANCH_CUT_APPROVAL,
            slack_channel_id="C_RELEASE",
            github_action_run_id=202,
            github_action_status="in_progress",
        )
    )
    runtime = _RuntimeStub(
        decision=RuntimeDecision(
            next_step=ReleaseStep.WAIT_BRANCH_CUT_APPROVAL,
            next_state=WorkflowState.from_dict(state.to_dict()),
            tool_calls=[],
            audit_reason="waiting_for_branch_cut_approval",
            flow_lifecycle="paused",
        )
    )
    github = _GitHubGatewayStub()
    workflow = ReleaseWorkflow(
        config=_config(tmp_path),
        slack_gateway=_NoopSlackGateway(),  # type: ignore[arg-type]
        state_store=_StateStore(state=state),
        runtime_engine=runtime,
        github_gateway=github,  # type: ignore[arg-type]
    )

    next_state = workflow.tick(trigger_reason="manual_tick")

    assert github.poll_calls == 0
    assert next_state.active_release is not None
    assert next_state.active_release.step == ReleaseStep.WAIT_BRANCH_CUT_APPROVAL
    assert next_state.active_release.github_action_status == "in_progress"


def test_stateless_github_action_trigger_does_not_change_state(tmp_path: Path) -> None:
    state = WorkflowState()
    runtime = _RuntimeStub(
        decision=RuntimeDecision(
            next_step=ReleaseStep.IDLE,
            next_state=WorkflowState.from_dict(state.to_dict()),
            tool_calls=[
                {
                    "tool": "github_action",
                    "reason": "manual build rc",
                    "args": {
                        "workflow_file": "build_rc.yml",
                        "ref": "dev",
                        "inputs": {},
                    },
                }
            ],
            audit_reason="manual_build_rc_trigger",
            flow_lifecycle="running",
        )
    )
    github = _GitHubGatewayStub()
    workflow = ReleaseWorkflow(
        config=_config(tmp_path),
        slack_gateway=_NoopSlackGateway(),  # type: ignore[arg-type]
        state_store=_StateStore(state=state),
        runtime_engine=runtime,
        github_gateway=github,  # type: ignore[arg-type]
    )

    next_state = workflow.tick(trigger_reason="manual_tick")

    assert github.trigger_calls == 1
    assert github.trigger_payloads == [
        {
            "workflow_file": "build_rc.yml",
            "ref": "dev",
            "inputs": {},
        }
    ]
    assert next_state.to_dict() == state.to_dict()


def test_active_release_build_rc_does_not_append_version_or_mutate_run_context(tmp_path: Path) -> None:
    state = WorkflowState(
        active_release=ReleaseContext(
            release_version="5.105.0",
            step=ReleaseStep.WAIT_MEETING_CONFIRMATION,
            slack_channel_id="C_RELEASE",
        )
    )
    runtime = _RuntimeStub(
        decision=RuntimeDecision(
            next_step=ReleaseStep.WAIT_MEETING_CONFIRMATION,
            next_state=WorkflowState.from_dict(state.to_dict()),
            tool_calls=[
                {
                    "tool": "github_action",
                    "reason": "manual build rc",
                    "args": {
                        "workflow_file": "build_rc.yml",
                        "ref": "dev",
                        "inputs": {},
                    },
                }
            ],
            audit_reason="manual_build_rc_trigger",
            flow_lifecycle="paused",
        )
    )
    github = _GitHubGatewayStub()
    workflow = ReleaseWorkflow(
        config=_config(tmp_path),
        slack_gateway=_NoopSlackGateway(),  # type: ignore[arg-type]
        state_store=_StateStore(state=state),
        runtime_engine=runtime,
        github_gateway=github,  # type: ignore[arg-type]
    )

    next_state = workflow.tick(trigger_reason="manual_tick")

    assert github.trigger_calls == 1
    assert github.trigger_payloads == [
        {
            "workflow_file": "build_rc.yml",
            "ref": "dev",
            "inputs": {},
        }
    ]
    assert next_state.active_release is not None
    assert next_state.active_release.github_action_run_id is None
    assert next_state.active_release.github_action_status is None
    assert next_state.active_release.github_action_conclusion is None


def test_build_rc_adds_white_check_mark_reaction(tmp_path: Path) -> None:
    state = WorkflowState()
    runtime = _RuntimeStub(
        decision=RuntimeDecision(
            next_step=ReleaseStep.IDLE,
            next_state=WorkflowState.from_dict(state.to_dict()),
            tool_calls=[
                {
                    "tool": "github_action",
                    "reason": "manual build rc",
                    "args": {
                        "workflow_file": "build_rc.yml",
                        "ref": "dev",
                        "inputs": {},
                    },
                }
            ],
            audit_reason="manual_build_rc_trigger",
            flow_lifecycle="running",
        )
    )
    gateway = _EventSlackGateway(
        events=[
            SlackEvent(
                event_id="evt-100",
                event_type="message",
                channel_id="C_RELEASE",
                text="Собери сборку",
                message_ts="1700000000.111111",
            )
        ]
    )
    github = _GitHubGatewayStub()
    workflow = ReleaseWorkflow(
        config=_config(tmp_path),
        slack_gateway=gateway,  # type: ignore[arg-type]
        state_store=_StateStore(state=state),
        runtime_engine=runtime,
        github_gateway=github,  # type: ignore[arg-type]
    )

    workflow.tick(trigger_reason="manual_tick")

    assert github.trigger_calls == 1
    assert gateway.reaction_calls is not None
    assert any(call["emoji"] == "white_check_mark" for call in gateway.reaction_calls)


def test_manual_override_adds_white_check_mark_reaction(tmp_path: Path) -> None:
    state = WorkflowState()
    runtime = _RuntimeStub(
        decision=RuntimeDecision(
            next_step=ReleaseStep.IDLE,
            next_state=WorkflowState.from_dict(state.to_dict()),
            tool_calls=[],
            audit_reason="manual_status_override:5.105.0->IDLE",
            flow_lifecycle="completed",
        )
    )
    gateway = _EventSlackGateway(
        events=[
            SlackEvent(
                event_id="evt-101",
                event_type="message",
                channel_id="C_RELEASE",
                text="Переведи 5.105.0 в IDLE",
                message_ts="1700000000.222222",
            )
        ]
    )
    workflow = ReleaseWorkflow(
        config=_config(tmp_path),
        slack_gateway=gateway,  # type: ignore[arg-type]
        state_store=_StateStore(state=state),
        runtime_engine=runtime,
        github_gateway=_GitHubGatewayStub(),  # type: ignore[arg-type]
    )

    workflow.tick(trigger_reason="manual_tick")

    assert gateway.reaction_calls is not None
    assert any(call["emoji"] == "white_check_mark" for call in gateway.reaction_calls)


def test_active_release_build_rc_adds_white_check_mark_reaction(tmp_path: Path) -> None:
    state = WorkflowState(
        active_release=ReleaseContext(
            release_version="5.105.0",
            step=ReleaseStep.WAIT_MEETING_CONFIRMATION,
            slack_channel_id="C_RELEASE",
        )
    )
    runtime = _RuntimeStub(
        decision=RuntimeDecision(
            next_step=ReleaseStep.WAIT_MEETING_CONFIRMATION,
            next_state=WorkflowState.from_dict(state.to_dict()),
            tool_calls=[
                {
                    "tool": "github_action",
                    "reason": "manual build rc",
                    "args": {
                        "workflow_file": "build_rc.yml",
                        "ref": "dev",
                        "inputs": {},
                    },
                }
            ],
            audit_reason="manual_build_rc_trigger",
            flow_lifecycle="paused",
        )
    )
    gateway = _EventSlackGateway(
        events=[
            SlackEvent(
                event_id="evt-102",
                event_type="message",
                channel_id="C_RELEASE",
                text="Собери сборку",
                message_ts="1700000000.333333",
            )
        ]
    )
    github = _GitHubGatewayStub()
    workflow = ReleaseWorkflow(
        config=_config(tmp_path),
        slack_gateway=gateway,  # type: ignore[arg-type]
        state_store=_StateStore(state=state),
        runtime_engine=runtime,
        github_gateway=github,  # type: ignore[arg-type]
    )

    workflow.tick(trigger_reason="manual_tick")

    assert github.trigger_calls == 1
    assert gateway.reaction_calls is not None
    assert any(call["emoji"] == "white_check_mark" for call in gateway.reaction_calls)


def test_wait_branch_cut_approval_confirmation_updates_text_and_triggers_build_rc(tmp_path: Path) -> None:
    state = WorkflowState(
        active_release=ReleaseContext(
            release_version="5.105.0",
            step=ReleaseStep.WAIT_BRANCH_CUT_APPROVAL,
            slack_channel_id="C_RELEASE",
        )
    )
    decision_state = WorkflowState.from_dict(state.to_dict())
    assert decision_state.active_release is not None
    decision_state.active_release.set_step(ReleaseStep.WAIT_RC_READINESS)
    runtime = _RuntimeStub(
        decision=RuntimeDecision(
            next_step=ReleaseStep.WAIT_RC_READINESS,
            next_state=decision_state,
            tool_calls=[
                {
                    "tool": "slack_update",
                    "reason": "confirm branch readiness",
                    "args": {
                        "channel_id": "C_RELEASE",
                        "message_ts": "1700000000.666666",
                        "text": "Подтверждено :white_check_mark:\nя запустил сборку",
                    },
                },
                {
                    "tool": "github_action",
                    "reason": "trigger build rc",
                    "args": {
                        "workflow_file": "build_rc.yml",
                        "ref": "dev",
                        "inputs": {},
                    },
                },
            ],
            audit_reason="branch_cut_approved",
            flow_lifecycle="paused",
        )
    )
    gateway = _EventSlackGateway(
        events=[
            SlackEvent(
                event_id="evt-branch-approval-1",
                event_type="approval_confirmed",
                channel_id="C_RELEASE",
                text="approve",
                message_ts="1700000000.666666",
                metadata={"trigger_message_text": "Выделил RC ветку. Подтвердите готовность ветки."},
            )
        ]
    )
    github = _GitHubGatewayStub()
    workflow = ReleaseWorkflow(
        config=_config(tmp_path),
        slack_gateway=gateway,  # type: ignore[arg-type]
        state_store=_StateStore(state=state),
        runtime_engine=runtime,
        github_gateway=github,  # type: ignore[arg-type]
    )

    next_state = workflow.tick(trigger_reason="manual_tick")

    assert next_state.active_release is not None
    assert next_state.active_release.step == ReleaseStep.WAIT_RC_READINESS
    assert gateway.updated_payloads == [
        {
            "channel_id": "C_RELEASE",
            "message_ts": "1700000000.666666",
            "text": "Выделил RC ветку. Подтвердите готовность ветки.\n\nПодтверждено :white_check_mark:\nя запустил сборку",
        }
    ]
    assert github.trigger_payloads == [
        {
            "workflow_file": "build_rc.yml",
            "ref": "dev",
            "inputs": {},
        }
    ]


def test_wait_rc_readiness_message_adds_white_check_mark_reaction(tmp_path: Path) -> None:
    state = WorkflowState(
        active_release=ReleaseContext(
            release_version="5.105.0",
            step=ReleaseStep.WAIT_RC_READINESS,
            slack_channel_id="C_RELEASE",
        ),
        flow_execution_id="flow-rc-ready-1",
        flow_paused_at="2026-01-01T00:00:00+00:00",
        pause_reason="awaiting_confirmation:WAIT_RC_READINESS",
    )
    runtime = _RuntimeStub(
        decision=RuntimeDecision(
            next_step=ReleaseStep.IDLE,
            next_state=WorkflowState(active_release=None),
            tool_calls=[
                {
                    "tool": "slack_message",
                    "reason": "celebrate release finish",
                    "args": {
                        "channel_id": "C_RELEASE",
                        "text": "Поздравляем — релиз готов к отправке! :tada: Отличная работа всей команде.",
                    },
                }
            ],
            audit_reason="release_ready_for_send",
            flow_lifecycle="completed",
        )
    )
    gateway = _EventSlackGateway(
        events=[
            SlackEvent(
                event_id="evt-103",
                event_type="message",
                channel_id="C_RELEASE",
                text="Релиз готов к отправке",
                message_ts="1700000000.444444",
            )
        ]
    )
    workflow = ReleaseWorkflow(
        config=_config(tmp_path),
        slack_gateway=gateway,  # type: ignore[arg-type]
        state_store=_StateStore(state=state),
        runtime_engine=runtime,
        github_gateway=_GitHubGatewayStub(),  # type: ignore[arg-type]
    )

    next_state = workflow.tick(trigger_reason="manual_tick")

    assert next_state.step == ReleaseStep.IDLE
    assert next_state.active_release is None
    assert next_state.previous_release_version == "5.105.0"
    assert next_state.previous_release_completed_at is not None
    datetime.fromisoformat(next_state.previous_release_completed_at)
    assert gateway.sent_messages == [
        {
            "channel_id": "C_RELEASE",
            "text": "Поздравляем — релиз готов к отправке! :tada: Отличная работа всей команде.",
            "thread_ts": "",
        }
    ]
    assert gateway.reaction_calls is not None
    assert any(call["emoji"] == "white_check_mark" for call in gateway.reaction_calls)
