from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.release_workflow import ReleaseWorkflow, RuntimeConfig
from src.runtime_engine import RuntimeDecision
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

    def trigger_workflow(self, *, workflow_file: str, ref: str, inputs: dict[str, Any]) -> Any:
        _ = (workflow_file, ref, inputs)
        self.trigger_calls += 1
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
        github_ref="main",
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
    decision_state.active_release.set_step(ReleaseStep.WAIT_BRANCH_CUT)
    runtime = _RuntimeStub(
        decision=RuntimeDecision(
            next_step=ReleaseStep.WAIT_BRANCH_CUT,
            next_state=decision_state,
            tool_calls=[
                {
                    "tool": "github_action",
                    "reason": "dispatch branch cut workflow",
                    "args": {
                        "workflow_file": "create_release_branch.yml",
                        "ref": "main",
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
    assert next_state.active_release.step == ReleaseStep.WAIT_BRANCH_CUT
    assert next_state.active_release.github_action_run_id == 101
    assert next_state.active_release.github_action_status == "queued"


def test_wait_branch_cut_polls_every_30s_and_moves_to_approval(tmp_path: Path) -> None:
    state = WorkflowState(
        active_release=ReleaseContext(
            release_version="5.105.0",
            step=ReleaseStep.WAIT_BRANCH_CUT,
            slack_channel_id="C_RELEASE",
            github_action_run_id=202,
            github_action_status="in_progress",
            github_action_last_polled_at=(datetime.now(timezone.utc) - timedelta(seconds=31)).isoformat(),
        )
    )
    runtime = _RuntimeStub(
        decision=RuntimeDecision(
            next_step=ReleaseStep.WAIT_BRANCH_CUT,
            next_state=WorkflowState.from_dict(state.to_dict()),
            tool_calls=[],
            audit_reason="waiting_for_branch_cut_action",
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

    assert github.poll_calls == 1
    assert next_state.active_release is not None
    assert next_state.active_release.step == ReleaseStep.WAIT_BRANCH_CUT_APPROVAL
    assert next_state.active_release.github_action_status == "completed"
    assert next_state.active_release.github_action_conclusion == "success"
