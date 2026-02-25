"""Crew runtime coordinator for decision-first workflow execution."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from crewai import Crew, Process
from crewai.memory.long_term.long_term_memory import LTMSQLiteStorage, LongTermMemory

from .agents import build_orchestrator_agent, build_release_manager_agent
from .policies import PolicyConfig
from .tasks import build_orchestrator_task, build_release_manager_task
from .tools.jira_tools import CreateCrossspaceReleaseTool, JiraGateway
from .tools.slack_tools import SlackApproveTool, SlackEvent, SlackGateway, SlackMessageTool, SlackUpdateTool
from .workflow_state import ReleaseStep, WorkflowState

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class CrewDecision:
    """Structured result returned by Crew runtime."""

    next_step: ReleaseStep
    next_state: WorkflowState
    state_patch: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    audit_reason: str = "agent_decision"
    actor: str = "orchestrator"

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        current_state: WorkflowState,
        *,
        actor: str = "orchestrator",
    ) -> "CrewDecision":
        next_step_raw = payload.get("next_step", current_state.step.value)
        next_state_raw = payload.get("next_state")
        state_patch = payload.get("state_patch", {})
        tool_calls = payload.get("tool_calls", [])
        audit_reason = str(payload.get("audit_reason", "agent_decision"))

        if isinstance(next_state_raw, dict):
            next_state = WorkflowState.from_dict(next_state_raw)
        elif isinstance(state_patch, dict) and state_patch:
            next_state = _apply_state_patch(current_state, state_patch)
        else:
            next_state = current_state

        try:
            next_step = ReleaseStep(str(next_step_raw))
        except ValueError:
            next_step = next_state.step

        if next_state.active_release and next_state.active_release.step != next_step:
            next_state.active_release.set_step(next_step)
        if next_step == ReleaseStep.IDLE and next_state.active_release is not None:
            next_state.active_release = None

        return cls(
            next_step=next_step,
            next_state=next_state,
            state_patch=state_patch if isinstance(state_patch, dict) else {},
            tool_calls=tool_calls if isinstance(tool_calls, list) else [],
            audit_reason=audit_reason,
            actor=actor,
        )


class CrewRuntimeCoordinator:
    """Build Crew, execute one decision turn and parse output."""

    def __init__(
        self,
        *,
        policy: PolicyConfig,
        slack_gateway: SlackGateway,
        jira_gateway: JiraGateway,
        memory_db_path: str,
    ) -> None:
        self.policy = policy
        self.memory_db_path = memory_db_path
        self.slack_tools = [
            SlackMessageTool(gateway=slack_gateway),
            SlackApproveTool(gateway=slack_gateway),
            SlackUpdateTool(gateway=slack_gateway),
        ]
        self.jira_tool = CreateCrossspaceReleaseTool(gateway=jira_gateway)

    def decide(
        self,
        *,
        state: WorkflowState,
        events: list[SlackEvent],
        config: Any,
        now: datetime | None = None,
    ) -> CrewDecision:
        payload = {
            "state": state.to_dict(),
            "events": [self._event_to_dict(event) for event in events],
            "config": {
                "slack_channel_id": config.slack_channel_id,
                "timezone": config.timezone,
                "jira_project_keys": config.jira_project_keys,
                "readiness_owners": config.readiness_owners,
            },
            "now_iso": (now or datetime.utcnow()).isoformat(),
        }

        try:
            orchestrator_agent = build_orchestrator_agent(policies=self.policy)
            orchestrator_task = build_orchestrator_task(
                agent=orchestrator_agent,
                slack_tools=self.slack_tools,
            )
            orchestrator_payload = self._run_crew(
                agent=orchestrator_agent,
                task=orchestrator_task,
                payload=payload,
            )
            orchestrator_decision = CrewDecision.from_payload(
                orchestrator_payload,
                current_state=state,
                actor="orchestrator",
            )

            invoke_release_manager = _as_bool(orchestrator_payload.get("invoke_release_manager"))
            release_version = (
                orchestrator_decision.next_state.active_release.release_version
                if orchestrator_decision.next_state.active_release
                else ""
            )

            if not invoke_release_manager or not release_version:
                return orchestrator_decision

            release_manager_agent = build_release_manager_agent(
                policies=self.policy,
                release_version=release_version,
            )
            release_manager_task = build_release_manager_task(
                agent=release_manager_agent,
                jira_tool=self.jira_tool,
                slack_tools=self.slack_tools,
            )
            release_manager_payload = self._run_crew(
                agent=release_manager_agent,
                task=release_manager_task,
                payload={
                    **payload,
                    "state": orchestrator_decision.next_state.to_dict(),
                },
            )
            release_manager_decision = CrewDecision.from_payload(
                release_manager_payload,
                current_state=orchestrator_decision.next_state,
                actor="release_manager",
            )
            release_manager_decision.tool_calls = [
                *orchestrator_decision.tool_calls,
                {"tool": "release_manager_spawn", "reason": f"created_for:{release_version}"},
                *release_manager_decision.tool_calls,
            ]
            release_manager_decision.audit_reason = (
                f"{orchestrator_decision.audit_reason};{release_manager_decision.audit_reason}"
            )
            return release_manager_decision
        except Exception as exc:  # pragma: no cover - defensive path for runtime robustness
            return CrewDecision(
                next_step=state.step,
                next_state=state,
                audit_reason=f"crew_runtime_error:{exc.__class__.__name__}",
            )

    @staticmethod
    def _event_to_dict(event: SlackEvent) -> dict[str, Any]:
        return {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "channel_id": event.channel_id,
            "text": event.text,
            "thread_ts": event.thread_ts,
            "message_ts": event.message_ts,
            "metadata": event.metadata,
        }

    def _run_crew(self, *, agent: Any, task: Any, payload: dict[str, Any]) -> dict[str, Any]:
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
            memory=True,
            long_term_memory=LongTermMemory(
                storage=LTMSQLiteStorage(db_path=self.memory_db_path, verbose=False)
            ),
        )
        result = crew.kickoff(inputs=payload)
        return _extract_payload(result)


def _extract_payload(result: Any) -> dict[str, Any]:
    raw = getattr(result, "raw", None)
    if not raw:
        raw = str(result)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = JSON_BLOCK_RE.search(raw)
    if not match:
        raise ValueError("Crew output does not contain JSON payload.")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Crew output JSON payload must be an object.")
    return parsed


def _apply_state_patch(state: WorkflowState, patch: dict[str, Any]) -> WorkflowState:
    raw = state.to_dict()
    merged = _deep_merge(raw, patch)
    return WorkflowState.from_dict(merged)


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if isinstance(value, (int, float)):
        return value != 0
    return False
