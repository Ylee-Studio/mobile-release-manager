"""Crew runtime coordinator for decision-first workflow execution."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .agents import build_orchestrator_agent, build_release_manager_agent
from .policies import PolicyConfig
from .tasks import build_orchestrator_task, build_release_manager_task
from .tool_calling import extract_native_tool_calls, validate_and_normalize_tool_calls
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
        memory_db_path: str,
    ) -> None:
        self.policy = policy
        self.memory_db_path = memory_db_path
        self.slack_tools = [
            SlackMessageTool(gateway=slack_gateway),
            SlackApproveTool(gateway=slack_gateway),
            SlackUpdateTool(gateway=slack_gateway),
        ]
        self.logger = logging.getLogger("crew_runtime")
        self.tool_args_schema = {tool.name: tool.args_schema for tool in self.slack_tools}
        self._orchestrator_agent = build_orchestrator_agent(policies=self.policy)
        self._release_manager_agents: dict[str, Any] = {}

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
            orchestrator_payload, orchestrator_native_calls = self._run_orchestrator(payload=payload)
            orchestrator_decision = CrewDecision.from_payload(
                orchestrator_payload,
                current_state=state,
                actor="orchestrator",
            )
            orchestrator_decision.tool_calls = self._resolve_tool_calls(
                payload=orchestrator_payload,
                native_tool_calls=orchestrator_native_calls,
            )

            invoke_release_manager = _as_bool(orchestrator_payload.get("invoke_release_manager"))
            release_version = (
                orchestrator_decision.next_state.active_release.release_version
                if orchestrator_decision.next_state.active_release
                else ""
            )

            if not invoke_release_manager or not release_version:
                self._cleanup_release_manager_agents(next_state=orchestrator_decision.next_state)
                return orchestrator_decision

            release_manager_agent = self._get_or_create_release_manager_agent(release_version=release_version)
            release_manager_payload, release_manager_native_calls = self._run_release_manager(
                agent=release_manager_agent,
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
            release_manager_decision.tool_calls = self._resolve_tool_calls(
                payload=release_manager_payload,
                native_tool_calls=release_manager_native_calls,
            )
            release_manager_decision.tool_calls = [
                *orchestrator_decision.tool_calls,
                *release_manager_decision.tool_calls,
            ]
            release_manager_decision.audit_reason = (
                f"{orchestrator_decision.audit_reason};{release_manager_decision.audit_reason}"
            )
            self._cleanup_release_manager_agents(next_state=release_manager_decision.next_state)
            return release_manager_decision
        except Exception as exc:  # pragma: no cover - defensive path for runtime robustness
            self.logger.exception("crew runtime decide failed: %s", exc)
            self._cleanup_release_manager_agents(next_state=state)
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

    def _run_orchestrator(self, *, payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        orchestrator_task = build_orchestrator_task(
            agent=self._orchestrator_agent,
            slack_tools=self.slack_tools,
        )
        orchestrator_task.interpolate_inputs_and_add_conversation_history(payload)
        result = self._orchestrator_agent.execute_task(
            task=orchestrator_task,
            tools=self.slack_tools,
        )
        return _extract_payload(result), extract_native_tool_calls(result)

    def _get_or_create_release_manager_agent(self, *, release_version: str) -> Any:
        key = release_version.strip()
        if key in self._release_manager_agents:
            return self._release_manager_agents[key]

        release_manager_agent = build_release_manager_agent(
            policies=self.policy,
            release_version=key,
        )
        self._release_manager_agents[key] = release_manager_agent
        return release_manager_agent

    def _cleanup_release_manager_agents(self, *, next_state: WorkflowState) -> None:
        active_release = next_state.active_release
        if active_release is None:
            self._release_manager_agents.clear()
            return

        active_version = active_release.release_version.strip()
        if not active_version:
            self._release_manager_agents.clear()
            return

        stale_versions = [
            version for version in self._release_manager_agents.keys() if version != active_version
        ]
        for version in stale_versions:
            self._release_manager_agents.pop(version, None)

    def _run_release_manager(
        self,
        *,
        agent: Any,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        task = build_release_manager_task(
            agent=agent,
            slack_tools=self.slack_tools,
        )
        task.interpolate_inputs_and_add_conversation_history(payload)
        result = agent.execute_task(
            task=task,
            tools=self.slack_tools,
        )
        return _extract_payload(result), extract_native_tool_calls(result)

    def _resolve_tool_calls(
        self,
        *,
        payload: dict[str, Any],
        native_tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        source = native_tool_calls if native_tool_calls else payload.get("tool_calls", [])
        return validate_and_normalize_tool_calls(source, schema_by_tool=self.tool_args_schema)


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
