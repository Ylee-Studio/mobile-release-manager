"""Crew runtime coordinator for decision-first workflow execution."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

try:  # pragma: no cover - tests may run without CrewAI installed
    from crewai import Crew, Process
except Exception:  # pragma: no cover
    Crew = None  # type: ignore[assignment]
    Process = None  # type: ignore[assignment]

from .agents import build_orchestrator_agent, build_release_manager_agent
from .observability import CrewRuntimeCallbacks
from .policies import PolicyConfig
from .runtime_contracts import AgentDecisionPayload
from .tasks import build_orchestrator_task, build_release_manager_task
from .tool_calling import extract_native_tool_calls, validate_and_normalize_tool_calls
from .tools.slack_tools import SlackApproveTool, SlackEvent, SlackGateway, SlackMessageTool, SlackUpdateTool
from .workflow_state import ReleaseStep, WorkflowState

try:  # pragma: no cover - optional crewai flow support
    from crewai.flow.flow import Flow, listen, start
except Exception:  # pragma: no cover - fallback for tests without crewai
    class Flow:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()

    def start() -> Any:  # type: ignore[override]
        def _decorator(func: Any) -> Any:
            return func

        return _decorator

    def listen(_event: Any) -> Any:  # type: ignore[override]
        def _decorator(func: Any) -> Any:
            return func

        return _decorator

try:  # pragma: no cover - optional decorator path varies by crewai version
    from crewai.flow.flow import persist
except Exception:  # pragma: no cover - fallback for tests without crewai
    def persist(*args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        def _decorator(target: Any) -> Any:
            return target

        return _decorator


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


@dataclass
class FlowTurnContext:
    payload: dict[str, Any]
    current_state: WorkflowState
    orchestrator_payload: dict[str, Any]
    orchestrator_native_calls: list[dict[str, Any]]
    orchestrator_decision: CrewDecision


@persist()
class ReleaseDecisionFlow(Flow):
    """CrewAI Flow wrapper for orchestrator/release-manager turn."""

    def __init__(self, *, coordinator: "CrewRuntimeCoordinator") -> None:
        super().__init__()
        self.coordinator = coordinator

    @start()
    def orchestrator_turn(self, inputs: dict[str, Any]) -> FlowTurnContext:
        return self.coordinator._run_flow_orchestrator_turn(inputs=inputs)

    @listen(orchestrator_turn)
    def release_manager_turn(self, context: FlowTurnContext) -> CrewDecision:
        return self.coordinator._run_flow_release_manager_turn(context=context)

    def kickoff(self, *, inputs: dict[str, Any]) -> CrewDecision:
        context = self.orchestrator_turn(inputs)
        return self.release_manager_turn(context)


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
        self.callbacks = CrewRuntimeCallbacks()
        self.tool_args_schema = {tool.name: tool.args_schema for tool in self.slack_tools}
        self._orchestrator_agent = build_orchestrator_agent(policies=self.policy)
        self._release_manager_agents: dict[str, Any] = {}
        self._orchestrator_crew: Any | None = None
        self._release_manager_crews: dict[str, Any] = {}
        self.flow = ReleaseDecisionFlow(coordinator=self)

    def kickoff(
        self,
        *,
        state: WorkflowState,
        events: list[SlackEvent],
        config: Any,
        now: datetime | None = None,
        trigger_reason: str = "heartbeat_timer",
    ) -> CrewDecision:
        inputs = {
            "state": state,
            "events": events,
            "config": config,
            "now": now,
            "trigger_reason": trigger_reason,
        }
        try:
            return self.flow.kickoff(inputs=inputs)
        except Exception as exc:  # pragma: no cover - defensive path for runtime robustness
            self.logger.exception("crew flow kickoff failed: %s", exc)
            self._cleanup_release_manager_agents(next_state=state)
            return CrewDecision(
                next_step=state.step,
                next_state=state,
                audit_reason=f"crew_runtime_error:{exc.__class__.__name__}",
            )

    def decide(
        self,
        *,
        state: WorkflowState,
        events: list[SlackEvent],
        config: Any,
        now: datetime | None = None,
        trigger_reason: str = "heartbeat_timer",
    ) -> CrewDecision:
        """Backward-compatible alias for flow kickoff."""
        return self.kickoff(
            state=state,
            events=events,
            config=config,
            now=now,
            trigger_reason=trigger_reason,
        )

    def _run_flow_orchestrator_turn(self, *, inputs: dict[str, Any]) -> FlowTurnContext:
        state = inputs["state"]
        events = inputs["events"]
        config = inputs["config"]
        now = inputs.get("now")
        trigger_reason = str(inputs.get("trigger_reason", "heartbeat_timer"))
        payload = {
            "state": state.to_dict(),
            "events": [self._event_to_dict(event) for event in events],
            "config": {
                "slack_channel_id": config.slack_channel_id,
                "timezone": config.timezone,
                "jira_project_keys": config.jira_project_keys,
                "readiness_owners": config.readiness_owners,
            },
            "now_iso": (now or datetime.now(timezone.utc)).isoformat(),
            "trigger_reason": trigger_reason,
            "memory_context": {
                "active_step": state.step.value,
                "processed_event_ids": list(state.processed_event_ids),
                "completed_actions_count": len(state.completed_actions),
                "snapshot_backend": "crewai_memory",
            },
        }
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
        return FlowTurnContext(
            payload=payload,
            current_state=state,
            orchestrator_payload=orchestrator_payload,
            orchestrator_native_calls=orchestrator_native_calls,
            orchestrator_decision=orchestrator_decision,
        )

    def _run_flow_release_manager_turn(self, *, context: FlowTurnContext) -> CrewDecision:
        try:
            orchestrator_payload = context.orchestrator_payload
            orchestrator_decision = context.orchestrator_decision
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
                    **context.payload,
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
            self._cleanup_release_manager_agents(next_state=context.current_state)
            return CrewDecision(
                next_step=context.current_state.step,
                next_state=context.current_state,
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
        if self._orchestrator_crew is None:
            self._orchestrator_crew = self._build_orchestrator_crew()
        result = self._execute_crew(
            crew=self._orchestrator_crew,
            agent=self._orchestrator_agent,
            payload=payload,
            task_builder=build_orchestrator_task,
        )
        return _extract_structured_payload(result), extract_native_tool_calls(result)

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

    def _build_orchestrator_crew(self) -> Any | None:
        if Crew is None or Process is None:
            return None
        task = build_orchestrator_task(agent=self._orchestrator_agent, slack_tools=self.slack_tools)
        return Crew(
            agents=[self._orchestrator_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=self.policy.agent_verbose,
            tracing=False,
            step_callback=self.callbacks.on_step,
            task_callback=self.callbacks.on_task,
        )

    def _build_release_manager_crew(self, *, release_version: str, agent: Any) -> Any | None:
        if Crew is None or Process is None:
            return None
        _ = release_version
        task = build_release_manager_task(agent=agent, slack_tools=self.slack_tools)
        return Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=self.policy.agent_verbose,
            tracing=False,
            step_callback=self.callbacks.on_step,
            task_callback=self.callbacks.on_task,
        )

    def _cleanup_release_manager_agents(self, *, next_state: WorkflowState) -> None:
        active_release = next_state.active_release
        if active_release is None:
            self._release_manager_agents.clear()
            self._release_manager_crews.clear()
            return

        active_version = active_release.release_version.strip()
        if not active_version:
            self._release_manager_agents.clear()
            self._release_manager_crews.clear()
            return

        stale_versions = [version for version in self._release_manager_agents.keys() if version != active_version]
        for version in stale_versions:
            self._release_manager_agents.pop(version, None)
            self._release_manager_crews.pop(version, None)

    def _run_release_manager(
        self,
        *,
        agent: Any,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        active_release = payload.get("state", {}).get("active_release", {})
        release_version = str(active_release.get("release_version", "")).strip()
        if not release_version:
            release_version = "unknown-release"
        crew = self._release_manager_crews.get(release_version)
        if crew is None:
            crew = self._build_release_manager_crew(release_version=release_version, agent=agent)
            self._release_manager_crews[release_version] = crew
        result = self._execute_crew(
            crew=crew,
            agent=agent,
            payload=payload,
            task_builder=build_release_manager_task,
        )
        return _extract_structured_payload(result), extract_native_tool_calls(result)

    def _execute_crew(
        self,
        *,
        crew: Any | None,
        agent: Any,
        payload: dict[str, Any],
        task_builder: Any,
    ) -> Any:
        if crew is not None and hasattr(crew, "kickoff"):
            return crew.kickoff(inputs=payload)

        # Compatibility path for tests and environments where Crew is unavailable.
        task = task_builder(
            agent=agent,
            slack_tools=self.slack_tools,
        )
        interpolator = getattr(task, "interpolate_inputs_and_add_conversation_history", None)
        if callable(interpolator):
            interpolator(payload)
        return agent.execute_task(
            task=task,
            tools=self.slack_tools,
        )

    def _resolve_tool_calls(
        self,
        *,
        payload: dict[str, Any],
        native_tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        source = native_tool_calls if native_tool_calls else payload.get("tool_calls", [])
        return validate_and_normalize_tool_calls(source, schema_by_tool=self.tool_args_schema)


def _extract_structured_payload(result: Any) -> dict[str, Any]:
    candidate = _extract_payload_candidate(result)
    if isinstance(candidate, AgentDecisionPayload):
        return candidate.to_payload()
    if isinstance(candidate, dict):
        return AgentDecisionPayload.model_validate(candidate).to_payload()
    if isinstance(candidate, str):
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return AgentDecisionPayload.model_validate(parsed).to_payload()
    raise ValueError("Crew output does not contain structured decision payload.")


def _extract_payload_candidate(result: Any) -> Any:
    pydantic_payload = _extract_attr_or_key(result, "pydantic")
    if pydantic_payload is not None:
        return pydantic_payload
    json_payload = _extract_attr_or_key(result, "json_dict")
    if isinstance(json_payload, dict):
        return json_payload

    tasks_output = _extract_attr_or_key(result, "tasks_output")
    if isinstance(tasks_output, list) and tasks_output:
        last_task = tasks_output[-1]
        task_pydantic = _extract_attr_or_key(last_task, "pydantic")
        if task_pydantic is not None:
            return task_pydantic
        task_json = _extract_attr_or_key(last_task, "json_dict")
        if isinstance(task_json, dict):
            return task_json
        task_raw = _extract_attr_or_key(last_task, "raw")
        if isinstance(task_raw, str) and task_raw.strip():
            return task_raw

    raw = _extract_attr_or_key(result, "raw")
    if isinstance(raw, str) and raw.strip():
        return raw
    return result if isinstance(result, dict) else None


def _extract_attr_or_key(container: Any, key: str) -> Any:
    if isinstance(container, dict):
        return container.get(key)
    return getattr(container, key, None)


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
