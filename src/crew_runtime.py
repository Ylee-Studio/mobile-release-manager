"""Crew runtime coordinator for decision-first workflow execution."""
from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from crewai import Crew, Memory, Process

from .agents import build_orchestrator_agent, build_release_manager_agent
from .observability import CrewRuntimeCallbacks
from .policies import PolicyConfig
from .runtime_contracts import AgentDecisionPayload
from .tasks import build_orchestrator_task, build_release_manager_task
from .tool_calling import extract_native_tool_calls, validate_and_normalize_tool_calls
from .tools.slack_tools import SlackApproveTool, SlackEvent, SlackGateway, SlackMessageTool, SlackUpdateTool
from .workflow_state import ReleaseFlowState, ReleaseStep, WorkflowState

from crewai.flow.flow import Flow, listen, router, start
from crewai.flow.human_feedback import human_feedback

try:  # pragma: no cover - optional persistence import path may vary
    from crewai.flow.persistence import SQLiteFlowPersistence
except Exception:  # pragma: no cover - tests without latest crewai
    SQLiteFlowPersistence = None  # type: ignore[assignment]

try:  # pragma: no cover - optional async feedback control-flow signal
    from crewai.flow.async_feedback.types import HumanFeedbackPending, HumanFeedbackProvider, PendingFeedbackContext
except Exception:  # pragma: no cover
    HumanFeedbackPending = None  # type: ignore[assignment]
    HumanFeedbackProvider = Any  # type: ignore[assignment]
    PendingFeedbackContext = Any  # type: ignore[assignment]

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
    flow_lifecycle: str = "running"

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
            flow_lifecycle=str(payload.get("flow_lifecycle", "running")),
        )


@dataclass
class FlowTurnContext:
    payload: dict[str, Any]
    current_state: WorkflowState
    orchestrator_payload: dict[str, Any]
    orchestrator_native_calls: list[dict[str, Any]]
    orchestrator_decision: CrewDecision


class AsyncHumanFeedbackProvider(HumanFeedbackProvider):
    """Async provider that pauses Flow and waits for external feedback events."""

    def request_feedback(self, context: PendingFeedbackContext, flow: Flow) -> str:  # noqa: ARG002
        if HumanFeedbackPending is None:
            raise RuntimeError("HumanFeedbackPending is unavailable in current CrewAI version.")
        raise HumanFeedbackPending(
            context=context,
            callback_info={
                "flow_id": context.flow_id,
                "method_name": context.method_name,
            },
        )


_ASYNC_HUMAN_FEEDBACK_PROVIDER = AsyncHumanFeedbackProvider()


@persist()
class ReleaseFlow(Flow[ReleaseFlowState]):
    """Release workflow as explicit step-routed CrewAI Flow."""

    def __init__(self, *, coordinator: "CrewRuntimeCoordinator", persistence: Any | None = None) -> None:
        super().__init__(persistence=persistence)
        self.coordinator = coordinator
        self._current_context: FlowTurnContext | None = None

    @start()
    def orchestrator_turn(self) -> FlowTurnContext:
        if not self.state.now_iso:
            self.state.now_iso = datetime.now(timezone.utc).isoformat()
        return self.coordinator._run_flow_orchestrator_turn(
            inputs={
                "state": self.state.state,
                "events": self.state.events,
                "config": self.state.config,
                "now_iso": self.state.now_iso,
                "trigger_reason": self.state.trigger_reason,
            }
        )

    @router(orchestrator_turn)
    def route_release_step(self, context: FlowTurnContext) -> str:
        self._current_context = context
        step = context.orchestrator_decision.next_step
        if step in {
            ReleaseStep.IDLE,
            ReleaseStep.WAIT_START_APPROVAL,
            ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION,
            ReleaseStep.WAIT_MEETING_CONFIRMATION,
            ReleaseStep.WAIT_READINESS_CONFIRMATIONS,
            ReleaseStep.READY_FOR_BRANCH_CUT,
        }:
            return step.value
        return "UNKNOWN_STEP"

    @listen(ReleaseStep.IDLE.value)
    def on_idle(self, _label: str) -> CrewDecision:
        return self.coordinator.handle_step_idle(context=self._context())

    @listen(ReleaseStep.WAIT_START_APPROVAL.value)
    def on_wait_start_approval(self, _label: str) -> CrewDecision:
        decision = self.coordinator.handle_step_wait_start_approval(context=self._context())
        if self._should_pause_gate(decision=decision, gate_step=ReleaseStep.WAIT_START_APPROVAL):
            return self.gate_wait_start_approval(decision)
        return decision

    @listen(ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION.value)
    def on_wait_manual_release_confirmation(self, _label: str) -> CrewDecision:
        decision = self.coordinator.handle_step_wait_manual_release_confirmation(context=self._context())
        if self._should_pause_gate(decision=decision, gate_step=ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION):
            return self.gate_wait_manual_release_confirmation(decision)
        return decision

    @listen(ReleaseStep.WAIT_MEETING_CONFIRMATION.value)
    def on_wait_meeting_confirmation(self, _label: str) -> CrewDecision:
        decision = self.coordinator.handle_step_wait_meeting_confirmation(context=self._context())
        if self._should_pause_gate(decision=decision, gate_step=ReleaseStep.WAIT_MEETING_CONFIRMATION):
            return self.gate_wait_meeting_confirmation(decision)
        return decision

    @listen(ReleaseStep.WAIT_READINESS_CONFIRMATIONS.value)
    def on_wait_readiness_confirmations(self, _label: str) -> CrewDecision:
        return self.coordinator.handle_step_wait_readiness_confirmations(context=self._context())

    @listen(ReleaseStep.READY_FOR_BRANCH_CUT.value)
    def on_ready_for_branch_cut(self, _label: str) -> CrewDecision:
        return self.coordinator.handle_step_ready_for_branch_cut(context=self._context())

    @listen("UNKNOWN_STEP")
    def on_unknown_step(self, _label: str) -> CrewDecision:
        return self.coordinator.handle_step_unknown(context=self._context())

    @human_feedback(
        message="Awaiting release train start confirmation.",
        provider=_ASYNC_HUMAN_FEEDBACK_PROVIDER,
    )
    def gate_wait_start_approval(self, decision: CrewDecision) -> CrewDecision:
        return self._mark_paused(
            decision=decision,
            pause_reason=f"awaiting_confirmation:{ReleaseStep.WAIT_START_APPROVAL.value}",
        )

    @human_feedback(
        message="Awaiting manual Jira release confirmation.",
        provider=_ASYNC_HUMAN_FEEDBACK_PROVIDER,
    )
    def gate_wait_manual_release_confirmation(self, decision: CrewDecision) -> CrewDecision:
        return self._mark_paused(
            decision=decision,
            pause_reason=f"awaiting_confirmation:{ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION.value}",
        )

    @human_feedback(
        message="Awaiting release fixation meeting confirmation.",
        provider=_ASYNC_HUMAN_FEEDBACK_PROVIDER,
    )
    def gate_wait_meeting_confirmation(self, decision: CrewDecision) -> CrewDecision:
        return self._mark_paused(
            decision=decision,
            pause_reason=f"awaiting_confirmation:{ReleaseStep.WAIT_MEETING_CONFIRMATION.value}",
        )

    def _should_pause_gate(self, *, decision: CrewDecision, gate_step: ReleaseStep) -> bool:
        if decision.next_step != gate_step:
            return False
        context = self._context()
        return not _has_resume_event(context.payload.get("events", []))

    @staticmethod
    def _mark_paused(*, decision: CrewDecision, pause_reason: str) -> CrewDecision:
        next_state = decision.next_state
        if not next_state.flow_execution_id:
            next_state.flow_execution_id = str(uuid.uuid4())
        next_state.flow_paused_at = datetime.now(timezone.utc).isoformat()
        next_state.pause_reason = pause_reason
        decision.flow_lifecycle = "paused"
        return decision

    def _context(self) -> FlowTurnContext:
        if self._current_context is None:
            raise RuntimeError("release flow context is unavailable")
        return self._current_context


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
        self._ensure_crewai_storage_dir()
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
        self._orchestrator_memory = self._build_crewai_memory_scope(scope_name="orchestrator")
        self._release_manager_memories: dict[str, Any] = {}
        self._flow_persistence = self._build_flow_persistence()
        self.flow = ReleaseFlow(coordinator=self, persistence=self._flow_persistence)

    def kickoff(
        self,
        *,
        state: WorkflowState,
        events: list[SlackEvent],
        config: Any,
        now: datetime | None = None,
        trigger_reason: str = "heartbeat_timer",
    ) -> CrewDecision:
        if state.is_paused and not _has_resume_event(events):
            return CrewDecision(
                next_step=state.step,
                next_state=state,
                audit_reason=f"flow_paused_waiting_confirmation:{state.pause_reason or 'unknown'}",
                actor="flow_runtime",
                flow_lifecycle="paused",
            )

        inputs = {
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
        }
        try:
            decision = self._flow_kickoff(inputs=inputs)
            self._finalize_flow_metadata(decision=decision, previous_state=state, action="kickoff")
            return decision
        except Exception as exc:  # pragma: no cover - defensive path for runtime robustness
            self.logger.exception("crew flow kickoff failed: %s", exc)
            self._cleanup_release_manager_agents(next_state=state)
            return CrewDecision(
                next_step=state.step,
                next_state=state,
                audit_reason=f"crew_runtime_error:{exc.__class__.__name__}",
                flow_lifecycle="error",
            )

    def _flow_kickoff(self, *, inputs: dict[str, Any]) -> CrewDecision:
        result = self.flow.kickoff(inputs=inputs)
        return self._coerce_flow_output_to_decision(result=result)

    def resume_from_pending(
        self,
        *,
        flow_id: str,
        feedback: str,
        state: WorkflowState,
    ) -> CrewDecision:
        normalized_flow_id = flow_id.strip()
        if not normalized_flow_id:
            return CrewDecision(
                next_step=state.step,
                next_state=state,
                audit_reason="resume_failed_no_pending:empty_flow_id",
                actor="flow_runtime",
                flow_lifecycle="error",
            )
        flow_cls = type(self.flow)
        from_pending = getattr(flow_cls, "from_pending", None)
        if not callable(from_pending):
            return CrewDecision(
                next_step=state.step,
                next_state=state,
                audit_reason="resume_failed_no_pending:from_pending_unavailable",
                actor="flow_runtime",
                flow_lifecycle="error",
            )
        try:
            restored_flow = from_pending(
                normalized_flow_id,
                persistence=self._flow_persistence,
                coordinator=self,
            )
            restored_flow.resume(feedback)
            next_state = WorkflowState.from_dict(state.to_dict())
            next_state.flow_execution_id = normalized_flow_id
            next_state.flow_paused_at = None
            next_state.pause_reason = None
            return CrewDecision(
                next_step=next_state.step,
                next_state=next_state,
                audit_reason="resume_dispatched",
                actor="flow_runtime",
                flow_lifecycle="running",
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            self.logger.exception("crew flow resume failed: %s", exc)
            return CrewDecision(
                next_step=state.step,
                next_state=state,
                audit_reason=f"resume_failed_no_pending:{exc.__class__.__name__}",
                actor="flow_runtime",
                flow_lifecycle="error",
            )

    def _finalize_flow_metadata(
        self,
        *,
        decision: CrewDecision,
        previous_state: WorkflowState,
        action: str,
    ) -> None:
        next_state = decision.next_state
        if action == "kickoff" and not next_state.flow_execution_id:
            next_state.flow_execution_id = str(uuid.uuid4())
        if action == "resume" and previous_state.flow_execution_id:
            next_state.flow_execution_id = previous_state.flow_execution_id

        if decision.flow_lifecycle not in {"paused", "completed"}:
            decision.flow_lifecycle = "running"

        if next_state.active_release is None and decision.next_step == ReleaseStep.IDLE:
            next_state.flow_paused_at = None
            next_state.pause_reason = None
            if decision.flow_lifecycle != "paused":
                decision.flow_lifecycle = "completed"
                next_state.flow_execution_id = None

    def _coerce_flow_output_to_decision(self, *, result: Any) -> CrewDecision:
        if HumanFeedbackPending is not None and isinstance(result, HumanFeedbackPending):
            context = getattr(result, "context", None)
            method_output = getattr(context, "method_output", None)
            if isinstance(method_output, CrewDecision):
                paused_decision = method_output
            else:
                paused_decision = CrewDecision(
                    next_step=ReleaseStep.IDLE,
                    next_state=WorkflowState(),
                    audit_reason="pending_created_without_decision_payload",
                    actor="flow_runtime",
                )
            flow_id = str(getattr(context, "flow_id", "") or "").strip()
            if flow_id:
                paused_decision.next_state.flow_execution_id = flow_id
            paused_decision.next_state.flow_paused_at = datetime.now(timezone.utc).isoformat()
            pause_reason = str(getattr(context, "method_name", "") or "human_feedback_pending")
            paused_decision.next_state.pause_reason = pause_reason
            paused_decision.flow_lifecycle = "paused"
            if not paused_decision.audit_reason:
                paused_decision.audit_reason = "pending_created"
            return paused_decision
        if isinstance(result, CrewDecision):
            return result
        payload = _extract_structured_payload(result)
        current = WorkflowState.from_dict(self.flow.state.state if hasattr(self.flow.state, "state") else {})
        decision = CrewDecision.from_payload(payload, current_state=current)
        decision.tool_calls = self._resolve_tool_calls(
            payload=payload,
            native_tool_calls=extract_native_tool_calls(result),
        )
        return decision

    def _run_flow_orchestrator_turn(self, *, inputs: dict[str, Any]) -> FlowTurnContext:
        state = WorkflowState.from_dict(inputs.get("state", {}))
        events = list(inputs.get("events", []))
        config = inputs.get("config", {})
        trigger_reason = str(inputs.get("trigger_reason", "heartbeat_timer"))
        payload = {
            "state": state.to_dict(),
            "events": events,
            "config": config if isinstance(config, dict) else {},
            "now_iso": str(inputs.get("now_iso") or datetime.now(timezone.utc).isoformat()),
            "trigger_reason": trigger_reason,
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
            if _step_requires_release_manager(orchestrator_decision.next_step):
                invoke_release_manager = True
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

    # Step-specific flow handlers keep ReleaseFlow aligned with explicit ReleaseStep routing.
    def handle_step_idle(self, *, context: FlowTurnContext) -> CrewDecision:
        return self._run_flow_release_manager_turn(context=context)

    def handle_step_wait_start_approval(self, *, context: FlowTurnContext) -> CrewDecision:
        return self._run_flow_release_manager_turn(context=context)

    def handle_step_wait_manual_release_confirmation(self, *, context: FlowTurnContext) -> CrewDecision:
        return self._run_flow_release_manager_turn(context=context)

    def handle_step_wait_meeting_confirmation(self, *, context: FlowTurnContext) -> CrewDecision:
        return self._run_flow_release_manager_turn(context=context)

    def handle_step_wait_readiness_confirmations(self, *, context: FlowTurnContext) -> CrewDecision:
        return self._run_flow_release_manager_turn(context=context)

    def handle_step_ready_for_branch_cut(self, *, context: FlowTurnContext) -> CrewDecision:
        return self._run_flow_release_manager_turn(context=context)

    def handle_step_unknown(self, *, context: FlowTurnContext) -> CrewDecision:
        return self._run_flow_release_manager_turn(context=context)

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
            payload=payload,
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

    def _build_orchestrator_crew(self) -> Any:
        task = build_orchestrator_task(agent=self._orchestrator_agent, slack_tools=self.slack_tools)
        return Crew(
            agents=[self._orchestrator_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=self.policy.agent_verbose,
            tracing=False,
            step_callback=self.callbacks.on_step,
            task_callback=self.callbacks.on_task,
            **self._crew_memory_kwargs(memory_obj=self._orchestrator_memory),
        )

    def _build_release_manager_crew(self, *, release_version: str, agent: Any) -> Any:
        key = release_version.strip() or "unknown-release"
        memory_obj = self._release_manager_memories.get(key)
        if memory_obj is None:
            memory_obj = self._build_crewai_memory_scope(scope_name=f"release_manager_{key}")
            self._release_manager_memories[key] = memory_obj
        task = build_release_manager_task(agent=agent, slack_tools=self.slack_tools)
        return Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=self.policy.agent_verbose,
            tracing=False,
            step_callback=self.callbacks.on_step,
            task_callback=self.callbacks.on_task,
            **self._crew_memory_kwargs(memory_obj=memory_obj),
        )

    def _cleanup_release_manager_agents(self, *, next_state: WorkflowState) -> None:
        active_release = next_state.active_release
        if active_release is None:
            self._release_manager_agents.clear()
            self._release_manager_crews.clear()
            self._release_manager_memories.clear()
            return

        active_version = active_release.release_version.strip()
        if not active_version:
            self._release_manager_agents.clear()
            self._release_manager_crews.clear()
            self._release_manager_memories.clear()
            return

        stale_versions = [version for version in self._release_manager_agents.keys() if version != active_version]
        for version in stale_versions:
            self._release_manager_agents.pop(version, None)
            self._release_manager_crews.pop(version, None)
            self._release_manager_memories.pop(version, None)

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
            payload=payload,
        )
        return _extract_structured_payload(result), extract_native_tool_calls(result)

    def _execute_crew(
        self,
        *,
        crew: Any,
        payload: dict[str, Any],
    ) -> Any:
        if not hasattr(crew, "kickoff"):
            raise RuntimeError("Crew instance is required for runtime execution")
        return crew.kickoff(inputs=payload)

    def _resolve_tool_calls(
        self,
        *,
        payload: dict[str, Any],
        native_tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        source = native_tool_calls if native_tool_calls else payload.get("tool_calls", [])
        return validate_and_normalize_tool_calls(source, schema_by_tool=self.tool_args_schema)

    def _build_crewai_memory_scope(self, *, scope_name: str) -> Any | None:
        try:
            return Memory()
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.warning("failed to init CrewAI memory for scope=%s: %s", scope_name, exc)
            return None

    def _build_flow_persistence(self) -> Any | None:
        if SQLiteFlowPersistence is None:
            return None
        try:
            storage_root = self._memory_storage_root()
            storage_root.mkdir(parents=True, exist_ok=True)
            return SQLiteFlowPersistence(db_path=str(storage_root / "flow_pending.db"))
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.warning("failed to init Flow persistence: %s", exc)
            return None

    def _memory_storage_root(self) -> Path:
        base = Path(self.memory_db_path)
        normalized = base.with_suffix("") if base.suffix else base
        return normalized.parent / f"{normalized.name}_crew_memory"

    def _ensure_crewai_storage_dir(self) -> None:
        if os.getenv("CREWAI_STORAGE_DIR"):
            return
        storage_root = self._memory_storage_root()
        storage_root.mkdir(parents=True, exist_ok=True)
        os.environ["CREWAI_STORAGE_DIR"] = str(storage_root)

    @staticmethod
    def _crew_memory_kwargs(*, memory_obj: Any | None) -> dict[str, Any]:
        if memory_obj is None:
            return {}
        return {"memory": memory_obj}


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


def _step_requires_release_manager(step: ReleaseStep) -> bool:
    return step in {
        ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION,
        ReleaseStep.WAIT_MEETING_CONFIRMATION,
        ReleaseStep.WAIT_READINESS_CONFIRMATIONS,
        ReleaseStep.READY_FOR_BRANCH_CUT,
    }


def _has_resume_event(events: Any) -> bool:
    if not isinstance(events, list):
        return False
    for event in events:
        event_type = ""
        if isinstance(event, dict):
            event_type = str(event.get("event_type", "")).strip()
        else:
            event_type = str(getattr(event, "event_type", "")).strip()
        if event_type == "approval_confirmed":
            return True
    return False
