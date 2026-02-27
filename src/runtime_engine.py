"""Deterministic runtime coordinator with Pydantic AI integration."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from .agents import build_flow_agent
from .policies import PolicyConfig
from .runtime_contracts import AgentDecisionPayload, ToolCallPayload
from .tasks import build_flow_task
from .tool_calling import extract_native_tool_calls, validate_and_normalize_tool_calls
from .tools.slack_tools import SlackApproveInput, SlackEvent, SlackGateway, SlackMessageInput, SlackUpdateInput
from .workflow_state import (
    ReleaseStep,
    WorkflowState,
)

class RuntimeDecision(BaseModel):
    """Runtime decision contract."""

    next_step: ReleaseStep
    next_state: WorkflowState
    state_patch: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    audit_reason: str = "agent_decision"
    actor: str = "flow_agent"
    flow_lifecycle: str = "running"

    def to_dict(self) -> dict[str, Any]:
        return {
            "next_step": self.next_step.value,
            "next_state": self.next_state.to_dict(),
            "state_patch": self.state_patch,
            "tool_calls": self.tool_calls,
            "audit_reason": self.audit_reason,
            "actor": self.actor,
            "flow_lifecycle": self.flow_lifecycle,
        }

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any], *, fallback_state: WorkflowState | None = None
    ) -> "RuntimeDecision":
        current_state = fallback_state or WorkflowState()
        if not isinstance(raw, dict):
            return cls(
                next_step=current_state.step,
                next_state=current_state,
                audit_reason="invalid_decision_payload",
                actor="runtime_engine",
                flow_lifecycle="error",
            )
        return cls.from_payload(
            {
                "next_step": raw.get("next_step", current_state.step.value),
                "next_state": raw.get("next_state", current_state.to_dict()),
                "state_patch": raw.get("state_patch", {}),
                "tool_calls": raw.get("tool_calls", []),
                "audit_reason": raw.get("audit_reason", "agent_decision"),
                "actor": raw.get("actor", "flow_agent"),
                "flow_lifecycle": raw.get("flow_lifecycle", "running"),
            },
            current_state=current_state,
            actor=str(raw.get("actor", "flow_agent")),
        )

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        current_state: WorkflowState,
        *,
        actor: str = "flow_agent",
    ) -> "RuntimeDecision":
        next_step_raw = payload.get("next_step", current_state.step.value)
        try:
            resolved_next_step = ReleaseStep(str(next_step_raw))
        except ValueError:
            resolved_next_step = current_state.step

        next_state_raw = payload.get("next_state")
        state_patch = payload.get("state_patch", {})
        tool_calls = payload.get("tool_calls", [])
        if isinstance(next_state_raw, dict):
            next_state_raw = _normalize_active_release_container(
                container=next_state_raw,
                next_step=resolved_next_step,
                current_state=current_state,
            )
        if isinstance(state_patch, dict):
            state_patch = _normalize_active_release_container(
                container=state_patch,
                next_step=resolved_next_step,
                current_state=current_state,
            )
        if isinstance(next_state_raw, dict):
            next_state = WorkflowState.from_dict(next_state_raw)
        elif isinstance(state_patch, dict) and state_patch:
            next_state = _apply_state_patch(current_state, state_patch)
        else:
            next_state = current_state

        if next_state.active_release and next_state.active_release.step != resolved_next_step:
            next_state.active_release.set_step(resolved_next_step)
        if resolved_next_step == ReleaseStep.IDLE and next_state.active_release is not None:
            next_state.active_release = None

        return cls(
            next_step=resolved_next_step,
            next_state=next_state,
            state_patch=state_patch if isinstance(state_patch, dict) else {},
            tool_calls=tool_calls if isinstance(tool_calls, list) else [],
            audit_reason=str(payload.get("audit_reason", "agent_decision")),
            actor=actor,
            flow_lifecycle=str(payload.get("flow_lifecycle", "running")),
        )


class RuntimeCoordinator:
    """Runtime engine with deterministic fallback and optional direct LLM call."""

    def __init__(
        self,
        *,
        policy: PolicyConfig,
        slack_gateway: SlackGateway,
        memory_db_path: str,
    ) -> None:
        self.policy = policy
        self.memory_db_path = memory_db_path
        self.logger = logging.getLogger("runtime_engine")
        self.tool_args_schema = {
            "slack_message": SlackMessageInput,
            "slack_approve": SlackApproveInput,
            "slack_update": SlackUpdateInput,
        }
        self._llm_runtime = DirectLLMRuntime(policy=policy)
        self._agent = build_flow_agent(policy)
        self._task = build_flow_task(self._agent)
        self.flow = _CompatFlow(self)

    def kickoff(
        self,
        *,
        state: WorkflowState,
        events: list[SlackEvent],
        config: Any,
        now: datetime | None = None,
        trigger_reason: str = "event_trigger",
    ) -> RuntimeDecision:
        payload = {
            "state": state.to_dict(),
            "events": [self._event_to_dict(event) for event in events],
            "config": {
                "slack_channel_id": getattr(config, "slack_channel_id", ""),
                "timezone": getattr(config, "timezone", "UTC"),
                "jira_project_keys": getattr(config, "jira_project_keys", []),
                "readiness_owners": getattr(config, "readiness_owners", {}),
            },
            "now_iso": (now or datetime.now(timezone.utc)).isoformat(),
            "trigger_reason": trigger_reason,
        }
        try:
            result = self.flow.kickoff(inputs=payload)
            if isinstance(result, RuntimeDecision):
                decision = result
                self._finalize_flow_metadata(decision=decision, previous_state=state)
                return decision
            parsed_payload = _extract_structured_payload(result)
            decision = RuntimeDecision.from_payload(
                parsed_payload, current_state=state, actor="flow_agent"
            )
            decision.tool_calls = self._resolve_tool_calls(
                payload=parsed_payload,
                native_tool_calls=extract_native_tool_calls(result),
            )
            self._finalize_flow_metadata(decision=decision, previous_state=state)
            return decision
        except Exception as exc:  # pragma: no cover - defensive path for runtime robustness
            self.logger.exception("runtime decide failed: %s", exc)
            return RuntimeDecision(
                next_step=state.step,
                next_state=state,
                audit_reason=f"runtime_error:{exc.__class__.__name__}",
                tool_calls=[],
                actor="flow_runtime",
                flow_lifecycle="error",
            )

    def resume_from_pending(
        self,
        *,
        flow_id: str,
        feedback: str,
        state: WorkflowState,
    ) -> RuntimeDecision:
        normalized_flow_id = flow_id.strip()
        if not normalized_flow_id:
            return RuntimeDecision(
                next_step=state.step,
                next_state=state,
                audit_reason="resume_failed_no_pending:empty_flow_id",
                actor="flow_runtime",
                flow_lifecycle="error",
            )
        try:
            flow_cls = type(self.flow)
            from_pending = getattr(flow_cls, "from_pending", None)
            if callable(from_pending):
                restored_flow = from_pending(normalized_flow_id, coordinator=self)
                restored_flow.resume(feedback)
            next_state = WorkflowState.from_dict(state.to_dict())
            next_state.flow_execution_id = normalized_flow_id
            next_state.flow_paused_at = None
            next_state.pause_reason = None
            return RuntimeDecision(
                next_step=next_state.step,
                next_state=next_state,
                audit_reason="resume_dispatched",
                actor="flow_runtime",
                flow_lifecycle="running",
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            self.logger.exception("runtime flow resume failed: %s", exc)
            return RuntimeDecision(
                next_step=state.step,
                next_state=state,
                audit_reason=f"resume_failed:{exc.__class__.__name__}",
                actor="flow_runtime",
                flow_lifecycle="error",
            )

    def _finalize_flow_metadata(
        self,
        *,
        decision: RuntimeDecision,
        previous_state: WorkflowState,
    ) -> None:
        next_state = decision.next_state
        if not next_state.flow_execution_id:
            next_state.flow_execution_id = previous_state.flow_execution_id or str(uuid.uuid4())
        if decision.flow_lifecycle not in {"paused", "completed"}:
            decision.flow_lifecycle = "running"
        if next_state.active_release is None and decision.next_step == ReleaseStep.IDLE:
            next_state.flow_paused_at = None
            next_state.pause_reason = None
            if decision.flow_lifecycle != "paused":
                decision.flow_lifecycle = "completed"
                next_state.flow_execution_id = None

    def _resolve_tool_calls(
        self,
        *,
        payload: dict[str, Any],
        native_tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        payload_tool_calls = payload.get("tool_calls", [])
        source = payload_tool_calls if payload_tool_calls else native_tool_calls
        return validate_and_normalize_tool_calls(source, schema_by_tool=self.tool_args_schema)

    def _execute_runtime(self, *, payload: dict[str, Any]) -> Any:
        return self._llm_runtime.decide_payload(payload=payload, task=self._task)

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


class AgentDecision(BaseModel):
    """Structured output from Pydantic AI agent.

    This model defines the expected response structure from the LLM.
    Pydantic AI automatically validates the response against this schema.

    Note: next_step is str (not ReleaseStep) because Pydantic AI expects
    simple types for result_type validation. We convert to ReleaseStep later.
    """
    next_step: str
    next_state: dict[str, Any] = Field(default_factory=dict)
    state_patch: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[ToolCallPayload] = Field(default_factory=list)
    audit_reason: str = "agent_decision"
    flow_lifecycle: str = "running"


class DirectLLMRuntime:
    """Direct decision runtime using Pydantic AI for LLM calls."""

    def __init__(self, *, policy: PolicyConfig) -> None:
        self.policy = policy
        self._agent = self._build_agent()

    def _build_agent(self) -> Agent:
        """Build Pydantic AI agent with structured output."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        # OpenAIModel uses provider to pass api_key explicitly
        provider = OpenAIProvider(api_key=api_key)
        model = OpenAIModel(model_name, provider=provider)

        return Agent(
            model=model,
            output_type=AgentDecision,
            system_prompt="You are a release workflow orchestrator. Make decisions about workflow state transitions.",
            retries=3,  # Auto-retry on validation errors
        )

    def _build_user_prompt(
        self,
        *,
        payload: dict[str, Any],
        task: dict[str, Any] | None = None,
    ) -> str:
        """Build user prompt from payload and task description."""
        task = task or {}
        description = str(task.get("description", ""))

        # Build context section with JSON-serialized values
        context_parts = [
            "## Input Context",
            "",
            f"state: {json.dumps(payload.get('state', {}), ensure_ascii=False)}",
            f"events: {json.dumps(payload.get('events', []), ensure_ascii=False)}",
            f"config: {json.dumps(payload.get('config', {}), ensure_ascii=False)}",
            f"now_iso: {payload.get('now_iso', '')}",
            f"trigger_reason: {payload.get('trigger_reason', 'event_trigger')}",
            "",
        ]
        context = "\n".join(context_parts)

        # Combine context with task description
        return context + description

    def _run_agent_sync(
        self,
        payload: dict[str, Any],
        task: dict[str, Any] | None = None,
    ) -> AgentDecision:
        """Run Pydantic AI agent synchronously.

        Uses get_event_loop().run_until_complete() to safely handle
        both sync and async contexts without creating nested event loops.
        """
        import nest_asyncio
        nest_asyncio.apply()

        prompt = self._build_user_prompt(payload=payload, task=task)

        # Get or create event loop and run async agent
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, use nest_asyncio
                result = asyncio.run(self._agent.run(prompt))
            else:
                result = loop.run_until_complete(self._agent.run(prompt))
        except RuntimeError:
            # No event loop, create one
            result = asyncio.run(self._agent.run(prompt))

        return result.output

    def decide_payload(self, *, payload: dict[str, Any], task: dict[str, Any] | None = None) -> dict[str, Any]:
        """Main entry point: call Pydantic AI agent and return decision."""
        current_state = WorkflowState.from_dict(payload.get("state", {}))

        try:
            # Get structured decision from Pydantic AI
            agent_decision = self._run_agent_sync(payload, task)

            # Convert tool_calls from ToolCallPayload objects to dicts
            tool_calls_dicts = [tc.model_dump() for tc in agent_decision.tool_calls]

            # Convert AgentDecision to RuntimeDecision
            # next_step is already a string, no need for .value
            decision = RuntimeDecision.from_payload(
                {
                    "next_step": agent_decision.next_step,
                    "next_state": agent_decision.next_state or current_state.to_dict(),
                    "state_patch": agent_decision.state_patch,
                    "tool_calls": tool_calls_dicts,
                    "audit_reason": agent_decision.audit_reason,
                    "flow_lifecycle": agent_decision.flow_lifecycle,
                },
                current_state=current_state,
                actor="flow_agent",
            )
            return decision.to_dict()

        except Exception as exc:
            logging.getLogger("runtime_engine").exception("Pydantic AI agent failed: %s", exc)
            return RuntimeDecision(
                next_step=current_state.step,
                next_state=current_state,
                audit_reason=f"agent_error:{exc.__class__.__name__}",
                actor="flow_agent",
                flow_lifecycle="error",
            ).to_dict()


class _CompatFlow:
    """Compatibility layer for tests expecting flow-like API."""

    def __init__(self, coordinator: RuntimeCoordinator):
        self.coordinator = coordinator

    def kickoff(self, *, inputs: dict[str, Any]) -> Any:
        return self.coordinator._execute_runtime(payload=inputs)

    def resume(self, feedback: str = "") -> str:
        _ = feedback
        return "ok"

    @classmethod
    def from_pending(cls, flow_id: str, **kwargs: Any) -> "_CompatFlow":
        _ = flow_id
        coordinator = kwargs.get("coordinator")
        if isinstance(coordinator, RuntimeCoordinator):
            return cls(coordinator)
        raise RuntimeError("coordinator is required")


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
    raise ValueError("Runtime output does not contain structured decision payload.")


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


def _normalize_active_release_container(
    *,
    container: dict[str, Any],
    next_step: ReleaseStep,
    current_state: WorkflowState,
) -> dict[str, Any]:
    normalized = dict(container)
    active_release_patch = normalized.get("active_release")

    # Some agent outputs send active_release as a plain version string.
    # Convert it to the expected object shape so WorkflowState.from_dict keeps release context.
    if isinstance(active_release_patch, str):
        release_version = active_release_patch.strip()
        if not release_version:
            normalized["active_release"] = None
            return normalized

        previous_active = current_state.active_release
        previous_message_ts = dict(previous_active.message_ts) if previous_active else {}
        previous_thread_ts = dict(previous_active.thread_ts) if previous_active else {}
        previous_readiness = dict(previous_active.readiness_map) if previous_active else {}
        previous_channel_id = previous_active.slack_channel_id if previous_active else None
        normalized["active_release"] = {
            "release_version": release_version,
            "step": next_step.value,
            "message_ts": previous_message_ts,
            "thread_ts": previous_thread_ts,
            "readiness_map": previous_readiness,
            "slack_channel_id": previous_channel_id,
        }

    return normalized


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


