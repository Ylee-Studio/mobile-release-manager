"""Structured contracts for Crew runtime outputs."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolCallPayload(BaseModel):
    tool: str
    reason: str = ""
    args: dict[str, Any] = Field(default_factory=dict)


class AgentDecisionPayload(BaseModel):
    next_step: str
    next_state: dict[str, Any] | None = None
    state_patch: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[ToolCallPayload] = Field(default_factory=list)
    audit_reason: str = "agent_decision"
    flow_lifecycle: str = "running"

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)
