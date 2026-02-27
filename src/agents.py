"""Single-agent prompt builder for direct LLM runtime."""
from __future__ import annotations

from .policies import PolicyConfig

_CANONICAL_TOOLS = "slack_message, slack_approve, slack_update"
_TOOL_FORMAT_RULE = (
    f"Use canonical tool names ({_CANONICAL_TOOLS}) without `functions.` prefix, "
    "and ensure each tool call includes `args` matching args_schema."
)


def build_flow_agent(policies: PolicyConfig) -> dict[str, object]:
    """Create single flow agent prompt package for direct SDK call."""
    return {
        "name": "flow_agent",
        "verbose": policies.agent_verbose,
        "instructions": (
            "Return strict JSON with next_step, next_state/state_patch, tool_calls, audit_reason, flow_lifecycle. "
            "You own the full flow from WAIT_START_APPROVAL to READY_FOR_BRANCH_CUT without delegating to other agents. "
            f"{_TOOL_FORMAT_RULE}"
        ),
    }
