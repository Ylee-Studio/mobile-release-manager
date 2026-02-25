"""CrewAI task definitions for orchestrated runtime."""
from __future__ import annotations

from crewai import Task


def build_orchestrator_task(agent, slack_tools):
    """Return orchestrator task that controls workflow routing."""
    return Task(
        description=(
            "You receive `state`, `events`, `config`, `now_iso` as inputs. "
            "Read workflow context, choose next release step, and decide whether "
            "Release Manager must be invoked for release-specific actions."
            "\n"
            "Required output JSON schema:\n"
            "{\n"
            '  "next_step": "IDLE|WAIT_START_APPROVAL|RELEASE_MANAGER_CREATED|'
            'JIRA_RELEASE_CREATED|WAIT_MEETING_CONFIRMATION|'
            'WAIT_READINESS_CONFIRMATIONS|READY_FOR_BRANCH_CUT",\n'
            '  "next_state": { ... WorkflowState dict ... },\n'
            '  "tool_calls": [{"tool": "name", "reason": "why"}],\n'
            '  "audit_reason": "short explanation",\n'
            '  "invoke_release_manager": true|false\n'
            "}\n"
            "Set invoke_release_manager=true only when active release exists and "
            "release manager should continue execution. "
            "Use `next_state` as the complete state snapshot for persistence."
        ),
        expected_output=(
            "Valid JSON object with next_step, next_state, tool_calls, audit_reason, invoke_release_manager."
        ),
        tools=[*slack_tools],
        agent=agent,
    )


def build_release_manager_task(agent, jira_tool, slack_tools):
    """Return release-manager task that executes release side effects."""
    return Task(
        description=(
            "You receive `state`, `events`, `config`, `now_iso` as inputs. "
            "Use active release context only, execute Jira/Slack side effects for "
            "that release, and return the updated workflow state."
            "\n"
            "Required output JSON schema:\n"
            "{\n"
            '  "next_step": "IDLE|WAIT_START_APPROVAL|RELEASE_MANAGER_CREATED|'
            'JIRA_RELEASE_CREATED|WAIT_MEETING_CONFIRMATION|'
            'WAIT_READINESS_CONFIRMATIONS|READY_FOR_BRANCH_CUT",\n'
            '  "next_state": { ... WorkflowState dict ... },\n'
            '  "tool_calls": [{"tool": "name", "reason": "why"}],\n'
            '  "audit_reason": "short explanation"\n'
            "}\n"
            "Use `next_state` as the complete state snapshot for persistence. "
            "If no change is needed, return the same state with matching next_step."
        ),
        expected_output=(
            "Valid JSON object with next_step, next_state, tool_calls, audit_reason."
        ),
        tools=[*slack_tools, jira_tool],
        agent=agent,
    )
