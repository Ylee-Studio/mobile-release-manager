"""CrewAI task definitions for orchestrated runtime."""
from __future__ import annotations

from crewai import Task

from .runtime_contracts import AgentDecisionPayload

_RELEASE_STEP_VALUES = (
    "IDLE|WAIT_START_APPROVAL|"
    "WAIT_MANUAL_RELEASE_CONFIRMATION|WAIT_MEETING_CONFIRMATION|"
    "WAIT_READINESS_CONFIRMATIONS|READY_FOR_BRANCH_CUT"
)
_TOOL_NAMES = "slack_message, slack_approve, slack_update"
_TOOL_EXAMPLES = (
    "{tool:'slack_message',args:{channel_id:'C123',text:'ok',thread_ts:'177.1'}}; "
    "{tool:'slack_approve',args:{channel_id:'C123',text:'approve?',approve_label:'Подтвердить'}}; "
    "{tool:'slack_update',args:{channel_id:'C123',message_ts:'177.1',text:'done'}}."
)
_COMMON_SCHEMA_DEFS = (
    f"Schema definitions:\n"
    f"- ReleaseStep := {_RELEASE_STEP_VALUES}\n"
    '- FlowLifecycle := "running" | "paused" | "completed"\n'
    '- ToolCall := {"tool":"name","reason":"why","args":{...}}\n'
)
_COMMON_FORMAT_RULES = (
    f"Use only canonical tool names: {_TOOL_NAMES} "
    "(never use functions.* aliases). "
    f"Tool args must match args_schema exactly. Valid examples: {_TOOL_EXAMPLES}"
)


def build_orchestrator_task(agent, slack_tools):
    """Return orchestrator task that controls workflow routing."""
    return Task(
        description=(
            "You receive `state`, `events`, `config`, `now_iso`, `trigger_reason` as inputs. "
            "Read workflow context, choose next release step, and decide whether "
            "Release Manager must be invoked for release-specific actions."
            "\n"
            "Input context (authoritative):\n"
            "state={state}\n"
            "events={events}\n"
            "config={config}\n"
            "now_iso={now_iso}\n"
            "trigger_reason={trigger_reason}\n"
            "\n"
            f"{_COMMON_SCHEMA_DEFS}"
            "Required output JSON schema:\n"
            "{\n"
            '  "next_step": "<ReleaseStep>",\n'
            '  "next_state": { ... WorkflowState dict ... },\n'
            '  "tool_calls": [<ToolCall>],\n'
            '  "audit_reason": "short explanation",\n'
            '  "invoke_release_manager": true|false,\n'
            '  "flow_lifecycle": "running|paused|completed"\n'
            "}\n"
            "Behavior requirements:\n"
            "- Orchestrator is responsible for routing and state transitions.\n"
            "- For human-confirmation stages, keep next_step at WAIT_* and "
            "flow_lifecycle='paused' until approval_confirmed event arrives.\n"
            "- Keep state flow metadata consistent: set flow_execution_id for active flow, "
            "set flow_paused_at/pause_reason only when flow_lifecycle='paused', "
            "and clear pause fields after resume.\n"
            "- Resume semantics: approval_confirmed events are processed only after runtime "
            "resumes the pending flow via flow_execution_id in workflow state.\n"
            "- For release-manager phases (WAIT_MANUAL_RELEASE_CONFIRMATION, "
            "WAIT_MEETING_CONFIRMATION, "
            "WAIT_READINESS_CONFIRMATIONS, "
            "READY_FOR_BRANCH_CUT), prefer delegating release-specific Slack side effects "
            "to Release Manager to avoid duplicate actions.\n"
            "Set invoke_release_manager=true only when active release exists and "
            "release manager should continue execution. "
            "If next_step is one of WAIT_MANUAL_RELEASE_CONFIRMATION, "
            "WAIT_MEETING_CONFIRMATION, WAIT_READINESS_CONFIRMATIONS, "
            "or READY_FOR_BRANCH_CUT, set invoke_release_manager=true. "
            "Use `next_state` as the complete state snapshot for persistence. "
            f"{_COMMON_FORMAT_RULES}"
        ),
        expected_output=(
            "Valid JSON object with next_step, next_state, tool_calls, audit_reason, invoke_release_manager, flow_lifecycle."
        ),
        output_pydantic=AgentDecisionPayload,
        tools=[*slack_tools],
        agent=agent,
    )


def build_release_manager_task(agent, slack_tools):
    """Return release-manager task that executes release side effects."""
    return Task(
        description=(
            "You receive `state`, `events`, `config`, `now_iso`, `trigger_reason` as inputs. "
            "Use active release context only, execute Slack side effects for "
            "that release, and return the updated workflow state."
            "\n"
            "Input context (authoritative):\n"
            "state={state}\n"
            "events={events}\n"
            "config={config}\n"
            "now_iso={now_iso}\n"
            "trigger_reason={trigger_reason}\n"
            "\n"
            "Behavior requirements:\n"
            "- Do NOT auto-create Jira releases.\n"
            "- Return flow_lifecycle='paused' while waiting for human confirmation, "
            "flow_lifecycle='running' after approval_confirmed transitions, and "
            "flow_lifecycle='completed' only when flow is truly finished.\n"
            "- Ensure all confirmation gates are compatible with pending-flow recovery via flow_execution_id.\n"
            "- Move from WAIT_START_APPROVAL directly to WAIT_MANUAL_RELEASE_CONFIRMATION and "
            "request release creation confirmation via exactly one slack_approve tool call.\n"
            "- For that transition, do not emit slack_message; the confirmation text in "
            "slack_approve must include all required context.\n"
            "- For WAIT_MANUAL_RELEASE_CONFIRMATION wording, explicitly state: the Jira release "
            "is created manually by a human, and Release Manager only waits for confirmation.\n"
            "- WAIT_MANUAL_RELEASE_CONFIRMATION text must explicitly contain "
            "'Подтвердите создание релиза X.Y.Z в JIRA' (with actual version).\n"
            "- Do not use wording that implies automatic Jira creation or that execution is "
            "being delegated for this step.\n"
            "- After any approval_confirmed event, include slack_update for the trigger "
            "message so buttons disappear and text ends with :white_check_mark:.\n"
            "- After manual release confirmation event, move directly to "
            "WAIT_MEETING_CONFIRMATION and include exactly one slack_approve tool call "
            "that asks to confirm the release-fixation meeting has already happened "
            "(\"подтвердите, что встреча фиксации релиза уже прошла\").\n"
            "- Never use wording about meeting scheduling/appointment "
            "(\"назначение встречи\") for WAIT_MEETING_CONFIRMATION.\n"
            "- When transitioning to WAIT_READINESS_CONFIRMATIONS, include exactly one "
            "slack_message tool call with non-empty args.channel_id and args.text. "
            "args.text must include: "
            "'Релиз <https://instories.atlassian.net/issues/?jql=JQL|X.Y.Z>', blank line, "
            "'Статус готовности к срезу:', one ':hourglass_flowing_sand: <Team> <Owner mention>' "
            "line per team from config.readiness_owners, blank line, "
            "'Напишите в треде по готовности своей части.', and "
            "'**Важное напоминание** – все задачи, не влитые в ветку RC до 15:00 МСК "
            "едут в релиз только после одобрения QA'.\n"
            "- If next_step is WAIT_READINESS_CONFIRMATIONS and required readiness message "
            "is missing, keep current step unchanged and return audit_reason explaining why.\n"
            "- In WAIT_READINESS_CONFIRMATIONS, do not transition to READY_FOR_BRANCH_CUT "
            "until all teams from config.readiness_owners are confirmed in "
            "next_state.active_release.readiness_map with value true.\n"
            "\n"
            f"{_COMMON_SCHEMA_DEFS}"
            "Required output JSON schema:\n"
            "{\n"
            '  "next_step": "<ReleaseStep>",\n'
            '  "next_state": { ... WorkflowState dict ... },\n'
            '  "tool_calls": [<ToolCall>],\n'
            '  "audit_reason": "short explanation",\n'
            '  "flow_lifecycle": "running|paused|completed"\n'
            "}\n"
            "Use `next_state` as the complete state snapshot for persistence. "
            f"{_COMMON_FORMAT_RULES} "
            "Include all required tool fields. "
            "If no change is needed, return the same state with matching next_step and consistent flow_lifecycle."
        ),
        expected_output=(
            "Valid JSON object with next_step, next_state, tool_calls, audit_reason, flow_lifecycle."
        ),
        output_pydantic=AgentDecisionPayload,
        tools=[*slack_tools],
        agent=agent,
    )
