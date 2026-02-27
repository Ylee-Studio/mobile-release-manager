"""Single-agent task definition for direct runtime."""
from __future__ import annotations

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


def build_flow_task(agent: dict[str, object]) -> dict[str, object]:
    """Return single flow-agent prompt payload."""
    return {
        "agent": agent,
        "schema": AgentDecisionPayload,
        "description": (
            "You receive `state`, `events`, `config`, `now_iso`, `trigger_reason` as inputs. "
            "Execute the whole release flow from WAIT_START_APPROVAL to READY_FOR_BRANCH_CUT "
            "using one agent and return the updated workflow state."
            "\n"
            "Input context (authoritative):\n"
            "state={state}\n"
            "events={events}\n"
            "config={config}\n"
            "now_iso={now_iso}\n"
            "trigger_reason={trigger_reason}\n"
            "\n"
            "Behavior requirements:\n"
            "- Return flow_lifecycle='paused' while waiting for human confirmation, "
            "flow_lifecycle='running' after approval_confirmed transitions, and "
            "flow_lifecycle='completed' only when flow is truly finished.\n"
            "- Ensure all confirmation gates are compatible with pending-flow recovery via flow_execution_id.\n"
            "- Move from WAIT_START_APPROVAL directly to WAIT_MANUAL_RELEASE_CONFIRMATION and "
            "request release creation confirmation via exactly one slack_approve tool call.\n"
            "- WAIT_MANUAL_RELEASE_CONFIRMATION text must explicitly contain "
            "'Подтвердите создание релиза X.Y.Z в JIRA' (with actual version).\n"
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
    }
