"""Task definition and detailed behavior instructions for Pydantic AI agent."""
from __future__ import annotations

from .runtime_contracts import AgentDecisionPayload

# Schema definitions for LLM context
_RELEASE_STEP_VALUES = (
    "IDLE|WAIT_START_APPROVAL|WAIT_MANUAL_RELEASE_CONFIRMATION|"
    "WAIT_MEETING_CONFIRMATION|WAIT_READINESS_CONFIRMATIONS|WAIT_BRANCH_CUT|WAIT_BRANCH_CUT_APPROVAL"
)

_TOOL_SCHEMA_DEFS = """
Available tools:
1. slack_message - Send message to Slack channel
   Args: {channel_id: string, text: string, thread_ts?: string}

2. slack_approve - Send approval request with button
   Args: {channel_id: string, text: string, approve_label: string}

3. slack_update - Update existing Slack message
   Args: {channel_id: string, message_ts: string, text: string}

4. github_action - Trigger GitHub Actions workflow
   Args: {workflow_file: string, ref: string, inputs?: object}

Tool call format:
{"tool": "slack_message", "reason": "why this call is made", "args": {...}}
"""

_RESPONSE_SCHEMA_DEFS = f"""
Response schema (AgentDecision):
- next_step: ReleaseStep enum value, one of: {_RELEASE_STEP_VALUES}
- next_state: dict with active_release, flow_execution_id, checkpoints, etc.
- state_patch: optional partial updates to merge into next_state
- tool_calls: list of tool call objects
- audit_reason: string explaining the decision
- flow_lifecycle: "running" | "paused" | "completed"
"""


def build_flow_task(agent: dict[str, object]) -> dict[str, object]:
    """Return detailed task instructions for Pydantic AI agent.

    This defines the step-by-step behavior logic that the agent must follow.
    The system-level role and constraints come from build_flow_agent().
    """
    return {
        "agent": agent,
        "schema": AgentDecisionPayload,
        "description": (
            "# Release Workflow Orchestrator Task\n\n"
            "You receive the following inputs:\n"
            "- state: Current WorkflowState dict\n"
            "- events: List of Slack events since last tick\n"
            "- config: {slack_channel_id, timezone, jira_project_keys, readiness_owners}\n"
            "- now_iso: Current timestamp ISO format\n"
            "- trigger_reason: Why this tick was triggered\n\n"
            "Runtime contract:\n"
            "- Runtime is agent-first and does not apply deterministic business transitions.\n"
            "- If runtime cannot parse/validate your output, it falls back to safe no-op.\n\n"
            f"{_RESPONSE_SCHEMA_DEFS}\n\n"
            f"{_TOOL_SCHEMA_DEFS}\n\n"
            "## Workflow State Transitions\n\n"
            "### 0. Manual status override command (highest priority)\n"
            "Trigger: events contain message 'Переведи X.Y.Z в STATUS' (case-insensitive)\n"
            "Message preprocessing:\n"
            "- If message starts with Slack mention(s) like '<@U...>', ignore mention prefix and parse remaining text\n"
            "Allowed STATUS values:\n"
            "- Any ReleaseStep value: IDLE, WAIT_START_APPROVAL, WAIT_MANUAL_RELEASE_CONFIRMATION, "
            "WAIT_MEETING_CONFIRMATION, WAIT_READINESS_CONFIRMATIONS, WAIT_BRANCH_CUT, "
            "WAIT_BRANCH_CUT_APPROVAL\n"
            "Actions:\n"
            "- Parse release version from message text (must match X.Y.Z)\n"
            "- Parse STATUS and convert to next_step\n"
            "- Do not generate tool_calls for this command\n"
            "- This command is valid even when flow is paused if current tick contains message events\n"
            "- If STATUS resolves to IDLE: clear active_release, clear pause metadata, set flow_lifecycle='completed'\n"
            "- Otherwise: set/create active_release for that version with parsed next_step, set flow_lifecycle='paused'\n"
            "- Set audit_reason to 'manual_status_override:{release_version}->{next_step}'\n\n"
            "### 1. IDLE → WAIT_START_APPROVAL\n"
            "Trigger: events contain 'manual_start' event with version string (e.g., '5.105.0')\n"
            "Actions:\n"
            "- Parse version from event.text (must match X.Y.Z format)\n"
            "- Create active_release: {release_version, step: WAIT_START_APPROVAL, slack_channel_id}\n"
            "- Set flow_execution_id to new UUID if not present\n"
            "- Add exactly ONE slack_approve tool call for start approval:\n"
            '  Args: {channel_id, text: "Подтвердите старт релизного трейна {version}. Если нужен другой номер для релиза, напишите в треде.", approve_label: "Подтвердить"}\n'
            "- Set flow_lifecycle='paused' (waiting for start approval)\n"
            "- Do NOT send slack_message instead of slack_approve for this step\n\n"
            "### 2. WAIT_START_APPROVAL → WAIT_MANUAL_RELEASE_CONFIRMATION\n"
            "Trigger: approval_confirmed received while current step is WAIT_START_APPROVAL\n"
            "Actions:\n"
            "- Set next_step = WAIT_MANUAL_RELEASE_CONFIRMATION\n"
            "- Add slack_update for approved start-approval message.\n"
            '  Args: {channel_id, message_ts, text: "Подтверждено :white_check_mark:"} (runtime appends confirmation to original message text from event metadata)\n'
            "- Add exactly ONE slack_approve tool call for Jira release creation:\n"
            '  Args: {channel_id, text: "Подтвердите создание релиза {version} в <https://instories.atlassian.net/jira/plans/1/scenarios/1/releases|JIRA>.", approve_label: "Подтвердить"}\n'
            "- Save message_ts from this approval message to active_release.message_ts['manual_release_confirmation']\n"
            "- Return flow_lifecycle='paused' (waiting for human)\n\n"
            "### 3. On approval_confirmed event (button clicked)\n"
            "Trigger: events contain 'approval_confirmed' event\n"
            "Actions:\n"
            "- Find matching approval message via message_ts\n"
            "- Add slack_update tool call to mark approval as confirmed:\n"
            '  Args: {channel_id, message_ts, text: "Подтверждено :white_check_mark:"} (runtime appends confirmation to original message text from event metadata)\n'
            "- Transition based on current step\n\n"
            "### 4. WAIT_MANUAL_RELEASE_CONFIRMATION → WAIT_MEETING_CONFIRMATION\n"
            "Trigger: approval_confirmed received in WAIT_MANUAL_RELEASE_CONFIRMATION\n"
            "Actions:\n"
            "- Set next_step = WAIT_MEETING_CONFIRMATION\n"
            "- Add slack_approve tool call:\n"
            '  Args: {channel_id, text: "Подтвердите, что встреча фиксации <https://instories.atlassian.net/issues/?jql=fixVersion={version}|релиза> уже прошла.", approve_label: "Встреча прошла"}\n'
            "- Return flow_lifecycle='paused'\n\n"
            "### 5. WAIT_MEETING_CONFIRMATION → WAIT_READINESS_CONFIRMATIONS\n"
            "Trigger: approval_confirmed received in WAIT_MEETING_CONFIRMATION\n"
            "Actions:\n"
            "- Set next_step = WAIT_READINESS_CONFIRMATIONS\n"
            "- Build readiness message text:\n"
            '  "Релиз <https://instories.atlassian.net/issues/?jql=fixVersion={version}|{version}>\n\n'
            '  Статус готовности к срезу:\n'
            '  {for each team in readiness_owners: ":hourglass_flowing_sand: {team} {owner_mention}\n"}\n\n'
            '  Напишите в треде по готовности своей части.\n\n'
            '  *Важное напоминание* – все задачи, не влитые в ветку RC до 15:00 МСК '
            '  едут в релиз только после одобрения QA"\n'
            "- Add slack_message with readiness text, save thread_ts from response\n"
            "- Return flow_lifecycle='paused'\n\n"
            "### 6. WAIT_READINESS_CONFIRMATIONS handling\n"
            "For each 'message' event in readiness thread:\n"
            "- Check event.thread_ts matches readiness thread\n"
            "- Normalize text: lowercase, remove special chars except spaces\n"
            "- Positive markers: 'готов', 'ready', 'done', 'ok', 'merged', 'влит', 'смержен'\n"
            "- Negative markers: 'не готов', 'not ready', 'blocked', 'блокер'\n"
            "- Skip messages with negative markers\n"
            "- For positive messages, match readiness point:\n"
            "  * Check if point name/alias appears in message text\n"
            "  * OR check if message user's ID matches point owner's mention\n"
            "- Update readiness_map[point] = true for matched points\n"
            "- Do NOT send standalone slack_message acknowledgements per readiness confirmation\n"
            "- Instead, send slack_update for the original readiness checklist message_ts and update only affected lines:\n"
            "  * matched points -> ':white_check_mark:'\n"
            "  * unmatched points remain ':hourglass_flowing_sand:'\n"
            "- If ALL points from config.readiness_owners are true:\n"
            "  * Set next_step = WAIT_BRANCH_CUT\n"
            "  * Add exactly ONE github_action tool call to trigger release branch workflow:\n"
            '    Args: {workflow_file: "create_release_branch.yml", ref: "main", inputs: {"version": "{version}"}}\n'
            '  * Add exactly ONE slack_message tool call with text: "Можно выделять RC ветку" (в канал, без thread_ts)\n'
            "  * Return flow_lifecycle='paused'\n"
            "- Otherwise stay in WAIT_READINESS_CONFIRMATIONS with flow_lifecycle='paused'\n\n"
            "### 7. WAIT_BRANCH_CUT\n"
            "Trigger: release branch action has been started and is being polled by runtime\n"
            "Actions:\n"
            "- Keep current step while action status is queued/in_progress\n"
            "- Return no tool_calls by default\n"
            "- Return flow_lifecycle='paused'\n\n"
            "### 8. WAIT_BRANCH_CUT_APPROVAL\n"
            "Trigger: runtime detected branch-cut action completed\n"
            "Actions:\n"
            "- Hold state for downstream manual approval\n"
            "- Return flow_lifecycle='paused'\n\n"
            "## Output Format\n\n"
            "Return valid JSON matching AgentDecision schema. Example:\n\n"
            '{\n'
            '  "next_step": "WAIT_MANUAL_RELEASE_CONFIRMATION",\n'
            '  "next_state": {\n'
            '    "active_release": {\n'
            '      "release_version": "5.105.0",\n'
            '      "step": "WAIT_MANUAL_RELEASE_CONFIRMATION",\n'
            '      "readiness_map": {},\n'
            '      "message_ts": {},\n'
            '      "thread_ts": {}\n'
            '    },\n'
            '    "flow_execution_id": "uuid-here",\n'
            '    "flow_paused_at": "2026-02-27T16:00:00+00:00"\n'
            '  },\n'
            '  "tool_calls": [\n'
            '    {\n'
            '      "tool": "slack_approve",\n'
            '      "reason": "Request release creation confirmation",\n'
            '      "args": {\n'
            '        "channel_id": "C123",\n'
            '        "text": "Подтвердите создание релиза 5.105.0 в <https://instories.atlassian.net/jira/plans/1/scenarios/1/releases|JIRA>",\n'
            '        "approve_label": "Подтвердить"\n'
            '      }\n'
            '    }\n'
            '  ],\n'
            '  "audit_reason": "created_release_request",\n'
            '  "flow_lifecycle": "paused"\n'
            '}\n\n'
            "## Important Notes\n\n"
            "- Always validate step transitions are allowed\n"
            "- Use next_state for complete state, state_patch only for partial updates\n"
            "- Include all required tool fields in args\n"
            "- Never use 'functions.' prefix in tool names\n"
            "- flow_lifecycle='paused' requires flow_execution_id to be set\n"
            "- After approval_confirmed, always send slack_update to mark the approval message\n"
            "- If you are uncertain, return conservative no-op (same step/state, empty tool_calls) with explicit audit_reason"
        ),
    }
