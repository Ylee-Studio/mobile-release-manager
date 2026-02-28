"""Task definition and detailed behavior instructions for Pydantic AI agent."""
from __future__ import annotations

from .runtime_contracts import AgentDecisionPayload

# Schema definitions for LLM context
_RELEASE_STEP_VALUES = (
    "IDLE|WAIT_START_APPROVAL|WAIT_MANUAL_RELEASE_CONFIRMATION|"
    "WAIT_MEETING_CONFIRMATION|WAIT_READINESS_CONFIRMATIONS|WAIT_BRANCH_CUT_APPROVAL|WAIT_RC_READINESS"
)

_TOOL_SCHEMA_DEFS = """
Available tools:
1. slack_message - Send message to Slack channel
   Args: {channel_id: string, text: string, thread_ts?: string}

2. slack_approve - Send approval request with button(s)
   Args: {channel_id: string, text: string, approve_label: string, reject_label?: string}

3. slack_update - Update existing Slack message
   Args: {channel_id: string, message_ts: string, text: string}

4. github_action - Trigger GitHub Actions workflow
   Args: {workflow_file: string, ref?: string, inputs?: object}

Tool call format:
{"tool": "slack_message", "reason": "why this call is made", "args": {...}}
"""

_RESPONSE_SCHEMA_DEFS = f"""
Response schema (AgentDecision):
- next_step: ReleaseStep enum value, one of: {_RELEASE_STEP_VALUES}
- next_state: optional full WorkflowState dict (use only when full replace is required)
- state_patch: preferred field for minimal partial updates
- tool_calls: list of tool call objects
- audit_reason: short reason code (1-4 words)
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
            "## Response Size Rules (high priority)\n"
            "- Return JSON only, no markdown, no explanations, no repeated input context.\n"
            "- Prefer minimal payload: keep next_state={} and write only changed fields in state_patch.\n"
            "- Use next_state only when a full replacement is strictly needed.\n"
            "- Keep audit_reason short and stable (1-4 words, snake_case preferred).\n"
            "- Keep tool_calls[].reason short (1-3 words) or empty string.\n"
            "- Do not include optional fields when they are unchanged.\n\n"
            f"{_RESPONSE_SCHEMA_DEFS}\n\n"
            f"{_TOOL_SCHEMA_DEFS}\n\n"
            "## Workflow State Transitions\n\n"
            "### 0. Manual status override command (highest priority)\n"
            "Trigger: events contain message 'Переведи X.Y.Z в STATUS' (case-insensitive)\n"
            "Message preprocessing:\n"
            "- If message starts with Slack mention(s) like '<@U...>', ignore mention prefix and parse remaining text\n"
            "Allowed STATUS values:\n"
            "- Any ReleaseStep value: IDLE, WAIT_START_APPROVAL, WAIT_MANUAL_RELEASE_CONFIRMATION, "
            "WAIT_MEETING_CONFIRMATION, WAIT_READINESS_CONFIRMATIONS, WAIT_BRANCH_CUT_APPROVAL, "
            "WAIT_RC_READINESS\n"
            "Actions:\n"
            "- Parse release version from message text (must match X.Y.Z)\n"
            "- Parse STATUS and convert to next_step\n"
            "- Do not generate tool_calls for this command\n"
            "- This command is valid even when flow is paused if current tick contains message events\n"
            "- If STATUS resolves to IDLE: clear active_release, clear pause metadata, set flow_lifecycle='completed'\n"
            "- Otherwise: set/create active_release for that version with parsed next_step, set flow_lifecycle='paused'\n"
            "- Set audit_reason to 'manual_status_override:{release_version}->{next_step}'\n\n"
            "### 0.5 Manual build trigger command (high priority)\n"
            "Trigger: events contain message 'Собери сборку' (case-insensitive)\n"
            "Message preprocessing:\n"
            "- If message starts with Slack mention(s) like '<@U...>', ignore mention prefix and parse remaining text\n"
            "Actions:\n"
            "- Do not change workflow state: keep next_step equal to current state.step\n"
            "- Keep next_state empty and state_patch empty\n"
            "- Add exactly ONE github_action tool call:\n"
            '  Args: {workflow_file: "build_rc.yml", ref: "dev", inputs: {}}\n'
            "- Set audit_reason to 'manual_build_rc_trigger'\n"
            "- Return flow_lifecycle='paused' when current flow is paused, otherwise keep current lifecycle semantics via unchanged state\n"
            "- Do not generate any slack_* tool calls for this command\n\n"
            "### 1. IDLE → WAIT_START_APPROVAL\n"
            "Trigger: events contain 'manual_start' event with version string (e.g., '5.105.0')\n"
            "Actions:\n"
            "- Parse version from event.text (must match X.Y.Z format)\n"
            "- Create active_release: {release_version, step: WAIT_START_APPROVAL, slack_channel_id}\n"
            "- Set flow_execution_id to new UUID if not present\n"
            "- Add exactly ONE slack_approve tool call for start approval:\n"
            '  Args: {channel_id, text: "Подтвердите старт релизного трейна {version}.", approve_label: "Подтвердить", reject_label: "Отклонить"}\n'
            "- Set flow_lifecycle='paused' (waiting for start approval)\n"
            "- Do NOT send slack_message instead of slack_approve for this step\n\n"
            "### 2. WAIT_START_APPROVAL → WAIT_MANUAL_RELEASE_CONFIRMATION\n"
            "Trigger:\n"
            "- approval_confirmed received while current step is WAIT_START_APPROVAL\n"
            "- OR message event with bot mention clearly indicates confirmation intent for current approval step\n"
            "Actions:\n"
            "- Set next_step = WAIT_MANUAL_RELEASE_CONFIRMATION\n"
            "- Add slack_update for approved start-approval message.\n"
            '  Args: {channel_id, message_ts, text: "Подтверждено :white_check_mark:"} (runtime appends confirmation to original message text from event metadata)\n'
            "- Add exactly ONE slack_approve tool call for Jira release creation:\n"
            '  Args: {channel_id, text: "Подтвердите создание релиза {version} в <https://instories.atlassian.net/jira/plans/1/scenarios/1/releases|JIRA>.", approve_label: "Подтвердить"}\n'
            "- Save message_ts from this approval message to active_release.message_ts['manual_release_confirmation']\n"
            "- Return flow_lifecycle='paused' (waiting for human)\n\n"
            "### 2.1 WAIT_START_APPROVAL → IDLE (rejected)\n"
            "Trigger: approval_rejected received while current step is WAIT_START_APPROVAL\n"
            "Actions:\n"
            "- Set next_step = IDLE\n"
            "- Add slack_update for rejected start-approval message.\n"
            '  Args: {channel_id, message_ts, text: "Отклонено :x:"} (runtime appends rejection suffix to original message text from event metadata)\n'
            "- Clear active_release in state_patch/next_state\n"
            "- Set flow_lifecycle='completed'\n\n"
            "### 3. On approval_confirmed event (button clicked or message-confirm)\n"
            "Trigger: events contain 'approval_confirmed' event\n"
            "Actions:\n"
            "- Find matching approval message via message_ts\n"
            "- Add slack_update tool call to mark approval as confirmed:\n"
            '  Args: {channel_id, message_ts, text: "Подтверждено :white_check_mark:"} (runtime appends confirmation to original message text from event metadata)\n'
            "- Transition based on current step\n\n"
            "### 3.1 On approval_rejected event (button clicked)\n"
            "Trigger: events contain 'approval_rejected' event\n"
            "Actions:\n"
            "- Find matching approval message via message_ts\n"
            "- Add slack_update tool call to mark rejection:\n"
            '  Args: {channel_id, message_ts, text: "Отклонено :x:"} (runtime appends rejection suffix to original message text from event metadata)\n'
            "- If current step is WAIT_START_APPROVAL: transition to IDLE, clear active_release, flow_lifecycle=completed\n"
            "- For other steps: keep conservative no-op transition (same state, no additional transitions)\n\n"
            "### 4. WAIT_MANUAL_RELEASE_CONFIRMATION → WAIT_MEETING_CONFIRMATION\n"
            "Trigger:\n"
            "- approval_confirmed received in WAIT_MANUAL_RELEASE_CONFIRMATION\n"
            "- OR message event with bot mention clearly indicates confirmation intent for this step\n"
            "Actions:\n"
            "- Set next_step = WAIT_MEETING_CONFIRMATION\n"
            "- Add slack_approve tool call:\n"
            '  Args: {channel_id, text: "Подтвердите, что встреча фиксации <https://instories.atlassian.net/issues/?jql=fixVersion={version}|релиза> уже прошла.", approve_label: "Встреча прошла"}\n'
            "- Return flow_lifecycle='paused'\n\n"
            "### 5. WAIT_MEETING_CONFIRMATION → WAIT_READINESS_CONFIRMATIONS\n"
            "Trigger:\n"
            "- approval_confirmed received in WAIT_MEETING_CONFIRMATION\n"
            "- OR message event with bot mention clearly indicates confirmation intent for this step\n"
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
            "  * Set next_step = WAIT_BRANCH_CUT_APPROVAL\n"
            "  * Add exactly ONE github_action tool call to trigger release branch workflow:\n"
            '    Args: {workflow_file: "create_release_branch.yml", inputs: {"version": "{version}"}}\n'
            "  * Add exactly ONE slack_approve tool call:\n"
            '    Args: {channel_id, text: "Выделил RС ветку. Подтвердите <https://github.com/Ylee-Studio/instories-ios/actions|готовность> ветки.\\n- FontThumbnailManager.Spec обновлен при необходимости\\n- В L10n.extension.swift нет временной локализации\\n- В TestTemplates.swift нет шаблонов", approve_label: "Подтвердить"}\n'
            '  * Do NOT send slack_message with text "Можно выделять RC ветку"\n'
            "  * Return flow_lifecycle='paused'\n"
            "- Otherwise stay in WAIT_READINESS_CONFIRMATIONS with flow_lifecycle='paused'\n\n"
            "### 7. WAIT_BRANCH_CUT_APPROVAL\n"
            "Trigger:\n"
            "- approval_confirmed received in WAIT_BRANCH_CUT_APPROVAL\n"
            "- OR message event with bot mention clearly indicates confirmation intent for this step\n"
            "Actions:\n"
            "- Set next_step = WAIT_RC_READINESS\n"
            "- Add slack_update to mark approval as confirmed:\n"
            '  Args: {channel_id, message_ts, text: "Подтверждено :white_check_mark:\\nя запустил сборку"} (runtime appends confirmation to original message text from event metadata)\n'
            "- Add exactly ONE github_action tool call:\n"
            '  Args: {workflow_file: "build_rc.yml", ref: "dev", inputs: {}}\n'
            "- Return flow_lifecycle='paused'\n\n"
            "### 8. WAIT_RC_READINESS\n"
            "Trigger: events contain message exactly 'Релиз готов к отправке'\n"
            "Actions:\n"
            '- Add exactly ONE slack_message tool call with exact text: "Поздравляем — релиз готов к отправке! :tada: Отличная работа всей команде."\n'
            "- Send this final congratulation as a channel-level message (without thread_ts)\n"
            "- Set next_step = IDLE\n"
            "- Clear active_release in state_patch/next_state\n"
            "- Set flow_lifecycle='completed'\n"
            "- Otherwise keep WAIT_RC_READINESS with flow_lifecycle='paused'\n\n"
            "## Output Format\n\n"
            "Return compact valid JSON matching AgentDecision schema. Example:\n\n"
            '{\n'
            '  "next_step": "WAIT_MANUAL_RELEASE_CONFIRMATION",\n'
            '  "next_state": {},\n'
            '  "state_patch": {\n'
            '    "active_release": {\n'
            '      "step": "WAIT_MANUAL_RELEASE_CONFIRMATION"\n'
            '    }\n'
            '    },\n'
            '  "tool_calls": [\n'
            '    {\n'
            '      "tool": "slack_approve",\n'
            '      "reason": "request_approval",\n'
            '      "args": {\n'
            '        "channel_id": "C123",\n'
            '        "text": "Подтвердите создание релиза 5.105.0 в <https://instories.atlassian.net/jira/plans/1/scenarios/1/releases|JIRA>",\n'
            '        "approve_label": "Подтвердить"\n'
            '      }\n'
            '    }\n'
            '  ],\n'
            '  "audit_reason": "request_release_confirmation",\n'
            '  "flow_lifecycle": "paused"\n'
            '}\n\n'
            "## Important Notes\n\n"
            "- Always validate step transitions are allowed\n"
            "- Prefer state_patch for minimal updates; use next_state only for full replacement\n"
            "- Include all required tool fields in args\n"
            "- Never use 'functions.' prefix in tool names\n"
            "- flow_lifecycle='paused' requires flow_execution_id to be set\n"
            "- In approval-waiting steps, message events with bot mention may be treated as confirmation ONLY if intent is explicit; if intent is ambiguous or neutral, return conservative no-op\n"
            "- After approval_confirmed or approval_rejected, send slack_update for the clicked message\n"
            "- If you are uncertain, return conservative no-op (same step/state, empty tool_calls) with explicit audit_reason"
        ),
    }
