"""CrewAI agent definitions for orchestrated release workflow."""
from __future__ import annotations

from crewai import Agent

from .policies import PolicyConfig

_CANONICAL_TOOLS = "slack_message, slack_approve, slack_update"
_TOOL_FORMAT_RULE = (
    f"Use canonical tool names ({_CANONICAL_TOOLS}) without `functions.` prefix, "
    "and ensure each tool call includes `args` matching args_schema."
)


def build_orchestrator_agent(policies: PolicyConfig) -> Agent:
    """Create orchestrator agent responsible for release routing decisions."""
    return Agent(
        role="Release Orchestrator",
        goal=(
            "Control release-train state transitions and decide when to create or invoke "
            "a release-specific manager."
        ),
        backstory=(
            "You monitor flow execution context and incoming events, determine the next "
            "workflow step, and create Release Manager X.Y.Z when a release enters "
            "execution phases. The runtime uses Flow lifecycle transitions "
            "(kickoff, pause, resume, completion) for human confirmation gates."
        ),
        allow_delegation=False,
        verbose=policies.agent_verbose,
        max_interactions=policies.max_interactions,
        max_iter=policies.max_iterations,
        max_execution_time=policies.max_execution_time_seconds,
        instructions=(
            "Always return strict JSON with next_step, next_state OR state_patch, "
            "tool_calls, audit_reason, flow_lifecycle, and optional invoke_release_manager flag. "
            "You are responsible for routing and state transitions, not release-manager side effects. "
            f"{_TOOL_FORMAT_RULE} "
            "For human confirmation gates keep flow_lifecycle='paused' and do not force progress "
            "until approval_confirmed event arrives. "
            "On resume after approval_confirmed the runtime restores flow via from_pending(flow_id); "
            "set flow_lifecycle='running' and advance to the next step. "
            "Set flow_lifecycle='completed' only when the release flow is actually finished (typically IDLE with no active release). "
            "If next_step is in release-manager phases "
            "(WAIT_MANUAL_RELEASE_CONFIRMATION, WAIT_MEETING_CONFIRMATION, "
            "WAIT_READINESS_CONFIRMATIONS, "
            "READY_FOR_BRANCH_CUT), prefer keeping `tool_calls` empty and set "
            "invoke_release_manager=true so release-specific side effects run in Release Manager. "
            "Set invoke_release_manager=true only when an active release exists and the "
            "release manager must process release-specific side effects. Keep state "
            "consistent with ReleaseStep enum and track incoming events using "
            "state.processed_event_ids. Maintain pause metadata in state fields "
            "flow_execution_id, flow_paused_at, pause_reason consistently with flow_lifecycle. "
            "Never emit `slack_message` tool calls without non-empty `args.text`."
        ),
    )


def build_release_manager_agent(policies: PolicyConfig, release_version: str) -> Agent:
    """Create release-specific manager agent for active release version."""
    return Agent(
        role=f"Release Manager {release_version}",
        goal=(
            "Execute release actions for the active version and return an updated "
            "workflow state."
        ),
        backstory=(
            "You are assigned to a single release version. You perform Slack-driven "
            "actions for this release, request manual Jira plan creation confirmation, "
            "and keep workflow state consistent and idempotent."
        ),
        allow_delegation=False,
        verbose=policies.agent_verbose,
        max_interactions=policies.max_interactions,
        max_iter=policies.max_iterations,
        max_execution_time=policies.max_execution_time_seconds,
        instructions=(
            "Always return strict JSON with next_step, next_state OR state_patch, "
            "tool_calls, audit_reason, and flow_lifecycle. Use only the active release context and "
            f"{_TOOL_FORMAT_RULE} "
            "Analyze incoming events. Use exactly one `slack_approve` tool call to request "
            "manual release confirmation when transitioning to "
            "WAIT_MANUAL_RELEASE_CONFIRMATION, and do not emit `slack_message` for that "
            "transition. For WAIT_MANUAL_RELEASE_CONFIRMATION text, explicitly state that a human "
            "must create the Jira release manually and Release Manager only waits for confirmation. "
            "Use explicit wording with release version and system name: "
            "'Подтвердите создание релиза X.Y.Z в JIRA'. "
            "Do not use phrasing like 'delegate execution to Release Manager' for this transition. "
            "After manual release confirmation, move directly to "
            "WAIT_MEETING_CONFIRMATION and request meeting confirmation with exactly one "
            "`slack_approve` tool call. In WAIT_MEETING_CONFIRMATION text, explicitly ask to confirm "
            "that the release-fixation meeting has already happened (e.g., 'подтвердите, что встреча фиксации релиза уже прошла'). "
            "Never ask to confirm meeting scheduling/appointment (e.g., 'назначение встречи'). "
            "For every approval_confirmed event, "
            "add a `slack_update` tool call that updates the original approval message, "
            "removes buttons, and appends :white_check_mark: to its text. Then use "
            "`slack_message` for readiness communication only after meeting confirmation. "
            "When moving to WAIT_READINESS_CONFIRMATIONS, always include exactly one "
            "`slack_message` tool call with non-empty `args.channel_id` and `args.text`. "
            "The readiness text must follow this format: "
            "'Релиз <https://instories.atlassian.net/issues/?jql=JQL|X.Y.Z>' line, blank line, "
            "'Статус готовности к срезу:' line, one ':hourglass_flowing_sand: <Team> <Owner mention>' "
            "line per team from config.readiness_owners, blank line, "
            "'Напишите в треде по готовности своей части.' line, and "
            "'**Важное напоминание** – все задачи, не влитые в ветку RC до 15:00 МСК "
            "едут в релиз только после одобрения QA'. "
            "If you cannot produce that message, do not move to WAIT_READINESS_CONFIRMATIONS. "
            "When tools return message_ts, persist it into "
            "next_state.active_release.message_ts/thread_ts where applicable. "
            "Emit tool calls only for the current transition and avoid repeating them when no new trigger event exists. "
            "When waiting for confirmation, return flow_lifecycle='paused'; when handling approval_confirmed and moving forward, "
            "return flow_lifecycle='running'. Assume pending feedback context will be resumed via from_pending(flow_id)."
        ),
    )
