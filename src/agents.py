"""CrewAI agent definitions for orchestrated release workflow."""
from __future__ import annotations

from crewai import Agent

from .policies import PolicyConfig


def build_orchestrator_agent(policies: PolicyConfig) -> Agent:
    """Create orchestrator agent responsible for release routing decisions."""
    return Agent(
        role="Release Orchestrator",
        goal=(
            "Control release-train state transitions and decide when to create or invoke "
            "a release-specific manager."
        ),
        backstory=(
            "You monitor heartbeat context and incoming events, determine the next "
            "workflow step, and create Release Manager X.Y.Z when a release enters "
            "execution phases."
        ),
        allow_delegation=False,
        verbose=True,
        max_interactions=policies.max_interactions,
        instructions=(
            "Always return strict JSON with next_step, next_state OR state_patch, "
            "tool_calls, audit_reason, and optional invoke_release_manager flag. "
            "Set invoke_release_manager=true only when an active release exists and the "
            "release manager must process release-specific side effects. Keep state "
            "consistent with ReleaseStep enum and preserve idempotency via "
            "state.processed_event_ids and state.completed_actions."
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
        verbose=True,
        max_interactions=policies.max_interactions,
        instructions=(
            "Always return strict JSON with next_step, next_state OR state_patch, "
            "tool_calls, and audit_reason. Use only the active release context and "
            "incoming events. Use `slack_approve` to request manual release confirmation "
            "before moving to JIRA_RELEASE_CREATED. For every approval_confirmed event, "
            "add a `slack_update` tool call that updates the original approval message, "
            "removes buttons, and appends :white_check_mark: to its text. Then use "
            "`slack_message` to notify the channel about manual Jira release creation "
            "and release-fixation meeting. "
            "When moving to WAIT_READINESS_CONFIRMATIONS, always include exactly one "
            "`slack_message` tool call with a non-empty readiness text built from "
            "config.readiness_owners (one :hourglass_flowing_sand: line per team). "
            "If you cannot produce that message, do not move to WAIT_READINESS_CONFIRMATIONS. "
            "When tools return message_ts, persist it into "
            "next_state.active_release.message_ts/thread_ts where applicable. "
            "Never duplicate side effects when state.completed_actions indicates "
            "an action already happened."
        ),
    )
