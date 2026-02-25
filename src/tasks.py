"""CrewAI task definitions aligned with 001.001 workflow."""
from __future__ import annotations

from crewai import Task


def build_release_workflow_tasks(orchestrator_agent, release_manager_agent, jira_tool, slack_tools):
    """Return task templates used by orchestrator and release manager."""
    start_release_train = Task(
        description=(
            "Start release train from heartbeat schedule or manual Slack request."
            " Prepare start approval in channel and persist WAIT_START_APPROVAL."
        ),
        expected_output=(
            "One start-approval message exists, release version is set, and state is persisted."
        ),
        tools=slack_tools,
        agent=orchestrator_agent,
    )

    confirm_or_update_version = Task(
        description=(
            "Track replies in start approval thread: accept alternative X.Y.Z if provided,"
            " then continue once approval is confirmed."
        ),
        expected_output="Final approved version and transition to RELEASE_MANAGER_CREATED.",
        tools=slack_tools,
        agent=release_manager_agent,
    )

    create_crossspace_release = Task(
        description=(
            "Create Jira cross-space release for approved X.Y.Z with configured projects."
        ),
        expected_output="Jira cross-space release exists and state moved to JIRA_RELEASE_CREATED.",
        tools=[jira_tool],
        agent=release_manager_agent,
    )

    announce_kickoff_meeting = Task(
        description=(
            "Post kickoff message with JQL link, request meeting confirmation,"
            " and transition to WAIT_MEETING_CONFIRMATION."
        ),
        expected_output="Meeting confirmation request posted via slack_approve.",
        tools=slack_tools,
        agent=release_manager_agent,
    )

    collect_readiness_statuses = Task(
        description=(
            "Post one readiness status message and update it incrementally by thread confirmations."
        ),
        expected_output="All readiness owners marked :white_check_mark: and state READY_FOR_BRANCH_CUT.",
        tools=slack_tools,
        agent=release_manager_agent,
    )

    return [
        start_release_train,
        confirm_or_update_version,
        create_crossspace_release,
        announce_kickoff_meeting,
        collect_readiness_statuses,
    ]
