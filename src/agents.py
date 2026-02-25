"""CrewAI agent definitions for release train workflow."""
from __future__ import annotations

from crewai import Agent

from .policies import PolicyConfig


def build_orchestrator_agent(policies: PolicyConfig) -> Agent:
    """Create the orchestration agent."""
    return Agent(
        role="Orchestrator",
        goal="Start release trains at the right time and hand off to release managers.",
        backstory=(
            "You coordinate release lifecycle boundaries. You can start from schedule "
            "or manual Slack command, then initialize state and release manager context."
        ),
        allow_delegation=False,
        verbose=True,
        max_interactions=policies.max_interactions,
        instructions=(
            "Use heartbeat and Slack start signals. Never duplicate side effects. "
            "Create one release manager per version and persist state transitions."
        ),
    )


def build_release_manager_agent(policies: PolicyConfig, release_version: str) -> Agent:
    """Create the release manager agent for a concrete version."""
    return Agent(
        role=f"Release Manager {release_version}",
        goal=f"Drive release {release_version} to READY_FOR_BRANCH_CUT.",
        backstory=(
            "You run operational release steps in Slack and Jira, keeping one source "
            "of truth in state storage and updating readiness in-place."
        ),
        allow_delegation=False,
        verbose=True,
        max_interactions=policies.max_interactions,
        instructions=(
            "Use Slack approve/message/update and Jira cross-space release. "
            "Handle thread events directly, update readiness map, and stay idempotent."
        ),
    )
