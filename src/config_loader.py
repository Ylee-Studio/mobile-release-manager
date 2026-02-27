"""Runtime configuration mapping from raw config dict."""
from __future__ import annotations

import os
from pathlib import Path

from .release_workflow import RuntimeConfig

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE_ROOT / "config.yaml"

REQUIRED_ENV_VARS = [
    "OPENAI_API_KEY",
    "SLACK_BOT_TOKEN",
]


def resolve_env_value(raw_value: str, *, fallback_env_var: str) -> str:
    value = raw_value
    if value.startswith("${") and value.endswith("}"):
        value = os.getenv(value[2:-1], "")
    if not value:
        value = os.getenv(fallback_env_var, "")
    return value


def load_runtime_config(config: dict) -> RuntimeConfig:
    workflow = config.get("workflow", {})
    storage = workflow.get("storage", {})
    slack = config.get("slack", {})
    jira = config.get("jira", {})
    github = config.get("github", {})

    readiness_owners = {
        str(team): str(owner)
        for team, owner in workflow.get("readiness_owners", {}).items()
    }
    channel_id = resolve_env_value(
        str(slack.get("channel_id", "")),
        fallback_env_var="SLACK_ANNOUNCE_CHANNEL",
    )
    bot_token = resolve_env_value(
        str(slack.get("bot_token", "")),
        fallback_env_var="SLACK_BOT_TOKEN",
    )
    github_repo = resolve_env_value(
        str(github.get("repo", "Ylee-Studio/instories-ios")),
        fallback_env_var="GITHUB_REPO",
    ) or "Ylee-Studio/instories-ios"
    github_token = resolve_env_value(
        str(github.get("token", "")),
        fallback_env_var="GITHUB_TOKEN",
    )
    github_workflow_file = str(github.get("workflow_file", "create_release_branch.yml") or "").strip() or "create_release_branch.yml"
    github_ref = str(github.get("ref", "dev") or "").strip() or "dev"
    github_poll_interval_seconds_raw = github.get("poll_interval_seconds", 30)
    try:
        github_poll_interval_seconds = int(github_poll_interval_seconds_raw)
    except (TypeError, ValueError):
        github_poll_interval_seconds = 30
    if github_poll_interval_seconds <= 0:
        github_poll_interval_seconds = 30
    if not readiness_owners:
        raise RuntimeError("workflow.readiness_owners must be configured in config.yaml")
    if not channel_id:
        raise RuntimeError("slack.channel_id must be set in config.yaml or env")
    if not bot_token:
        raise RuntimeError("slack.bot_token must be set in config.yaml or env")
    return RuntimeConfig(
        slack_channel_id=channel_id,
        slack_bot_token=bot_token,
        timezone=workflow.get("timezone", "Europe/Moscow"),
        jira_project_keys=[str(v) for v in jira.get("project_keys", [])],
        readiness_owners=readiness_owners,
        memory_db_path=storage.get("memory_db_path", "artifacts/workflow_state.jsonl"),
        audit_log_path=storage.get("audit_log_path", "artifacts/workflow_audit.jsonl"),
        slack_events_path=storage.get("slack_events_path", "artifacts/slack_events.jsonl"),
        github_repo=github_repo,
        github_token=github_token,
        github_workflow_file=github_workflow_file,
        github_ref=github_ref,
        github_poll_interval_seconds=github_poll_interval_seconds,
    )
