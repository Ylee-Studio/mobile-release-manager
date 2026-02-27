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
    heartbeat = workflow.get("heartbeat", {})
    storage = workflow.get("storage", {})
    slack = config.get("slack", {})
    jira = config.get("jira", {})

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
        heartbeat_active_minutes=int(heartbeat.get("active_minutes", 15)),
        heartbeat_idle_minutes=int(heartbeat.get("idle_minutes", 240)),
        jira_project_keys=[str(v) for v in jira.get("project_keys", [])],
        readiness_owners=readiness_owners,
        memory_db_path=storage.get("memory_db_path", "artifacts/crewai_memory.db"),
        audit_log_path=storage.get("audit_log_path", "artifacts/workflow_audit.jsonl"),
        slack_events_path=storage.get("slack_events_path", "artifacts/slack_events.jsonl"),
        agent_pid_path=storage.get("agent_pid_path", "artifacts/agent.pid"),
    )
