"""Configuration helpers for release workflow runtime."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import yaml

from .release_workflow import RuntimeConfig

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE_ROOT / "config.yaml"

REQUIRED_ENV_VARS = [
    "OPENAI_API_KEY",
    "SLACK_BOT_TOKEN",
    "JIRA_BASE_URL",
    "JIRA_EMAIL",
    "JIRA_API_TOKEN",
]


def load_config() -> dict:
    """Load the YAML configuration file."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config.yaml at {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_env_vars(vars_to_check: Iterable[str] | None = None) -> None:
    """Ensure environment variables are set before running the crew."""
    missing = [var for var in (vars_to_check or REQUIRED_ENV_VARS) if not os.getenv(var)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(sorted(missing))
        )


def _resolve_env_value(raw_value: str, *, fallback_env_var: str) -> str:
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
    retry = config.get("retry", {})

    readiness_owners = {
        str(team): str(owner)
        for team, owner in workflow.get("readiness_owners", {}).items()
    }
    channel_id = _resolve_env_value(
        str(slack.get("channel_id", "")),
        fallback_env_var="SLACK_ANNOUNCE_CHANNEL",
    )
    if not readiness_owners:
        raise RuntimeError("workflow.readiness_owners must be configured in config.yaml")
    if not channel_id:
        raise RuntimeError("slack.channel_id must be set in config.yaml or env")
    jira_base_url = _resolve_env_value(
        str(jira.get("base_url", "")),
        fallback_env_var="JIRA_BASE_URL",
    ).rstrip("/")
    jira_email = _resolve_env_value(
        str(jira.get("email", "")),
        fallback_env_var="JIRA_EMAIL",
    )
    jira_api_token = _resolve_env_value(
        str(jira.get("api_token", "")),
        fallback_env_var="JIRA_API_TOKEN",
    )
    if not jira_base_url:
        raise RuntimeError("jira.base_url must be set in config.yaml or env")
    if not jira_email:
        raise RuntimeError("jira.email must be set in config.yaml or env")
    if not jira_api_token:
        raise RuntimeError("jira.api_token must be set in config.yaml or env")

    return RuntimeConfig(
        slack_channel_id=channel_id,
        timezone=workflow.get("timezone", "Europe/Moscow"),
        heartbeat_active_minutes=int(heartbeat.get("active_minutes", 15)),
        heartbeat_idle_minutes=int(heartbeat.get("idle_minutes", 240)),
        jira_project_keys=[str(v) for v in jira.get("project_keys", [])],
        readiness_owners=readiness_owners,
        release_state_path=storage.get("state_path", "artifacts/workflow_state.json"),
        slack_outbox_path=storage.get("slack_outbox_path", "artifacts/mock_slack_outbox.jsonl"),
        slack_events_path=storage.get("slack_events_path", "artifacts/mock_slack_events.jsonl"),
        jira_outbox_path=storage.get("jira_outbox_path", "artifacts/mock_jira_outbox.jsonl"),
        jira_base_url=jira_base_url,
        jira_email=jira_email,
        jira_api_token=jira_api_token,
        retry_max_attempts=int(retry.get("max_attempts", 3)),
        retry_base_delay_seconds=float(retry.get("base_delay_seconds", 0.5)),
        retry_max_delay_seconds=float(retry.get("max_delay_seconds", 2.0)),
    )
