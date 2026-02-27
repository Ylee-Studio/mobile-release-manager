"""Operational policies for the release workflow runtime."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo


@dataclass
class QuietHoursPolicy:
    timezone: str
    start: str
    end: str

    def is_quiet_now(self, now: Optional[datetime] = None) -> bool:
        tz = ZoneInfo(self.timezone)
        now = now.astimezone(tz) if now else datetime.now(tz)
        start_time = _parse_time(self.start)
        end_time = _parse_time(self.end)
        if start_time < end_time:
            return start_time <= now.time() < end_time
        # window wraps midnight
        return now.time() >= start_time or now.time() < end_time


@dataclass
class ApprovalPolicy:
    required: bool
    approver: str

    def validate(self, approved_by: Optional[str]) -> None:
        if not self.required:
            return
        if not approved_by:
            raise PermissionError(
                "Approval required before sending release communications. Pass --approved-by."
            )


@dataclass
class RateLimitPolicy:
    slack_messages_per_hour: int
    jira_updates_per_hour: int


@dataclass
class PolicyConfig:
    max_interactions: int
    max_iterations: int
    max_execution_time_seconds: int | None
    agent_verbose: bool
    quiet_hours: QuietHoursPolicy
    approvals: ApprovalPolicy
    rate_limits: RateLimitPolicy

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], *, default_max_interactions: int = 4) -> "PolicyConfig":
        quiet_cfg = raw.get("quiet_hours", {})
        approval_cfg = raw.get("approvals", {})
        rate_cfg = raw.get("rate_limits", {})
        return cls(
            max_interactions=raw.get("max_interactions", default_max_interactions),
            max_iterations=raw.get("max_iterations", raw.get("max_interactions", default_max_interactions)),
            max_execution_time_seconds=raw.get("max_execution_time_seconds"),
            agent_verbose=raw.get("agent_verbose", True),
            quiet_hours=QuietHoursPolicy(
                timezone=quiet_cfg.get("timezone", "UTC"),
                start=quiet_cfg.get("start", "22:00"),
                end=quiet_cfg.get("end", "06:00"),
            ),
            approvals=ApprovalPolicy(
                required=approval_cfg.get("required", True),
                approver=approval_cfg.get("approver", "release-director@example.com"),
            ),
            rate_limits=RateLimitPolicy(
                slack_messages_per_hour=rate_cfg.get("slack_messages_per_hour", 12),
                jira_updates_per_hour=rate_cfg.get("jira_updates_per_hour", 6),
            ),
        )


def _parse_time(value: str) -> time:
    hour, minute = value.split(":", maxsplit=1)
    return time(hour=int(hour), minute=int(minute))
