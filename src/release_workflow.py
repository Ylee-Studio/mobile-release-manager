"""Release train orchestration up to READY_FOR_BRANCH_CUT."""
from __future__ import annotations

import logging
import re
import uuid
from hashlib import sha1
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from .retry import RetryPolicy, call_with_retry
from .state_store import StateStore
from .tools.jira_tools import JiraGateway
from .tools.slack_tools import SlackEvent, SlackGateway
from .workflow_state import ReleaseContext, ReleaseStep, WorkflowState

VERSION_RE = re.compile(r"\b(\d+)\.(\d+)\.(\d+)\b")


def increment_minor_version(previous_version: str | None) -> str:
    if not previous_version:
        return "1.0.0"
    match = VERSION_RE.search(previous_version)
    if not match:
        raise ValueError(f"Invalid previous version: {previous_version}")
    major, minor, patch = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return f"{major}.{minor + 1}.{patch}"


def extract_version_from_text(text: str) -> str | None:
    match = VERSION_RE.search(text)
    if not match:
        return None
    return f"{int(match.group(1))}.{int(match.group(2))}.{int(match.group(3))}"


@dataclass
class RuntimeConfig:
    slack_channel_id: str
    timezone: str
    heartbeat_active_minutes: int
    heartbeat_idle_minutes: int
    jira_project_keys: list[str]
    readiness_owners: dict[str, str]
    release_state_path: str
    slack_outbox_path: str
    slack_events_path: str
    jira_outbox_path: str
    retry_max_attempts: int = 3
    retry_base_delay_seconds: float = 0.5
    retry_max_delay_seconds: float = 2.0


class ReleaseWorkflow:
    """Coordinates orchestrator and release manager actions."""

    def __init__(
        self,
        *,
        config: RuntimeConfig,
        state_store: StateStore,
        slack_gateway: SlackGateway,
        jira_gateway: JiraGateway,
    ):
        self.config = config
        self.state_store = state_store
        self.slack_gateway = slack_gateway
        self.jira_gateway = jira_gateway
        self.logger = logging.getLogger("release_workflow")
        self.retry_policy = RetryPolicy(
            max_attempts=config.retry_max_attempts,
            base_delay_seconds=config.retry_base_delay_seconds,
            max_delay_seconds=config.retry_max_delay_seconds,
        )

    def tick(self, *, now: datetime | None = None) -> WorkflowState:
        state = self.state_store.load()
        trace_id = str(uuid.uuid4())
        self._log(trace_id, state, "tick_start", result="ok")

        events = self.slack_gateway.poll_events()
        for event in events:
            if event.event_id in state.processed_event_ids:
                continue
            self._route_event(event, state, trace_id=trace_id)
            state.mark_event_processed(event.event_id)

        if state.step == ReleaseStep.IDLE and self._is_scheduled_start(now=now):
            self._start_release(state, trigger="schedule", trace_id=trace_id)

        if state.active_release:
            self._run_release_manager(state, trace_id=trace_id)

        self.state_store.save(state)
        self._log(trace_id, state, "tick_end", result="ok")
        return state

    def _route_event(self, event: SlackEvent, state: WorkflowState, *, trace_id: str) -> None:
        if (
            state.step == ReleaseStep.IDLE
            and event.event_type == "manual_start"
            and event.channel_id == self.config.slack_channel_id
        ):
            self._start_release(state, trigger="manual", trace_id=trace_id)
            return
        if not state.active_release:
            return
        self._handle_release_event(event, state, trace_id=trace_id)

    def _start_release(self, state: WorkflowState, *, trigger: str, trace_id: str) -> None:
        version = increment_minor_version(state.previous_release_version)
        state.active_release = ReleaseContext(
            release_version=version,
            step=ReleaseStep.WAIT_START_APPROVAL,
            slack_channel_id=self.config.slack_channel_id,
            readiness_map={team: False for team in self.config.readiness_owners},
        )
        self._log(trace_id, state, "orchestrator_start", tool="internal", result=trigger)
        self._send_start_approval(state, trace_id=trace_id)

    def _send_start_approval(self, state: WorkflowState, *, trace_id: str) -> None:
        assert state.active_release
        action_key = f"start-approval:{state.active_release.release_version}"
        if state.is_action_done(action_key):
            return
        text = (
            f"Подтвердите старт релизного трейна {state.active_release.release_version}. "
            "Если нужен другой номер для релиза, напишите в треде."
        )
        response = self._call_with_checkpoint(
            state=state,
            tool="slack_approve",
            trace_id=trace_id,
            fn=lambda: self.slack_gateway.send_approve(
                channel_id=self.config.slack_channel_id,
                text=text,
                approve_label="Подтвердить",
            ),
        )
        state.active_release.message_ts["start_approval"] = response["message_ts"]
        state.active_release.thread_ts["start_approval"] = response["message_ts"]
        state.mark_action_done(action_key)

    def _run_release_manager(self, state: WorkflowState, *, trace_id: str) -> None:
        assert state.active_release
        if state.step == ReleaseStep.WAIT_START_APPROVAL:
            return
        if state.step == ReleaseStep.RELEASE_MANAGER_CREATED:
            self._create_jira_release(state, trace_id=trace_id)
        if state.step == ReleaseStep.JIRA_RELEASE_CREATED:
            self._announce_meeting(state, trace_id=trace_id)
        if state.step == ReleaseStep.WAIT_MEETING_CONFIRMATION:
            return
        if state.step == ReleaseStep.WAIT_READINESS_CONFIRMATIONS:
            self._update_readiness_message(state, trace_id=trace_id, force=False)

    def _handle_release_event(self, event: SlackEvent, state: WorkflowState, *, trace_id: str) -> None:
        assert state.active_release

        if state.step == ReleaseStep.WAIT_START_APPROVAL:
            self._handle_start_approval_event(event, state, trace_id=trace_id)
            return
        if state.step == ReleaseStep.WAIT_MEETING_CONFIRMATION:
            if "провели" in event.text.lower():
                state.active_release.set_step(ReleaseStep.WAIT_READINESS_CONFIRMATIONS)
                self._publish_readiness_message(state, trace_id=trace_id)
            return
        if state.step == ReleaseStep.WAIT_READINESS_CONFIRMATIONS:
            updated = self._handle_readiness_confirmation(event, state)
            if updated:
                self._update_readiness_message(state, trace_id=trace_id, force=True)
            if state.active_release and all(state.active_release.readiness_map.values()):
                state.active_release.set_step(ReleaseStep.READY_FOR_BRANCH_CUT)

    def _handle_start_approval_event(self, event: SlackEvent, state: WorkflowState, *, trace_id: str) -> None:
        assert state.active_release
        if event.thread_ts != state.active_release.thread_ts.get("start_approval"):
            return

        alternative = extract_version_from_text(event.text)
        if alternative and alternative != state.active_release.release_version:
            state.active_release.release_version = alternative
            text = (
                f"Подтвердите старт релизного трейна {alternative}. "
                "Если нужен другой номер для релиза, напишите в треде."
            )
            self._call_with_checkpoint(
                state=state,
                tool="slack_update",
                trace_id=trace_id,
                fn=lambda: self.slack_gateway.update_message(
                    channel_id=self.config.slack_channel_id,
                    message_ts=state.active_release.message_ts["start_approval"],
                    text=text,
                ),
            )
            return

        if event.event_type == "approval_confirmed":
            state.active_release.set_step(ReleaseStep.RELEASE_MANAGER_CREATED)

    def _create_jira_release(self, state: WorkflowState, *, trace_id: str) -> None:
        assert state.active_release
        action_key = f"jira-release:{state.active_release.release_version}"
        if state.is_action_done(action_key):
            state.active_release.set_step(ReleaseStep.JIRA_RELEASE_CREATED)
            return
        self._call_with_checkpoint(
            state=state,
            tool="create_crossspace_release",
            trace_id=trace_id,
            fn=lambda: self.jira_gateway.create_crossspace_release(
                version=state.active_release.release_version,
                project_keys=self.config.jira_project_keys,
            ),
        )
        state.mark_action_done(action_key)
        state.active_release.set_step(ReleaseStep.JIRA_RELEASE_CREATED)

    def _announce_meeting(self, state: WorkflowState, *, trace_id: str) -> None:
        assert state.active_release
        action_key = f"meeting-approval:{state.active_release.release_version}"
        if state.is_action_done(action_key):
            state.active_release.set_step(ReleaseStep.WAIT_MEETING_CONFIRMATION)
            return
        jql = (
            f"fixVersion = {state.active_release.release_version} "
            "ORDER BY project ASC, type DESC, key ASC"
        )
        text = (
            "Проведите встречу фиксации релиза. "
            f"<https://instories.atlassian.net/issues/?jql={jql}|Ссылка на релизные задачи>. "
            'В 15:00 будет код-фриз.'
        )
        response = self._call_with_checkpoint(
            state=state,
            tool="slack_approve",
            trace_id=trace_id,
            fn=lambda: self.slack_gateway.send_approve(
                channel_id=self.config.slack_channel_id,
                text=text,
                approve_label="Провели",
            ),
        )
        state.active_release.message_ts["meeting_approval"] = response["message_ts"]
        state.active_release.thread_ts["meeting_approval"] = response["message_ts"]
        state.mark_action_done(action_key)
        state.active_release.set_step(ReleaseStep.WAIT_MEETING_CONFIRMATION)

    def _publish_readiness_message(self, state: WorkflowState, *, trace_id: str) -> None:
        assert state.active_release
        action_key = f"readiness-message:{state.active_release.release_version}"
        if state.is_action_done(action_key):
            return
        text = self._render_readiness_message(state)
        response = self._call_with_checkpoint(
            state=state,
            tool="slack_message",
            trace_id=trace_id,
            fn=lambda: self.slack_gateway.send_message(
                channel_id=self.config.slack_channel_id,
                text=text,
            ),
        )
        state.active_release.message_ts["readiness"] = response["message_ts"]
        state.active_release.thread_ts["readiness"] = response["message_ts"]
        state.mark_action_done(action_key)

    def _handle_readiness_confirmation(self, event: SlackEvent, state: WorkflowState) -> bool:
        assert state.active_release
        if event.thread_ts != state.active_release.thread_ts.get("readiness"):
            return False
        message = event.text.lower()
        updated = False
        for team in state.active_release.readiness_map:
            if state.active_release.readiness_map[team]:
                continue
            if team.lower() in message and "готов" in message:
                state.active_release.readiness_map[team] = True
                updated = True
        return updated

    def _update_readiness_message(self, state: WorkflowState, *, trace_id: str, force: bool) -> None:
        assert state.active_release
        ts = state.active_release.message_ts.get("readiness")
        if not ts:
            return
        text = self._render_readiness_message(state)
        text_hash = sha1(text.encode("utf-8")).hexdigest()
        action_key = f"readiness-update:{state.active_release.release_version}:{text_hash}"
        if not force and state.is_action_done(action_key):
            return
        self._call_with_checkpoint(
            state=state,
            tool="slack_update",
            trace_id=trace_id,
            fn=lambda: self.slack_gateway.update_message(
                channel_id=self.config.slack_channel_id,
                message_ts=ts,
                text=text,
            ),
        )
        state.mark_action_done(action_key)

    def _render_readiness_message(self, state: WorkflowState) -> str:
        assert state.active_release
        version = state.active_release.release_version
        jql = (
            f"fixVersion = {version} "
            "ORDER BY project ASC, type DESC, key ASC"
        )
        lines = [f"Релиз <https://instories.atlassian.net/issues/?jql={jql}|{version}>", "", "Статус готовности к срезу:"]
        for team, owner in self.config.readiness_owners.items():
            emoji = ":white_check_mark:" if state.active_release.readiness_map.get(team) else ":hourglass_flowing_sand:"
            lines.append(f"{emoji} {team} @{owner}")
        lines.extend(
            [
                "",
                "Напишите в треде по готовности своей части.",
                "**Важное напоминание** - все задачи, не влитые в ветку RC до 15:00 МСК едут в релиз только после одобрения QA",
            ]
        )
        return "\n".join(lines)

    def _call_with_checkpoint(self, *, state: WorkflowState, tool: str, trace_id: str, fn):
        assert state.active_release
        state.add_checkpoint(
            phase="before",
            step=state.active_release.step,
            tool=tool,
            release_version=state.active_release.release_version,
        )
        result = call_with_retry(
            fn,
            policy=self.retry_policy,
            on_retry=lambda attempt, exc: self._log(
                trace_id,
                state,
                task="retry",
                step=state.step.value,
                tool=tool,
                result=f"attempt={attempt} error={exc}",
            ),
        )
        state.add_checkpoint(
            phase="after",
            step=state.active_release.step,
            tool=tool,
            release_version=state.active_release.release_version,
        )
        self._log(trace_id, state, task="tool_call", tool=tool, result="ok")
        return result

    def _is_scheduled_start(self, *, now: datetime | None) -> bool:
        tz = ZoneInfo(self.config.timezone)
        current = now.astimezone(tz) if now else datetime.now(tz)
        return current.weekday() == 4 and current.hour == 11

    def _log(
        self,
        trace_id: str,
        state: WorkflowState,
        task: str,
        *,
        step: str | None = None,
        tool: str | None = None,
        result: str,
    ) -> None:
        self.logger.info(
            "trace_id=%s release_version=%s agent=%s task=%s step=%s tool=%s result=%s latency_ms=%s",
            trace_id,
            state.active_release.release_version if state.active_release else "-",
            "orchestrator" if state.step in {ReleaseStep.IDLE, ReleaseStep.WAIT_START_APPROVAL} else "release_manager",
            task,
            step or state.step.value,
            tool or "-",
            result,
            "-",
        )
