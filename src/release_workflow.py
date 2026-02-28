"""Release workflow runtime driven by deterministic state machine."""
from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .runtime_engine import RuntimeCoordinator, RuntimeDecision
from .state_store import WorkflowStateStore
from .tool_calling import ToolCallValidationError, validate_and_normalize_tool_calls
from .tools.github_tools import GitHubActionInput, GitHubGateway
from .tools.slack_tools import (
    SlackApproveInput,
    SlackGateway,
    SlackMessageInput,
    SlackUpdateInput,
)
from .workflow_state import ReleaseStep, WorkflowState

MANUAL_RELEASE_MESSAGE = (
    "Подтвердите создание релиза {release_version} в JIRA. "
    "<https://instories.atlassian.net/jira/plans/1/scenarios/1/releases|JIRA>"
)
RC_BRANCH_APPROVAL_MESSAGE = (
    "Выделил RC ветку. Подтвердите "
    "<https://github.com/Ylee-Studio/instories-ios/actions|готовность> ветки.\n"
    "- FontThumbnailManager.Spec обновлен при необходимости\n"
    "- В L10n.extension.swift нет временной локализации\n"
    "- В TestTemplates.swift нет шаблонов"
)

@dataclass
class RuntimeConfig:
    slack_channel_id: str
    slack_bot_token: str
    timezone: str
    jira_project_keys: list[str]
    readiness_owners: dict[str, str]
    memory_db_path: str
    audit_log_path: str
    slack_events_path: str
    github_repo: str = "Ylee-Studio/instories-ios"
    github_token: str = ""
    github_workflow_file: str = "create_release_branch.yml"
    github_ref: str = "dev"
    github_poll_interval_seconds: int = 30


WORKFLOW_STATE_KEY = "release_workflow_state"
WORKFLOW_MEMORY_SCOPE = "/workflow/state"
WORKFLOW_MEMORY_SOURCE = "release_workflow"
WORKFLOW_MEMORY_CATEGORY = "workflow_state"


class ReleaseWorkflow:
    """Thin runtime that delegates transitions to state machine + runtime engine."""

    def __init__(
        self,
        *,
        config: RuntimeConfig,
        slack_gateway: SlackGateway,
        state_store: WorkflowStateStore | Any | None = None,
        memory: Any | None = None,
        runtime_engine: RuntimeCoordinator | Any | None = None,
        legacy_runtime: RuntimeCoordinator | Any | None = None,
        github_gateway: GitHubGateway | None = None,
    ):
        self.config = config
        self.state_store = state_store or _build_legacy_state_store(memory=memory, fallback_path=config.memory_db_path)
        self.slack_gateway = slack_gateway
        self.runtime_engine = runtime_engine or legacy_runtime
        if self.runtime_engine is None:
            raise RuntimeError("runtime engine is required")
        self.audit_log_path = Path(config.audit_log_path)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("release_workflow")
        self._state = self.state_store.load()
        self.github_gateway = github_gateway or GitHubGateway(
            token=config.github_token,
            repo=config.github_repo,
        )

    def tick(
        self,
        *,
        now: datetime | None = None,
        trigger_reason: str = "event_trigger",
    ) -> WorkflowState:
        state = WorkflowState.from_dict(self._state.to_dict())
        previous_state = WorkflowState.from_dict(state.to_dict())
        events = self.slack_gateway.poll_events()
        trace_id = str(uuid.uuid4())

        self.logger.info("processing events: %s", events)
        self.logger.info("adding processing reactions before runtime decision")
        self._add_processing_reactions(events=events)

        decision = self.start_or_continue_release(
            state=state,
            events=events,
            now=now,
            trigger_reason=trigger_reason,
        )
        try:
            decision.tool_calls = validate_and_normalize_tool_calls(
                decision.tool_calls,
                schema_by_tool={
                    "slack_message": SlackMessageInput,
                    "slack_approve": SlackApproveInput,
                    "slack_update": SlackUpdateInput,
                    "github_action": GitHubActionInput,
                },
            )
        except ToolCallValidationError as exc:
            self.logger.warning("tool_call validation failed: %s", exc)
            decision = RuntimeDecision(
                next_step=state.step,
                next_state=state,
                audit_reason=f"invalid_tool_call:{exc}",
                actor=decision.actor,
                tool_calls=[],
            )
        self._execute_tool_calls(
            decision=decision,
            events=events,
            previous_state=previous_state,
        )
        self._confirm_manual_override_request(decision=decision, events=events)
        self._confirm_release_ready_request(
            previous_state=previous_state,
            decision=decision,
            events=events,
        )
        next_state = decision.next_state
        self.state_store.save(state=next_state, reason=decision.audit_reason)
        self._state = WorkflowState.from_dict(next_state.to_dict())
        queue_remaining = self.slack_gateway.pending_events_count()
        self._append_audit(
            trace_id=trace_id,
            previous_state=previous_state,
            decision=decision,
            next_state=next_state,
            events=[_event_to_dict(event) for event in events],
        )
        self.logger.info(
            "tick trace_id=%s reason=%s event_count=%s queue_remaining=%s",
            trace_id,
            trigger_reason,
            len(events),
            queue_remaining,
        )
        return next_state

    def start_or_continue_release(
        self,
        *,
        state: WorkflowState,
        events: list[Any],
        now: datetime | None,
        trigger_reason: str,
    ) -> RuntimeDecision:
        kickoff = getattr(self.runtime_engine, "kickoff", None)
        if callable(kickoff):
            return kickoff(
                state=state,
                events=events,
                config=self.config,
                now=now,
                trigger_reason=trigger_reason,
            )
        decide = getattr(self.runtime_engine, "decide", None)
        if callable(decide):
            return decide(
                state=state,
                events=events,
                config=self.config,
                now=now,
                trigger_reason=trigger_reason,
            )
        raise RuntimeError("runtime engine must provide kickoff() or decide()")

    def _add_processing_reactions(self, *, events: list[Any]) -> None:
        for event in events:
            channel_id = str(getattr(event, "channel_id", "") or "").strip()
            message_ts = str(getattr(event, "message_ts", "") or "").strip()
            if not channel_id or not message_ts:
                continue
            try:
                self.slack_gateway.add_reaction(
                    channel_id=channel_id,
                    message_ts=message_ts,
                    emoji="eyes",
                )
                self.logger.info(
                    "processing reaction added before runtime decision channel=%s ts=%s emoji=%s",
                    channel_id,
                    message_ts,
                    "eyes",
                )
            except Exception as exc:  # pragma: no cover - runtime safety
                self.logger.warning(
                    "failed to add processing reaction channel=%s ts=%s: %s",
                    channel_id,
                    message_ts,
                    exc,
                )

    def _append_audit(
        self,
        *,
        trace_id: str,
        previous_state: WorkflowState,
        decision: RuntimeDecision,
        next_state: WorkflowState,
        events: list[dict[str, Any]],
    ) -> None:
        incoming_slack_messages = _extract_event_messages(events)
        payload = {
            "trace_id": trace_id,
            "actor": decision.actor,
            "flow_lifecycle": decision.flow_lifecycle,
            "flow_event": _flow_event_from_lifecycle(decision.flow_lifecycle),
            "flow_id_lookup_status": _audit_field(decision.audit_reason, "flow_id_lookup_status"),
            "pending_key": _audit_field(decision.audit_reason, "pending_key"),
            "event_count": len(events),
            "events": events,
            "incoming_slack_messages": incoming_slack_messages,
            "previous_step": previous_state.step.value,
            "next_step": decision.next_step.value,
            "tool_calls": decision.tool_calls,
            "audit_reason": decision.audit_reason,
            "state_patch": decision.state_patch,
            "next_state": next_state.to_dict(),
            "at": datetime.now(timezone.utc).isoformat(),
        }
        with self.audit_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self.logger.info(
            "trace_id=%s lifecycle=%s decision next_step=%s reason=%s events=%s incoming_slack_messages=%s",
            trace_id,
            decision.flow_lifecycle,
            decision.next_step.value,
            decision.audit_reason,
            len(events),
            incoming_slack_messages,
        )

    def _execute_tool_calls(
        self,
        *,
        decision: RuntimeDecision,
        events: list[Any],
        previous_state: WorkflowState,
    ) -> None:
        for call in decision.tool_calls:
            if not isinstance(call, dict):
                continue
            tool_name = str(call.get("tool", "")).strip()
            if tool_name == "slack_approve":
                self._execute_slack_approve(call=call, decision=decision, events=events)
            if tool_name == "slack_message":
                self._execute_slack_message(
                    call=call,
                    decision=decision,
                    events=events,
                    previous_state=previous_state,
                )
            if tool_name == "slack_update":
                self._execute_slack_update(call=call, decision=decision, events=events)
            if tool_name == "github_action":
                self._execute_github_action(call=call, decision=decision, events=events)

    def _execute_github_action(
        self,
        *,
        call: dict[str, Any],
        decision: RuntimeDecision,
        events: list[Any],
    ) -> None:
        args_raw = call.get("args", {})
        args = args_raw if isinstance(args_raw, dict) else {}
        workflow_file = str(args.get("workflow_file") or self.config.github_workflow_file).strip()
        if not workflow_file:
            return
        ref = str(args.get("ref") or self.config.github_ref).strip() or "dev"
        inputs_raw = args.get("inputs", {})
        inputs = inputs_raw if isinstance(inputs_raw, dict) else {}

        release = decision.next_state.active_release
        if release is None:
            # Stateless trigger mode: dispatch workflow without touching workflow state.
            try:
                self.github_gateway.trigger_workflow(
                    workflow_file=workflow_file,
                    ref=ref,
                    inputs=inputs,
                )
            except Exception as exc:  # pragma: no cover - runtime safety
                self.logger.exception("github_action stateless trigger failed: %s", exc)
                return
            if _is_build_rc_workflow(workflow_file):
                self._add_white_check_mark_for_latest_message(
                    events=events,
                    reason="build_rc_dispatched",
                )
            return

        is_branch_cut_workflow = workflow_file == self.config.github_workflow_file
        if is_branch_cut_workflow and release.github_action_run_id:
            # Prevent duplicate dispatch across repeated ticks.
            return
        if is_branch_cut_workflow and "version" not in inputs:
            inputs["version"] = release.release_version
        try:
            run = self.github_gateway.trigger_workflow(
                workflow_file=workflow_file,
                ref=ref,
                inputs=inputs,
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            self.logger.exception("github_action failed for release=%s: %s", release.release_version, exc)
            return
        if not is_branch_cut_workflow:
            if _is_build_rc_workflow(workflow_file):
                self._add_white_check_mark_for_latest_message(
                    events=events,
                    reason="build_rc_dispatched",
                )
            return
        release.github_action_run_id = run.run_id
        release.github_action_status = run.status
        release.github_action_conclusion = run.conclusion
        release.github_action_last_polled_at = datetime.now(timezone.utc).isoformat()

    def _confirm_manual_override_request(
        self,
        *,
        decision: RuntimeDecision,
        events: list[Any],
    ) -> None:
        if not str(decision.audit_reason or "").startswith("manual_status_override:"):
            return
        self._add_white_check_mark_for_latest_message(
            events=events,
            reason="manual_status_override",
        )

    def _add_white_check_mark_for_latest_message(
        self,
        *,
        events: list[Any],
        reason: str,
    ) -> None:
        message_event = _extract_latest_message_event(events)
        if message_event is None:
            return
        channel_id = str(getattr(message_event, "channel_id", "") or "").strip()
        message_ts = str(getattr(message_event, "message_ts", "") or "").strip()
        if not channel_id or not message_ts:
            self.logger.warning(
                "skip confirmation reaction reason=%s missing channel/message ts",
                reason,
            )
            return
        try:
            self.slack_gateway.add_reaction(
                channel_id=channel_id,
                message_ts=message_ts,
                emoji="white_check_mark",
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            self.logger.warning(
                "failed to add confirmation reaction reason=%s channel=%s ts=%s: %s",
                reason,
                channel_id,
                message_ts,
                exc,
            )

    def _execute_slack_approve(
        self,
        *,
        call: dict[str, Any],
        decision: RuntimeDecision,
        events: list[Any],
    ) -> None:
        release = decision.next_state.active_release
        if release is None:
            return
        if decision.next_step == ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION:
            message_key = "manual_release_confirmation"
            default_text = MANUAL_RELEASE_MESSAGE.format(release_version=release.release_version)
            default_approve_label = "Релиз создан"
        elif decision.next_step == ReleaseStep.WAIT_MEETING_CONFIRMATION:
            message_key = "meeting_confirmation"
            release_issues_url = _readiness_issues_url(release.release_version)
            default_text = (
                "Подтвердите, что встреча фиксации "
                f"<{release_issues_url}|релиза> уже прошла."
            )
            default_approve_label = "Митинг проведен"
        elif decision.next_step == ReleaseStep.WAIT_BRANCH_CUT_APPROVAL:
            message_key = "branch_cut_approval"
            default_text = RC_BRANCH_APPROVAL_MESSAGE
            default_approve_label = "Подтвердить"
        else:
            message_key = "start_approval"
            default_text = f"Подтвердите старт релизного трейна {release.release_version}."
            default_approve_label = "Подтвердить"

        channel_id = release.slack_channel_id or _extract_event_channel_id(events) or self.config.slack_channel_id
        if not channel_id:
            self.logger.warning("skip slack_approve: empty channel_id for release=%s", release.release_version)
            return

        args_raw = call.get("args", {})
        args = args_raw if isinstance(args_raw, dict) else {}
        text = str(args.get("text") or "").strip() or default_text
        approve_label = str(args.get("approve_label") or default_approve_label)
        reject_label = str(args.get("reject_label") or "").strip() or None
        if message_key != "start_approval":
            reject_label = None

        decision.next_state.checkpoints.append(
            {
                "phase": "before",
                "tool": "slack_approve",
                "release_version": release.release_version,
                "step": decision.next_step.value,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )
        try:
            result = self.slack_gateway.send_approve(
                channel_id=channel_id,
                text=text,
                approve_label=approve_label,
                reject_label=reject_label,
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            self.logger.exception("slack_approve failed for release=%s: %s", release.release_version, exc)
            return

        message_ts = str(result.get("message_ts", "")).strip()
        if not message_ts:
            self.logger.warning("slack_approve returned empty message_ts for release=%s", release.release_version)
            return

        release.slack_channel_id = channel_id
        release.message_ts[message_key] = message_ts
        release.thread_ts[message_key] = message_ts
        decision.next_state.checkpoints.append(
            {
                "phase": "after",
                "tool": "slack_approve",
                "release_version": release.release_version,
                "step": decision.next_step.value,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _execute_slack_message(
        self,
        *,
        call: dict[str, Any],
        decision: RuntimeDecision,
        events: list[Any],
        previous_state: WorkflowState | None = None,
    ) -> None:
        release = decision.next_state.active_release
        args_raw = call.get("args", {})
        args = args_raw if isinstance(args_raw, dict) else {}
        if release is not None:
            message_key = "generic_message"
        else:
            message_key = ""
        text = str(args.get("text") or "").strip()
        if not text:
            self.logger.warning("skip slack_message: empty text after validation")
            return

        readiness_item = _readiness_item_from_audit_reason(decision.audit_reason)
        if (
            release is not None
            and decision.next_step == ReleaseStep.WAIT_READINESS_CONFIRMATIONS
            and readiness_item
        ):
            checklist_ts = (
                str(release.thread_ts.get("readiness", "") or "").strip()
                or str(release.thread_ts.get("generic_message", "") or "").strip()
                or str(release.message_ts.get("generic_message", "") or "").strip()
            )
            if checklist_ts:
                channel_id = (
                    str(args.get("channel_id") or "").strip()
                    or release.slack_channel_id
                    or _extract_event_channel_id(events)
                    or self.config.slack_channel_id
                )
                if not channel_id:
                    self.logger.warning(
                        "skip readiness checklist update: empty channel_id for release=%s",
                        release.release_version,
                    )
                    return
                checklist_text = _build_readiness_checklist_text(
                    release_version=release.release_version,
                    readiness_owners=self.config.readiness_owners,
                    readiness_map=release.readiness_map,
                )
                decision.next_state.checkpoints.append(
                    {
                        "phase": "before",
                        "tool": "slack_update",
                        "release_version": release.release_version,
                        "step": decision.next_step.value,
                        "at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                try:
                    self.slack_gateway.update_message(
                        channel_id=channel_id,
                        message_ts=checklist_ts,
                        text=checklist_text,
                    )
                except Exception as exc:  # pragma: no cover - runtime safety
                    self.logger.exception(
                        "readiness checklist update failed for release=%s: %s",
                        release.release_version,
                        exc,
                    )
                    return
                release.slack_channel_id = channel_id
                decision.next_state.checkpoints.append(
                    {
                        "phase": "after",
                        "tool": "slack_update",
                        "release_version": release.release_version,
                        "step": decision.next_step.value,
                        "at": datetime.now(timezone.utc).isoformat(),
                    }
                )
            else:
                self.logger.warning(
                    "skip readiness checklist update: empty message_ts for release=%s",
                    release.release_version,
                )
            return

        if release is not None and decision.next_step == ReleaseStep.WAIT_READINESS_CONFIRMATIONS:
            text = _normalize_readiness_message_text(
                text=text,
                release_version=release.release_version,
                readiness_owners=self.config.readiness_owners,
            )

        channel_id = (
            str(args.get("channel_id") or "").strip()
            or (release.slack_channel_id if release else "")
            or _extract_event_channel_id(events)
            or self.config.slack_channel_id
        )
        if not channel_id:
            release_version = release.release_version if release else "n/a"
            self.logger.warning("skip slack_message: empty channel_id for release=%s", release_version)
            return

        if _must_be_channel_level_message(text=text, next_step=decision.next_step) or (
            previous_state is not None
            and _is_final_release_congratulation(
                previous_step=previous_state.step,
                next_step=decision.next_step,
                events=events,
            )
        ):
            # Readiness checklist and RC-branch-ready notice must be top-level channel messages.
            thread_ts_value = None
        else:
            thread_ts = str(args.get("thread_ts") or "").strip()
            if not thread_ts and release is not None:
                thread_ts = release.thread_ts.get("manual_release_confirmation")
            if not thread_ts and release is not None:
                thread_ts = release.thread_ts.get("start_approval")
            if not thread_ts:
                thread_ts = _extract_latest_message_thread_ts(events)
            thread_ts_value = thread_ts or None

        decision.next_state.checkpoints.append(
            {
                "phase": "before",
                "tool": "slack_message",
                "release_version": release.release_version if release else "",
                "step": decision.next_step.value,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )
        try:
            result = self.slack_gateway.send_message(
                channel_id=channel_id,
                text=text,
                thread_ts=thread_ts_value,
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            release_version = release.release_version if release else "n/a"
            self.logger.exception("slack_message failed for release=%s: %s", release_version, exc)
            return

        message_ts = str(result.get("message_ts", "")).strip()
        if message_ts and release is not None and message_key:
            release.message_ts[message_key] = message_ts
        if (
            release is not None
            and decision.next_step == ReleaseStep.WAIT_READINESS_CONFIRMATIONS
            and message_ts
        ):
            checklist_ts = thread_ts_value or message_ts
            # Readiness replies are validated against the checklist thread key.
            # Keep both generic and explicit readiness keys aligned with the posted checklist ts.
            release.thread_ts[message_key] = checklist_ts
            release.thread_ts["readiness"] = checklist_ts
            release.message_ts["readiness"] = message_ts
        if release is not None:
            release.slack_channel_id = channel_id
        decision.next_state.checkpoints.append(
            {
                "phase": "after",
                "tool": "slack_message",
                "release_version": release.release_version if release else "",
                "step": decision.next_step.value,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _execute_slack_update(
        self,
        *,
        call: dict[str, Any],
        decision: RuntimeDecision,
        events: list[Any],
    ) -> None:
        release = decision.next_state.active_release
        release_version = release.release_version if release is not None else "n/a"

        args_raw = call.get("args", {})
        args = args_raw if isinstance(args_raw, dict) else {}
        approval_event = _extract_latest_approval_event(events)
        event_message_ts = str(getattr(approval_event, "message_ts", "") or "").strip() if approval_event else ""
        message_ts = str(args.get("message_ts") or "").strip() or event_message_ts
        if not message_ts:
            self.logger.warning("skip slack_update: empty message_ts for release=%s", release_version)
            return

        channel_id = (
            str(args.get("channel_id") or "").strip()
            or (release.slack_channel_id if release is not None else "")
            or _extract_event_channel_id(events)
            or self.config.slack_channel_id
        )
        if not channel_id:
            self.logger.warning("skip slack_update: empty channel_id for release=%s", release_version)
            return

        text = str(args.get("text") or "").strip()
        if approval_event is not None:
            metadata = getattr(approval_event, "metadata", None)
            trigger_message_text = ""
            if isinstance(metadata, dict):
                trigger_message_text = str(metadata.get("trigger_message_text") or "").strip()
            approval_suffix = "Подтверждено :white_check_mark:"
            if decision.next_step == ReleaseStep.WAIT_RC_READINESS:
                approval_suffix = f"{approval_suffix}\nя запустил сборку"
            if str(getattr(approval_event, "event_type", "")).strip() == "approval_rejected":
                approval_suffix = "Отклонено :x:"
            text = _with_approval_suffix(
                base_text=trigger_message_text or text,
                suffix=approval_suffix,
            )
        elif text and not text.endswith(":white_check_mark:") and not _is_readiness_checklist_message(text):
            text = f"{text} :white_check_mark:"
        if not text:
            self.logger.warning("skip slack_update: empty text for release=%s", release_version)
            return

        if release is not None:
            decision.next_state.checkpoints.append(
                {
                    "phase": "before",
                    "tool": "slack_update",
                    "release_version": release.release_version,
                    "step": decision.next_step.value,
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            )
        try:
            self.slack_gateway.update_message(
                channel_id=channel_id,
                message_ts=message_ts,
                text=text,
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            self.logger.exception("slack_update failed for release=%s: %s", release_version, exc)
            return

        if release is not None:
            release.slack_channel_id = channel_id
            decision.next_state.checkpoints.append(
                {
                    "phase": "after",
                    "tool": "slack_update",
                    "release_version": release.release_version,
                    "step": decision.next_step.value,
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            )

    def _confirm_release_ready_request(
        self,
        *,
        previous_state: WorkflowState,
        decision: RuntimeDecision,
        events: list[Any],
    ) -> None:
        if previous_state.step != ReleaseStep.WAIT_RC_READINESS:
            return
        if decision.next_step != ReleaseStep.IDLE:
            return
        if decision.next_state.active_release is not None:
            return
        message_event = _extract_latest_message_event(events)
        if message_event is None:
            return
        if not _is_release_ready_message(getattr(message_event, "text", "")):
            return
        previous_release = previous_state.active_release
        if previous_release is not None:
            decision.next_state.previous_release_version = previous_release.release_version
        decision.next_state.previous_release_completed_at = datetime.now(timezone.utc).isoformat()
        self._add_white_check_mark_for_latest_message(
            events=events,
            reason="release_ready_confirmed",
        )


class _LegacyMemoryStateStore:
    """Adapter to keep backward compatibility with legacy in-memory tests."""

    def __init__(self, *, memory: Any):
        self.memory = memory

    def load(self) -> WorkflowState:
        if self.memory is None:
            return WorkflowState()
        try:
            matches = self.memory.recall(
                query=WORKFLOW_STATE_KEY,
                scope=WORKFLOW_MEMORY_SCOPE,
                categories=[WORKFLOW_MEMORY_CATEGORY],
                limit=100,
                depth="shallow",
                source=WORKFLOW_MEMORY_SOURCE,
            )
        except Exception:
            return WorkflowState()
        if not isinstance(matches, list):
            return WorkflowState()
        for match in matches:
            record = getattr(match, "record", None)
            if record is None and isinstance(match, dict):
                entry = match
            else:
                entry = {
                    "content": getattr(record, "content", ""),
                    "metadata": getattr(record, "metadata", {}),
                }
            raw_state = _extract_workflow_state(entry)
            if isinstance(raw_state, dict):
                return WorkflowState.from_dict(raw_state)
        return WorkflowState()

    def save(self, *, state: WorkflowState, reason: str) -> None:
        if self.memory is None:
            return
        state_payload = state.to_dict()
        metadata: dict[str, Any] = {
            "state_key": WORKFLOW_STATE_KEY,
            "state": state_payload,
            "reason": reason,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        content = json.dumps(state_payload, ensure_ascii=True)
        try:
            self.memory.remember(
                content=content,
                scope=WORKFLOW_MEMORY_SCOPE,
                categories=[WORKFLOW_MEMORY_CATEGORY],
                metadata=metadata,
                importance=1.0,
                source=WORKFLOW_MEMORY_SOURCE,
            )
        except Exception:
            return


def _build_legacy_state_store(*, memory: Any | None, fallback_path: str) -> Any:
    if memory is not None and hasattr(memory, "remember") and hasattr(memory, "recall"):
        return _LegacyMemoryStateStore(memory=memory)
    return WorkflowStateStore(state_path=fallback_path)


def _extract_workflow_state(entry: dict[str, Any]) -> dict[str, Any] | None:
    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        state = metadata.get("state")
        if isinstance(state, dict):
            return state
        if isinstance(state, str):
            try:
                parsed = json.loads(state)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, dict):
                return parsed

    state = entry.get("state")
    if isinstance(state, dict):
        return state
    content = entry.get("content")
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    if isinstance(state, str):
        try:
            parsed = json.loads(state)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _event_to_dict(event: Any) -> dict[str, Any]:
    return {
        "event_id": getattr(event, "event_id", ""),
        "event_type": getattr(event, "event_type", ""),
        "channel_id": getattr(event, "channel_id", ""),
        "text": getattr(event, "text", ""),
        "thread_ts": getattr(event, "thread_ts", None),
        "message_ts": getattr(event, "message_ts", None),
        "metadata": getattr(event, "metadata", None),
    }


def _extract_event_messages(events: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for event in events:
        event_id = str(event.get("event_id", ""))
        event_type = str(event.get("event_type", ""))
        text = str(event.get("text", ""))
        if not text:
            continue
        messages.append(
            {
                "event_id": event_id,
                "event_type": event_type,
                "text": text,
            }
        )
    return messages


def _extract_event_channel_id(events: list[Any]) -> str:
    for event in events:
        channel_id = str(getattr(event, "channel_id", "")).strip()
        if channel_id:
            return channel_id
    return ""


def _extract_latest_approval_event(events: list[Any]) -> Any | None:
    for event in reversed(events):
        event_type = str(getattr(event, "event_type", "")).strip()
        if event_type in {"approval_confirmed", "approval_rejected"}:
            return event
    return None


def _extract_latest_message_event(events: list[Any]) -> Any | None:
    for event in reversed(events):
        if str(getattr(event, "event_type", "")).strip() == "message":
            return event
    return None


def _is_build_rc_workflow(workflow_file: str) -> bool:
    return str(workflow_file or "").strip().lower().endswith("build_rc.yml")


def _with_approval_suffix(*, base_text: str, suffix: str) -> str:
    normalized_base = str(base_text or "").strip()
    normalized_suffix = str(suffix or "").strip()
    if not normalized_suffix:
        return normalized_base
    if not normalized_base:
        return normalized_suffix
    if normalized_base.endswith(normalized_suffix):
        return normalized_base
    return f"{normalized_base}\n\n{normalized_suffix}"


def _extract_latest_message_thread_ts(events: list[Any]) -> str:
    for event in reversed(events):
        if str(getattr(event, "event_type", "")).strip() != "message":
            continue
        thread_ts = str(getattr(event, "thread_ts", "") or "").strip()
        if thread_ts:
            return thread_ts
        message_ts = str(getattr(event, "message_ts", "") or "").strip()
        if message_ts:
            return message_ts
    return ""


def _readiness_jql_query(release_version: str) -> str:
    return f"fixVersion = {release_version} ORDER BY project ASC, type DESC, key ASC"


def _readiness_issues_url(release_version: str) -> str:
    encoded = quote(_readiness_jql_query(release_version), safe="")
    return f"https://instories.atlassian.net/issues/?jql={encoded}"


def _format_slack_owner_mention(owner: str) -> str:
    normalized = str(owner or "").strip()
    if not normalized:
        return normalized
    if normalized.startswith("<!subteam^") and normalized.endswith(">"):
        return normalized
    if normalized.startswith("<#") and normalized.endswith(">"):
        return normalized
    if normalized.startswith("<@") and normalized.endswith(">"):
        entity_id = normalized[2:-1].strip()
        if re.fullmatch(r"[CDG][A-Z0-9]+", entity_id):
            return f"<#{entity_id}>"
        return normalized
    if re.fullmatch(r"[UW][A-Z0-9]+", normalized):
        return f"<@{normalized}>"
    if re.fullmatch(r"S[A-Z0-9]+", normalized):
        return f"<!subteam^{normalized}>"
    if re.fullmatch(r"[CDG][A-Z0-9]+", normalized):
        return f"<#{normalized}>"
    if normalized.startswith("@"):
        mention_id = normalized[1:].strip()
        if re.fullmatch(r"[UW][A-Z0-9]+", mention_id):
            return f"<@{mention_id}>"
        return normalized
    return f"@{normalized}"


def _normalize_readiness_message_text(
    *,
    text: str,
    release_version: str,
    readiness_owners: dict[str, str],
) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return f"Релиз <{_readiness_issues_url(release_version)}|{release_version}>"

    lines = normalized.splitlines()
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        lines[idx] = f"Релиз <{_readiness_issues_url(release_version)}|{release_version}>"
        break

    status_idx = next(
        (idx for idx, line in enumerate(lines) if line.strip() == "Статус готовности к срезу:"),
        None,
    )
    prompt_idx = next(
        (
            idx
            for idx, line in enumerate(lines)
            if line.strip().startswith("Напишите в треде по готовности своей части.")
        ),
        None,
    )
    if status_idx is not None and prompt_idx is not None and prompt_idx > status_idx:
        readiness_lines = [
            f":hourglass_flowing_sand: {team} {_format_slack_owner_mention(owner)}"
            for team, owner in readiness_owners.items()
        ]
        lines = [
            *lines[: status_idx + 1],
            *readiness_lines,
            "",
            *lines[prompt_idx:],
        ]

    normalized_message = "\n".join(lines)
    normalized_message = re.sub(
        r"\*{0,2}Важное напоминание\*{0,2}",
        "*Важное напоминание*",
        normalized_message,
        count=1,
    )
    return normalized_message


def _build_readiness_checklist_text(
    *,
    release_version: str,
    readiness_owners: dict[str, str],
    readiness_map: dict[str, bool],
) -> str:
    readiness_lines = []
    for team, owner in readiness_owners.items():
        is_ready = bool(readiness_map.get(team, False))
        emoji = ":white_check_mark:" if is_ready else ":hourglass_flowing_sand:"
        readiness_lines.append(f"{emoji} {team} {_format_slack_owner_mention(owner)}")
    return (
        f"Релиз <{_readiness_issues_url(release_version)}|{release_version}>\n\n"
        "Статус готовности к срезу:\n"
        + "\n".join(readiness_lines)
        + "\n\n"
        "Напишите в треде по готовности своей части.\n\n"
        "*Важное напоминание* – все задачи, не влитые в ветку RC до 15:00 МСК "
        "едут в релиз только после одобрения QA"
    )


def _is_readiness_checklist_message(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return False
    return "Статус готовности к срезу:" in normalized


def _must_be_channel_level_message(*, text: str, next_step: ReleaseStep) -> bool:
    return next_step == ReleaseStep.WAIT_READINESS_CONFIRMATIONS


def _is_final_release_congratulation(
    *,
    previous_step: ReleaseStep,
    next_step: ReleaseStep,
    events: list[Any],
) -> bool:
    if previous_step != ReleaseStep.WAIT_RC_READINESS:
        return False
    if next_step != ReleaseStep.IDLE:
        return False
    latest_message = _extract_latest_message_event(events)
    if latest_message is None:
        return False
    return _is_release_ready_message(getattr(latest_message, "text", ""))


def _is_release_ready_message(text: str) -> bool:
    return str(text or "").strip() == "Релиз готов к отправке"


def _readiness_item_from_audit_reason(audit_reason: str) -> str:
    match = re.search(r"readiness_confirmation:([^\s;]+)", str(audit_reason or ""))
    if not match:
        return ""
    return match.group(1).strip()


def _flow_event_from_lifecycle(lifecycle: str) -> str:
    normalized = str(lifecycle or "").strip().lower()
    if normalized == "paused":
        return "flow_paused"
    if normalized == "completed":
        return "flow_completed"
    if normalized == "running":
        return "flow_resumed"
    return "flow_transition"


def _audit_field(audit_reason: str, key: str) -> str | None:
    target = f"{key}:"
    for part in str(audit_reason or "").split(";"):
        normalized = part.strip()
        if not normalized.startswith(target):
            continue
        return normalized.split(":", 1)[-1].strip() or None
    return None


