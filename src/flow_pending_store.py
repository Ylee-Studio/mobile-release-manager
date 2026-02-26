"""Persistent mapping between Slack confirmation messages and pending Flow IDs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FlowPendingStore:
    """Small JSON-backed store for message_ts/thread_ts -> flow_id lookup."""

    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def put_pending(
        self,
        *,
        message_ts: str,
        thread_ts: str | None,
        flow_id: str,
        approval_kind: str,
        release_version: str,
    ) -> None:
        msg = message_ts.strip()
        flow = flow_id.strip()
        if not msg or not flow:
            return
        record = {
            "message_ts": msg,
            "thread_ts": (thread_ts or "").strip() or msg,
            "flow_id": flow,
            "approval_kind": approval_kind.strip(),
            "release_version": release_version.strip(),
            "resolved": False,
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        }
        payload = self._load()
        items = payload.setdefault("pending", [])
        if not isinstance(items, list):
            items = []
            payload["pending"] = items
        replaced = False
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            if str(item.get("message_ts", "")).strip() == msg:
                items[idx] = record
                replaced = True
                break
        if not replaced:
            items.append(record)
        self._save(payload)

    def get_pending(self, *, message_ts: str | None, thread_ts: str | None) -> dict[str, Any] | None:
        msg = str(message_ts or "").strip()
        thr = str(thread_ts or "").strip()
        payload = self._load()
        items = payload.get("pending", [])
        if not isinstance(items, list):
            return None
        for item in reversed(items):
            if not isinstance(item, dict):
                continue
            if bool(item.get("resolved", False)):
                continue
            item_msg = str(item.get("message_ts", "")).strip()
            item_thr = str(item.get("thread_ts", "")).strip()
            if msg and item_msg == msg:
                return item
            if thr and item_thr == thr:
                return item
        return None

    def mark_resolved(self, *, flow_id: str | None = None, message_ts: str | None = None) -> None:
        flow = str(flow_id or "").strip()
        msg = str(message_ts or "").strip()
        if not flow and not msg:
            return
        payload = self._load()
        items = payload.get("pending", [])
        if not isinstance(items, list):
            return
        changed = False
        for item in items:
            if not isinstance(item, dict):
                continue
            same_flow = flow and str(item.get("flow_id", "")).strip() == flow
            same_msg = msg and str(item.get("message_ts", "")).strip() == msg
            if same_flow or same_msg:
                item["resolved"] = True
                item["updated_at"] = _utc_now_iso()
                changed = True
        if changed:
            self._save(payload)

    def cleanup_stale(self, *, ttl_days: int = 7) -> int:
        ttl = max(1, int(ttl_days))
        cutoff = datetime.now(timezone.utc) - timedelta(days=ttl)
        payload = self._load()
        items = payload.get("pending", [])
        if not isinstance(items, list):
            return 0
        kept: list[dict[str, Any]] = []
        removed = 0
        for item in items:
            if not isinstance(item, dict):
                removed += 1
                continue
            created_at_raw = str(item.get("created_at", "")).strip()
            try:
                created_at = datetime.fromisoformat(created_at_raw)
            except ValueError:
                created_at = datetime.now(timezone.utc)
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            if created_at < cutoff:
                removed += 1
                continue
            kept.append(item)
        if removed:
            payload["pending"] = kept
            self._save(payload)
        return removed

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"pending": []}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return {"pending": []}
        if not isinstance(raw, dict):
            return {"pending": []}
        return raw

    def _save(self, payload: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
