from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.flow_pending_store import FlowPendingStore


def test_put_and_get_pending_by_message_ts(tmp_path: Path) -> None:
    store = FlowPendingStore(path=tmp_path / "pending.json")
    store.put_pending(
        message_ts="177.1",
        thread_ts="177.1",
        flow_id="flow-1",
        approval_kind="WAIT_START_APPROVAL",
        release_version="5.104.0",
    )

    found = store.get_pending(message_ts="177.1", thread_ts=None)

    assert found is not None
    assert found["flow_id"] == "flow-1"
    assert found["approval_kind"] == "WAIT_START_APPROVAL"


def test_mark_resolved_hides_record_from_lookup(tmp_path: Path) -> None:
    store = FlowPendingStore(path=tmp_path / "pending.json")
    store.put_pending(
        message_ts="177.2",
        thread_ts="177.2",
        flow_id="flow-2",
        approval_kind="WAIT_MEETING_CONFIRMATION",
        release_version="5.104.0",
    )
    store.mark_resolved(flow_id="flow-2")

    found = store.get_pending(message_ts="177.2", thread_ts=None)

    assert found is None


def test_cleanup_stale_removes_old_records(tmp_path: Path) -> None:
    store = FlowPendingStore(path=tmp_path / "pending.json")
    store.put_pending(
        message_ts="177.3",
        thread_ts="177.3",
        flow_id="flow-3",
        approval_kind="WAIT_READINESS_CONFIRMATIONS",
        release_version="5.104.0",
    )
    payload = store._load()  # noqa: SLF001 - test helper access
    payload["pending"][0]["created_at"] = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    store._save(payload)  # noqa: SLF001 - test helper access

    removed = store.cleanup_stale(ttl_days=7)

    assert removed == 1
    assert store.get_pending(message_ts="177.3", thread_ts=None) is None
