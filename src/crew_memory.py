"""CrewAI-backed SQLite memory adapter for workflow state."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

try:  # pragma: no cover - optional compatibility path
    from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
except Exception:  # pragma: no cover - tests can run without crewai installed
    LTMSQLiteStorage = None

from .workflow_state import WorkflowState, utc_now_iso


@dataclass
class SQLiteMemory:
    """Persist workflow snapshots in CrewAI long-term SQLite storage."""

    db_path: str
    state_key: str = "release_workflow_state"
    bootstrap_key: str = "__bootstrap__"

    def __post_init__(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._legacy_storage = LTMSQLiteStorage(db_path=self.db_path, verbose=False) if LTMSQLiteStorage else None
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_state_snapshots (
                    snapshot_key TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    saved_at TEXT NOT NULL
                )
                """
            )

    def load_state(self) -> WorkflowState:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT state_json
                FROM workflow_state_snapshots
                ORDER BY saved_at DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            legacy_state = self._load_legacy_state()
            if legacy_state is None:
                return WorkflowState()
            self.save_state(legacy_state, reason="legacy_migration")
            return legacy_state
        try:
            raw_state = json.loads(row["state_json"])
        except json.JSONDecodeError:
            return WorkflowState()
        return WorkflowState.from_dict(raw_state if isinstance(raw_state, dict) else {})

    def save_state(self, state: WorkflowState, *, reason: str) -> None:
        state_json = json.dumps(state.to_dict(), ensure_ascii=True)
        saved_at = utc_now_iso()
        current_key = self._snapshot_key_for_state(state)
        retained_keys = self._retained_snapshot_keys(state=state, current_key=current_key)
        placeholders = ",".join("?" for _ in retained_keys)

        with self._connect() as connection:
            connection.execute("BEGIN")
            connection.execute(
                f"DELETE FROM workflow_state_snapshots WHERE snapshot_key NOT IN ({placeholders})",
                retained_keys,
            )
            connection.execute(
                """
                INSERT INTO workflow_state_snapshots (snapshot_key, state_json, reason, saved_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(snapshot_key) DO UPDATE SET
                    state_json=excluded.state_json,
                    reason=excluded.reason,
                    saved_at=excluded.saved_at
                """,
                (current_key, state_json, reason, saved_at),
            )

    def _snapshot_key_for_state(self, state: WorkflowState) -> str:
        active_release = state.active_release
        if active_release and active_release.release_version:
            return active_release.release_version
        if state.previous_release_version:
            return str(state.previous_release_version)
        return self.bootstrap_key

    def _retained_snapshot_keys(self, *, state: WorkflowState, current_key: str) -> tuple[str, ...]:
        keys: list[str] = [current_key]
        active_release = state.active_release
        if active_release and active_release.release_version:
            keys.append(active_release.release_version)
        if state.previous_release_version:
            keys.append(str(state.previous_release_version))
        seen: set[str] = set()
        unique_keys: list[str] = []
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            unique_keys.append(key)
        return tuple(unique_keys)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _load_legacy_state(self) -> WorkflowState | None:
        if self._legacy_storage is None:
            return None
        entries = self._legacy_storage.load(task_description=self.state_key, latest_n=1) or []
        if not entries:
            return None
        metadata = entries[0].get("metadata", {})
        raw_state = metadata.get("state", {})
        if not isinstance(raw_state, dict):
            return None
        return WorkflowState.from_dict(raw_state)
