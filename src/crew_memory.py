"""CrewAI-backed SQLite memory adapter for workflow state."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage

from .workflow_state import WorkflowState, utc_now_iso


@dataclass
class SQLiteMemory:
    """Persist workflow snapshots in CrewAI long-term SQLite storage."""

    db_path: str
    state_key: str = "release_workflow_state"

    def __post_init__(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._storage = LTMSQLiteStorage(db_path=self.db_path, verbose=False)

    def load_state(self) -> WorkflowState:
        entries = self._storage.load(task_description=self.state_key, latest_n=1) or []
        if not entries:
            return WorkflowState()
        metadata = entries[0].get("metadata", {})
        raw_state = metadata.get("state", {})
        return WorkflowState.from_dict(raw_state if isinstance(raw_state, dict) else {})

    def save_state(self, state: WorkflowState, *, reason: str) -> None:
        self._storage.save(
            task_description=self.state_key,
            metadata={"state": state.to_dict(), "reason": reason},
            datetime=utc_now_iso(),
            score=1.0,
        )
