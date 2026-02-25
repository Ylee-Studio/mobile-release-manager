"""Persistent storage for workflow state."""
from __future__ import annotations

import json
from pathlib import Path

from .workflow_state import WorkflowState


class StateStore:
    """JSON-backed state storage with atomic writes."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> WorkflowState:
        if not self.path.exists():
            return WorkflowState()
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return WorkflowState.from_dict(raw)

    def save(self, state: WorkflowState) -> None:
        payload = json.dumps(state.to_dict(), ensure_ascii=True, indent=2, sort_keys=True)
        temp_path = self.path.with_suffix(".tmp")
        temp_path.write_text(payload, encoding="utf-8")
        temp_path.replace(self.path)
