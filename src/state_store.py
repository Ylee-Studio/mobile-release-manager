"""Persistence for workflow state and pending gates."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .workflow_state import WorkflowState


class WorkflowStateStore:
    """Simple JSONL-backed state store."""

    def __init__(self, *, state_path: str):
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> WorkflowState:
        if not self.state_path.exists():
            return WorkflowState()
        try:
            lines = self.state_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return WorkflowState()
        for line in reversed(lines):
            payload = self._parse_line(line)
            if payload is None:
                continue
            state_raw = payload.get("state")
            if isinstance(state_raw, dict):
                return WorkflowState.from_dict(state_raw)
        return WorkflowState()

    def save(self, *, state: WorkflowState, reason: str) -> None:
        payload: dict[str, Any] = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "state": state.to_dict(),
        }
        with self.state_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    @staticmethod
    def _parse_line(line: str) -> dict[str, Any] | None:
        text = line.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

