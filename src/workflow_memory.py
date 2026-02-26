"""Workflow state persistence backed by built-in CrewAI Memory."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional import path in tests
    from crewai import Memory
except Exception:  # pragma: no cover
    Memory = None  # type: ignore[assignment]

from .workflow_state import WorkflowState, utc_now_iso


@dataclass
class CrewAIWorkflowMemory:
    """Persist and restore WorkflowState using CrewAI Memory."""

    storage_path: str
    state_key: str = "release_workflow_state"

    def __post_init__(self) -> None:
        if Memory is None:
            raise RuntimeError("CrewAI Memory is unavailable: install crewai to persist workflow state.")
        root = Path(self.storage_path)
        root.mkdir(parents=True, exist_ok=True)
        self._memory = Memory(storage=str(root))

    def load_state(self) -> WorkflowState:
        entries = self._load_entries()
        for entry in entries:
            raw_state = self._extract_state(entry)
            if isinstance(raw_state, dict):
                return WorkflowState.from_dict(raw_state)
        return WorkflowState()

    def save_state(self, state: WorkflowState, *, reason: str) -> None:
        metadata = {
            "state_key": self.state_key,
            "state": state.to_dict(),
            "reason": reason,
            "saved_at": utc_now_iso(),
        }
        saver = getattr(self._memory, "save", None)
        if not callable(saver):
            return

        call_variants = (
            {"task_description": self.state_key, "metadata": metadata},
            {"task_description": self.state_key, "metadata": metadata, "agent": "release_workflow"},
            {"task_description": self.state_key, "metadata": metadata, "score": 1.0},
        )
        for kwargs in call_variants:
            try:
                saver(**kwargs)
                return
            except TypeError:
                continue
            except Exception:
                return
        try:
            saver(self.state_key, metadata)
        except Exception:
            return

    def _load_entries(self) -> list[dict[str, Any]]:
        loader = getattr(self._memory, "load", None)
        if not callable(loader):
            return []

        entries: Any = []
        call_variants = (
            {"task_description": self.state_key, "latest_n": 100},
            {"task_description": self.state_key},
        )
        for kwargs in call_variants:
            try:
                entries = loader(**kwargs)
                break
            except TypeError:
                continue
            except Exception:
                return []
        else:
            try:
                entries = loader(self.state_key, 100)
            except Exception:
                return []

        if not isinstance(entries, list):
            return []
        normalized = [entry for entry in entries if isinstance(entry, dict)]
        normalized.sort(
            key=lambda entry: str(
                (entry.get("metadata") or {}).get("saved_at")
                if isinstance(entry.get("metadata"), dict)
                else entry.get("saved_at", "")
            ),
            reverse=True,
        )
        return normalized

    @staticmethod
    def _extract_state(entry: dict[str, Any]) -> dict[str, Any] | None:
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
        if isinstance(state, str):
            try:
                parsed = json.loads(state)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, dict):
                return parsed
        return None

