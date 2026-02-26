"""CrewAI Memory adapter for workflow state persistence."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional compatibility path
    from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
except Exception:  # pragma: no cover - tests can run without crewai installed
    LTMSQLiteStorage = None

from .workflow_state import WorkflowState, utc_now_iso


@dataclass
class CrewAIMemory:
    """Persist workflow snapshots via CrewAI Memory API."""

    db_path: str
    state_key: str = "release_workflow_state"
    bootstrap_key: str = "__bootstrap__"

    def __post_init__(self) -> None:
        self._file_path = Path(f"{self.db_path}.json")
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage = LTMSQLiteStorage(db_path=self.db_path, verbose=False) if LTMSQLiteStorage else None

    def load_state(self) -> WorkflowState:
        row = self._load_latest_snapshot()
        if row is None:
            legacy_state = self._load_legacy_state()
            if legacy_state is None:
                return WorkflowState()
            self.save_state(legacy_state, reason="legacy_migration")
            return legacy_state
        try:
            raw_state = json.loads(str(row.get("state_json", "{}")))
        except json.JSONDecodeError:
            return WorkflowState()
        return WorkflowState.from_dict(raw_state if isinstance(raw_state, dict) else {})

    def save_state(self, state: WorkflowState, *, reason: str) -> None:
        state_json = json.dumps(state.to_dict(), ensure_ascii=True)
        saved_at = utc_now_iso()
        current_key = self._snapshot_key_for_state(state)
        retained_keys = self._retained_snapshot_keys(state=state, current_key=current_key)
        if self._save_with_crewai(
            snapshot_key=current_key,
            state_json=state_json,
            reason=reason,
            saved_at=saved_at,
            retained_keys=retained_keys,
        ):
            return
        self._save_to_file(
            snapshot_key=current_key,
            state_json=state_json,
            reason=reason,
            saved_at=saved_at,
            retained_keys=retained_keys,
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

    def snapshot_keys(self) -> list[str]:
        """Return stored snapshot keys (debug/testing helper)."""
        snapshots = self._load_all_snapshots()
        return sorted(snapshots.keys())

    def snapshot_state(self, snapshot_key: str) -> dict[str, Any] | None:
        """Return raw state payload for snapshot key (debug/testing helper)."""
        snapshots = self._load_all_snapshots()
        snapshot = snapshots.get(snapshot_key)
        if not isinstance(snapshot, dict):
            return None
        try:
            parsed = json.loads(str(snapshot.get("state_json", "{}")))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _load_latest_snapshot(self) -> dict[str, Any] | None:
        snapshots = self._load_all_snapshots()
        if not snapshots:
            return None
        ordered = sorted(
            snapshots.values(),
            key=lambda item: str(item.get("saved_at", "")),
            reverse=True,
        )
        latest = ordered[0]
        return latest if isinstance(latest, dict) else None

    def _load_all_snapshots(self) -> dict[str, dict[str, Any]]:
        if self._storage is not None:
            snapshots = self._load_snapshots_from_crewai()
            if snapshots:
                return snapshots
        return self._load_snapshots_from_file()

    def _load_snapshots_from_crewai(self) -> dict[str, dict[str, Any]]:
        if self._storage is None:
            return {}
        loader = getattr(self._storage, "load", None)
        if not callable(loader):
            return {}
        try:
            entries = loader(task_description=self.state_key, latest_n=200) or []
        except TypeError:
            entries = loader(self.state_key, 200) or []
        except Exception:
            return {}
        snapshots: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            metadata = entry.get("metadata", {})
            if not isinstance(metadata, dict):
                continue
            snapshot_key = str(metadata.get("snapshot_key", "")).strip() or self.bootstrap_key
            raw = metadata.get("state")
            state_json = json.dumps(raw if isinstance(raw, dict) else {}, ensure_ascii=True)
            snapshots[snapshot_key] = {
                "snapshot_key": snapshot_key,
                "state_json": state_json,
                "reason": str(metadata.get("reason", "crewai_memory")),
                "saved_at": str(metadata.get("saved_at", utc_now_iso())),
            }
        return snapshots

    def _save_with_crewai(
        self,
        *,
        snapshot_key: str,
        state_json: str,
        reason: str,
        saved_at: str,
        retained_keys: tuple[str, ...],
    ) -> bool:
        if self._storage is None:
            return False
        saver = getattr(self._storage, "save", None)
        if not callable(saver):
            return False
        existing = self._load_snapshots_from_crewai()
        retained = {key: value for key, value in existing.items() if key in retained_keys}
        retained[snapshot_key] = {
            "snapshot_key": snapshot_key,
            "state_json": state_json,
            "reason": reason,
            "saved_at": saved_at,
        }
        for entry in retained.values():
            try:
                raw_state = json.loads(entry["state_json"])
            except json.JSONDecodeError:
                raw_state = {}
            metadata = {
                "snapshot_key": entry["snapshot_key"],
                "state": raw_state if isinstance(raw_state, dict) else {},
                "reason": entry["reason"],
                "saved_at": entry["saved_at"],
            }
            if not self._invoke_crewai_save(saver=saver, metadata=metadata):
                return False
        return True

    def _invoke_crewai_save(self, *, saver: Any, metadata: dict[str, Any]) -> bool:
        call_variants = (
            {"task_description": self.state_key, "metadata": metadata},
            {"task_description": self.state_key, "metadata": metadata, "agent": "release_workflow"},
            {"task_description": self.state_key, "metadata": metadata, "score": 1.0},
        )
        for kwargs in call_variants:
            try:
                saver(**kwargs)
                return True
            except TypeError:
                continue
            except Exception:
                return False
        try:
            saver(self.state_key, metadata)
            return True
        except Exception:
            return False

    def _save_to_file(
        self,
        *,
        snapshot_key: str,
        state_json: str,
        reason: str,
        saved_at: str,
        retained_keys: tuple[str, ...],
    ) -> None:
        data = self._load_file_store()
        snapshots = data.setdefault("snapshots", {})
        if not isinstance(snapshots, dict):
            snapshots = {}
            data["snapshots"] = snapshots
        snapshots = {
            key: value for key, value in snapshots.items() if key in retained_keys and isinstance(value, dict)
        }
        snapshots[snapshot_key] = {
            "snapshot_key": snapshot_key,
            "state_json": state_json,
            "reason": reason,
            "saved_at": saved_at,
        }
        data["snapshots"] = snapshots
        self._write_file_store(data)

    def _load_snapshots_from_file(self) -> dict[str, dict[str, Any]]:
        data = self._load_file_store()
        snapshots = data.get("snapshots", {})
        if not isinstance(snapshots, dict):
            return {}
        out: dict[str, dict[str, Any]] = {}
        for key, value in snapshots.items():
            if isinstance(value, dict):
                out[str(key)] = value
        return out

    def _load_file_store(self) -> dict[str, Any]:
        if not self._file_path.exists():
            return {}
        try:
            raw = json.loads(self._file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def _write_file_store(self, payload: dict[str, Any]) -> None:
        self._file_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def _load_legacy_state(self) -> WorkflowState | None:
        if self._storage is None:
            return None
        loader = getattr(self._storage, "load", None)
        if not callable(loader):
            return None
        try:
            entries = loader(task_description=self.state_key, latest_n=1) or []
        except TypeError:
            entries = loader(self.state_key, 1) or []
        except Exception:
            return None
        if not entries:
            return None
        metadata = entries[0].get("metadata", {})
        raw_state = metadata.get("state", {})
        if not isinstance(raw_state, dict):
            return None
        return WorkflowState.from_dict(raw_state)


SQLiteMemory = CrewAIMemory
