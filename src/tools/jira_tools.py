"""Jira tool contracts and local mock adapter."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class JiraGateway:
    """Small local adapter used by workflow and tests."""

    def __init__(self, *, outbox_path: Path):
        self.outbox_path = outbox_path
        self.outbox_path.parent.mkdir(parents=True, exist_ok=True)

    def create_crossspace_release(self, version: str, project_keys: list[str]) -> dict[str, Any]:
        payload = {
            "op": "create_crossspace_release",
            "version": version,
            "project_keys": project_keys,
        }
        with self.outbox_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        return {"version": version, "project_keys": project_keys, "status": "created"}


class JiraCreateCrossspaceInput(BaseModel):
    version: str = Field(..., description="Release version X.Y.Z.")
    project_keys: list[str] = Field(..., description="Projects linked to release.")


class CreateCrossspaceReleaseTool(BaseTool):
    name: str = "create_crossspace_release"
    description: str = "Create cross-space release in Jira Plans."
    args_schema: type[BaseModel] = JiraCreateCrossspaceInput

    gateway: JiraGateway

    def _run(self, version: str, project_keys: list[str]) -> dict[str, Any]:
        return self.gateway.create_crossspace_release(version=version, project_keys=project_keys)
