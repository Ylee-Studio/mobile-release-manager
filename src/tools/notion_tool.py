"""Notion stub tool for release checklist retrieval."""
from __future__ import annotations

import json
from typing import Any, Dict

from pydantic import BaseModel, Field


class NotionInput(BaseModel):
    action: str = Field(
        default="fetch_checklist",
        description="Action to perform: fetch_checklist | fetch_page",
    )
    page_id: str | None = Field(
        default=None,
        description="Optional Notion page identifier (human shorthand is fine).",
    )


class NotionReleaseTool(BaseModel):
    """Lightweight proxy that pretends to query a Notion release process page."""

    name: str = "notion_release_tool"
    description: str = "Reads the release checklist steps from Notion and returns structured items."
    release_doc_url: str
    database_id: str

    def run(self, action: str = "fetch_checklist", page_id: str | None = None) -> Dict[str, Any]:
        if action == "fetch_checklist":
            return self._mock_checklist()
        if action == "fetch_page":
            return {
                "page_id": page_id or self.database_id,
                "url": self.release_doc_url,
                "content": "Placeholder page body pulled from Notion API.",
            }
        raise ValueError(f"Unsupported Notion action: {action}")

    # ------------------------------------------------------------------ helpers
    def _mock_checklist(self) -> Dict[str, Any]:
        steps = [
            {
                "id": "prep-release-notes",
                "title": "Finalize release notes",
                "owner": "PM",
                "status": "in_progress",
                "due": "2026-02-24",
            },
            {
                "id": "sync-teams",
                "title": "Confirm team readiness",
                "owner": "Release Coordinator",
                "status": "blocked",
                "blockers": [
                    "iOS QA awaiting fix for REG-421",
                ],
            },
            {
                "id": "schedule-announcement",
                "title": "Schedule Slack announcement",
                "owner": "Comms",
                "status": "pending",
            },
        ]
        return {
            "source": self.release_doc_url,
            "database": self.database_id,
            "steps": steps,
            "raw": json.dumps(steps, indent=2),
        }
