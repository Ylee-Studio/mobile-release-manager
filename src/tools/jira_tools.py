"""Jira tool contracts and gateway adapter."""
from __future__ import annotations

import base64
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class JiraGateway:
    """Gateway used by workflow and tests."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        email: str | None = None,
        api_token: str | None = None,
        timeout_seconds: float = 15.0,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.email = email or ""
        self.api_token = api_token or ""
        self.timeout_seconds = timeout_seconds
        if not self.base_url or not self.email or not self.api_token:
            raise RuntimeError("Jira credentials are missing for JiraGateway.")

    def create_crossspace_release(self, version: str, project_keys: list[str]) -> dict[str, Any]:
        return self._create_crossspace_release_real(version=version, project_keys=project_keys)

    def _create_crossspace_release_real(self, *, version: str, project_keys: list[str]) -> dict[str, Any]:
        created_project_keys: list[str] = []
        existing_project_keys: list[str] = []

        for project_key in project_keys:
            versions = self._request_json(
                method="GET",
                path=f"/rest/api/3/project/{quote(project_key, safe='')}/versions",
            )
            version_items = versions.get("values", []) if isinstance(versions, dict) else versions
            if not isinstance(version_items, list):
                raise RuntimeError(f"Unexpected Jira versions payload for project '{project_key}'.")
            version_exists = any(str(item.get("name", "")) == version for item in version_items)
            if version_exists:
                existing_project_keys.append(project_key)
                continue

            project = self._request_json(
                method="GET",
                path=f"/rest/api/3/project/{quote(project_key, safe='')}",
            )
            project_id = project.get("id")
            if project_id is None:
                raise RuntimeError(f"Jira project id is missing for project '{project_key}'.")
            try:
                normalized_project_id = int(str(project_id))
            except ValueError as exc:
                raise RuntimeError(
                    f"Jira project id has unexpected format for project '{project_key}': {project_id}"
                ) from exc

            self._request_json(
                method="POST",
                path="/rest/api/3/version",
                payload={"name": version, "projectId": normalized_project_id},
            )
            created_project_keys.append(project_key)

        return {
            "version": version,
            "project_keys": project_keys,
            "status": "created" if created_project_keys else "already_exists",
            "created_project_keys": created_project_keys,
            "existing_project_keys": existing_project_keys,
        }

    def _request_json(self, *, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        if not self.base_url or not self.email or not self.api_token:
            raise RuntimeError("Jira credentials are missing for real API mode.")

        data = json.dumps(payload, ensure_ascii=True).encode("utf-8") if payload is not None else None
        token = f"{self.email}:{self.api_token}".encode("utf-8")
        auth_header = base64.b64encode(token).decode("ascii")
        headers = {
            "Accept": "application/json",
            "Authorization": f"Basic {auth_header}",
        }
        if payload is not None:
            headers["Content-Type"] = "application/json"
        request = Request(url=f"{self.base_url}{path}", data=data, headers=headers, method=method)

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Jira API {method} {path} failed with HTTP {exc.code}: {error_body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Jira API {method} {path} failed: {exc.reason}") from exc

        if not raw_body.strip():
            return {}
        return json.loads(raw_body)


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
