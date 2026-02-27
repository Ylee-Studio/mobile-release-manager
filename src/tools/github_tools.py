"""GitHub Actions tool contracts and API gateway."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from pydantic import BaseModel, Field, field_validator


@dataclass
class GitHubRun:
    run_id: int
    status: str
    conclusion: str | None = None


class GitHubGateway:
    """Small GitHub Actions gateway using REST API."""

    def __init__(self, *, token: str, repo: str, timeout_seconds: float = 15.0):
        self.token = str(token or "").strip()
        self.repo = str(repo or "").strip()
        self.timeout_seconds = timeout_seconds
        self.base_url = "https://api.github.com"

    @property
    def is_enabled(self) -> bool:
        return bool(self.token and self.repo)

    def trigger_workflow(
        self,
        *,
        workflow_file: str,
        ref: str,
        inputs: dict[str, Any] | None = None,
    ) -> GitHubRun:
        self._require_configured()
        workflow = str(workflow_file or "").strip()
        if not workflow:
            raise RuntimeError("github workflow_file must be non-empty")
        payload = {
            "ref": str(ref or "").strip() or "main",
            "inputs": inputs or {},
        }
        before_latest_run_id = self._latest_run_id(workflow_file=workflow)
        self._request_json(
            method="POST",
            path=f"/repos/{self.repo}/actions/workflows/{workflow}/dispatches",
            payload=payload,
            expect_status=(204,),
        )
        run = self._wait_for_new_run(workflow_file=workflow, previous_run_id=before_latest_run_id)
        if run is None:
            raise RuntimeError("workflow dispatched but run_id was not discovered")
        return run

    def get_workflow_run(self, *, run_id: int) -> GitHubRun:
        self._require_configured()
        data = self._request_json(
            method="GET",
            path=f"/repos/{self.repo}/actions/runs/{int(run_id)}",
            expect_status=(200,),
        )
        return _run_from_payload(data)

    def _wait_for_new_run(self, *, workflow_file: str, previous_run_id: int | None) -> GitHubRun | None:
        for _ in range(6):
            current = self._latest_run(workflow_file=workflow_file)
            if current is None:
                time.sleep(1)
                continue
            if previous_run_id is None or current.run_id != previous_run_id:
                return current
            time.sleep(1)
        return None

    def _latest_run(self, *, workflow_file: str) -> GitHubRun | None:
        data = self._request_json(
            method="GET",
            path=(
                f"/repos/{self.repo}/actions/workflows/{workflow_file}/runs"
                "?per_page=1&exclude_pull_requests=true"
            ),
            expect_status=(200,),
        )
        runs = data.get("workflow_runs")
        if not isinstance(runs, list) or not runs:
            return None
        first = runs[0]
        if not isinstance(first, dict):
            return None
        return _run_from_payload(first)

    def _latest_run_id(self, *, workflow_file: str) -> int | None:
        latest = self._latest_run(workflow_file=workflow_file)
        if latest is None:
            return None
        return latest.run_id

    def _request_json(
        self,
        *,
        method: str,
        path: str,
        expect_status: tuple[int, ...],
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = self.base_url + path
        body = None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        req = request.Request(url=url, data=body, method=method)
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        if body is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                status = int(getattr(resp, "status", 0))
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API error {exc.code}: {details}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"GitHub API request failed: {exc.reason}") from exc
        if status not in expect_status:
            raise RuntimeError(f"GitHub API unexpected status {status} for {method} {path}")
        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("GitHub API returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("GitHub API returned non-object payload")
        return parsed

    def _require_configured(self) -> None:
        if not self.repo:
            raise RuntimeError("GitHub repo is not configured")
        if not self.token:
            raise RuntimeError("GitHub token is not configured")


def _run_from_payload(payload: dict[str, Any]) -> GitHubRun:
    run_id = int(payload.get("id") or 0)
    if run_id <= 0:
        raise RuntimeError("GitHub workflow run payload missing id")
    status = str(payload.get("status") or "").strip() or "unknown"
    conclusion_raw = payload.get("conclusion")
    conclusion = str(conclusion_raw).strip() if isinstance(conclusion_raw, str) else None
    return GitHubRun(run_id=run_id, status=status, conclusion=conclusion)


class GitHubActionInput(BaseModel):
    workflow_file: str = Field(..., min_length=1, description="GitHub workflow file name.")
    ref: str = Field(default="main", min_length=1, description="Git ref for workflow_dispatch.")
    inputs: dict[str, Any] = Field(default_factory=dict, description="Optional workflow inputs.")

    @field_validator("workflow_file", "ref")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must be non-empty")
        return stripped
