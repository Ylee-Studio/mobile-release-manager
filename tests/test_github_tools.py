from __future__ import annotations

from typing import Any

from src.tools.github_tools import GitHubActionInput, GitHubGateway, GitHubRun


def test_trigger_workflow_retries_with_default_branch_on_missing_ref(monkeypatch) -> None:
    gateway = GitHubGateway(token="test-token", repo="owner/repo")
    post_refs: list[str] = []

    def fake_request_json(  # noqa: ANN202
        *,
        method: str,
        path: str,
        expect_status: tuple[int, ...],  # noqa: ARG001
        payload: dict[str, Any] | None = None,
    ):
        if method == "GET" and path.endswith("/actions/workflows/create_release_branch.yml/runs?per_page=1&exclude_pull_requests=true"):
            return {"workflow_runs": []}
        if method == "GET" and path == "/repos/owner/repo":
            return {"default_branch": "master"}
        if method == "POST" and path == "/repos/owner/repo/actions/workflows/create_release_branch.yml/dispatches":
            ref = str((payload or {}).get("ref") or "")
            post_refs.append(ref)
            if ref == "main":
                raise RuntimeError('GitHub API error 422: {"message":"No ref found for: main"}')
            if ref == "master":
                return {}
        raise AssertionError(f"unexpected request: method={method} path={path}")

    monkeypatch.setattr(gateway, "_request_json", fake_request_json)
    monkeypatch.setattr(
        gateway,
        "_wait_for_new_run",
        lambda **_: GitHubRun(run_id=321, status="queued", conclusion=None),
    )

    run = gateway.trigger_workflow(
        workflow_file="create_release_branch.yml",
        ref="main",
        inputs={"version": "5.105.0"},
    )

    assert run.run_id == 321
    assert post_refs == ["main", "master"]


def test_github_action_input_ref_is_optional() -> None:
    normalized = GitHubActionInput.model_validate(
        {
            "workflow_file": "create_release_branch.yml",
            "inputs": {"version": "5.105.0"},
        }
    ).model_dump(exclude_none=True)

    assert "ref" not in normalized
