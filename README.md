# Release Manager Crew

CrewAI-based release workflow runner with heartbeat orchestration, Slack event ingestion, and persisted workflow state.

## Capabilities
- CrewAI Flow runtime (`Flow + Crew + Process.sequential`) for orchestrator/release-manager turns.
- Persistent state snapshots via CrewAI memory adapter (`CrewAIMemory`).
- Heartbeat loop with immediate signal trigger support (`SIGUSR1`) for fast event processing.
- Audit trail in `artifacts/workflow_audit.jsonl`.

## Runtime Commands
- One tick: `python -m src.main tick`
- Continuous heartbeat: `python -m src.main run-heartbeat`
- Slack webhook ingress: `python -m src.main run-slack-webhook`

## Environment
Copy `.env.example` to `.env` and fill required values:
- `OPENAI_API_KEY`
- `SLACK_BOT_TOKEN`
- `SLACK_ANNOUNCE_CHANNEL`

Optional:
- `SLACK_SIGNING_SECRET` (needed for webhook signature validation in ingress mode)

## Configuration
Main runtime config lives in `config.yaml`:
- `workflow.heartbeat` for active/idle intervals
- `workflow.storage` for memory/audit/events paths
- `workflow.readiness_owners` for readiness map
- `slack` for channel/token resolution
- `jira.project_keys` for JQL messaging context

## Development
- Run targeted tests:
  - `python -m pytest tests/test_heartbeat_idempotency.py`
  - `python -m pytest tests/test_crew_runtime_sessions.py`
  - `python -m pytest tests/test_flow_crew_pipeline.py`
- Full test run:
  - `python -m pytest`