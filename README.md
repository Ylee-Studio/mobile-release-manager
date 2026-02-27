# Release Manager Runtime

Pydantic + deterministic state machine release workflow runner with event-driven webhook processing, Slack event ingestion, and persisted workflow state.

## Capabilities
- Deterministic gates via state machine before runtime decisions.
- Persistent state snapshots via local JSONL state store.
- Event-driven processing: each persisted Slack event immediately triggers one workflow turn.
- Audit trail in `artifacts/workflow_audit.jsonl`.

## Runtime Commands
- One tick: `python -m src.main tick`
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
- `workflow.storage` for memory/audit/events paths
- `workflow.readiness_owners` for readiness map
- `slack` for channel/token resolution
- `jira.project_keys` for JQL messaging context

## Development
- Run targeted tests:
  - `python -m pytest tests/test_event_driven_idempotency.py`
  - `python -m pytest tests/test_runtime_sessions.py`
  - `python -m pytest tests/test_flow_runtime_pipeline.py`
- Full test run:
  - `python -m pytest`