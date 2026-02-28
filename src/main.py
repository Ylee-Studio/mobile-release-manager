"""CLI entrypoint for release train orchestrator."""
from __future__ import annotations

import logging
import os
from queue import Queue
from threading import Thread
from pathlib import Path

from dotenv import load_dotenv
import typer

# Load environment variables from .env file
load_dotenv()
from rich.console import Console
import yaml

from .config_loader import CONFIG_PATH, REQUIRED_ENV_VARS, load_runtime_config
from .runtime_engine import RuntimeCoordinator
from .policies import PolicyConfig
from .release_workflow import ReleaseWorkflow
from .state_store import WorkflowStateStore
from .slack_ingress import build_ingress_config, run_slack_ingress
from .tools.github_tools import GitHubGateway
from .tools.slack_tools import SlackGateway

app = typer.Typer(help="Coordinate release train workflow up to WAIT_RC_READINESS.")
console = Console()


def _build_workflow() -> ReleaseWorkflow:
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        raise RuntimeError("Missing required environment variables: " + ", ".join(sorted(missing)))
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config.yaml at {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    config = raw_config if isinstance(raw_config, dict) else {}
    runtime = load_runtime_config(config)
    policy = PolicyConfig.from_dict(config.get("policies", {}))
    slack_gateway = SlackGateway(
        bot_token=runtime.slack_bot_token,
        events_path=Path(runtime.slack_events_path),
    )
    state_store = WorkflowStateStore(state_path=runtime.memory_db_path)
    runtime_engine = RuntimeCoordinator(
        policy=policy,
        slack_gateway=slack_gateway,
        memory_db_path=runtime.memory_db_path,
    )
    github_gateway = GitHubGateway(
        token=runtime.github_token,
        repo=runtime.github_repo,
    )
    workflow = ReleaseWorkflow(
        config=runtime,
        slack_gateway=slack_gateway,
        state_store=state_store,
        runtime_engine=runtime_engine,
        github_gateway=github_gateway,
    )
    return workflow


@app.command()
def tick() -> None:
    """Execute one workflow tick."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    workflow = _build_workflow()
    state = workflow.tick(trigger_reason="manual_tick")
    console.print(f"[green]Tick complete.[/] Current step: {state.step.value}")


@app.command("run-slack-webhook")
def run_slack_webhook(
    host: str = typer.Option("0.0.0.0", help="Host for webhook HTTP server."),
    port: int = typer.Option(8080, help="Port for webhook HTTP server."),
) -> None:
    """Run minimal Slack webhook ingress server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    workflow = _build_workflow()
    logger = logging.getLogger("slack_ingress")
    event_queue: Queue[object] = Queue()
    worker_stop_sentinel = object()

    def _process_event_worker() -> None:
        while True:
            queue_item = event_queue.get()
            try:
                if queue_item is worker_stop_sentinel:
                    return
                logger.info("start processing webhook event")
                state = workflow.tick(trigger_reason="webhook_event")
                logger.info("finished processing webhook event step=%s", state.step.value)
            except Exception as exc:  # pragma: no cover - runtime safety
                logger.exception("webhook-triggered tick failed: %s", exc)
            finally:
                event_queue.task_done()

    worker = Thread(target=_process_event_worker, name="slack-webhook-worker", daemon=True)
    worker.start()

    def _enqueue_event_for_processing() -> None:
        event_queue.put_nowait(None)

    def _get_workflow_state():
        return workflow.state_store.load()

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    config = raw_config if isinstance(raw_config, dict) else {}
    cfg = build_ingress_config(config)
    console.print(
        f"[cyan]Slack ingress listening on http://{host}:{port} "
        f"(events file: {cfg.events_path}; mode: async workflow tick)[/]"
    )
    try:
        run_slack_ingress(
            host=host,
            port=port,
            cfg=cfg,
            on_event_persisted=_enqueue_event_for_processing,
            get_workflow_state=_get_workflow_state,
        )
    finally:
        event_queue.put_nowait(worker_stop_sentinel)
        event_queue.join()
        worker.join(timeout=1)


if __name__ == "__main__":
    app()
