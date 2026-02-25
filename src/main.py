"""CLI entrypoint for release train orchestrator."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import typer
from rich.console import Console

from .config_loader import ensure_env_vars, load_config, load_runtime_config
from .release_workflow import ReleaseWorkflow
from .state_store import StateStore
from .tools.jira_tools import JiraGateway
from .tools.slack_tools import SlackGateway
from .workflow_state import ReleaseStep

app = typer.Typer(help="Coordinate release train workflow up to READY_FOR_BRANCH_CUT.")
console = Console()


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _build_workflow() -> tuple[ReleaseWorkflow, int, int]:
    ensure_env_vars()
    config = load_config()
    runtime = load_runtime_config(config)
    state_store = StateStore(path=Path(runtime.release_state_path))
    slack_gateway = SlackGateway(
        outbox_path=Path(runtime.slack_outbox_path),
        events_path=Path(runtime.slack_events_path),
    )
    jira_gateway = JiraGateway(outbox_path=Path(runtime.jira_outbox_path))
    workflow = ReleaseWorkflow(
        config=runtime,
        state_store=state_store,
        slack_gateway=slack_gateway,
        jira_gateway=jira_gateway,
    )
    return workflow, runtime.heartbeat_active_minutes, runtime.heartbeat_idle_minutes


@app.command()
def tick() -> None:
    """Execute one heartbeat tick."""
    _configure_logging()
    workflow, _active, _idle = _build_workflow()
    state = workflow.tick()
    console.print(f"[green]Tick complete.[/] Current step: {state.step.value}")


@app.command("run-heartbeat")
def run_heartbeat(iterations: int = typer.Option(0, help="0 means infinite loop.")) -> None:
    """Run the heartbeat loop."""
    _configure_logging()
    workflow, active_minutes, idle_minutes = _build_workflow()

    run_count = 0
    while True:
        state = workflow.tick()
        run_count += 1
        console.print(f"[cyan]Heartbeat tick #{run_count} step={state.step.value}")

        if iterations and run_count >= iterations:
            break

        delay_minutes = idle_minutes if state.step == ReleaseStep.IDLE else active_minutes
        time.sleep(delay_minutes * 60)


if __name__ == "__main__":
    app()
