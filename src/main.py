"""CLI entrypoint for release train orchestrator."""
from __future__ import annotations

import logging
import os
import signal
import time
from pathlib import Path

import typer
from rich.console import Console

from .config_loader import ensure_env_vars, load_config, load_runtime_config
from .crew_memory import SQLiteMemory
from .crew_runtime import CrewRuntimeCoordinator
from .policies import PolicyConfig
from .release_workflow import ReleaseWorkflow
from .slack_ingress import build_ingress_config, run_slack_ingress
from .tools.slack_tools import SlackGateway
from .workflow_state import ReleaseStep

app = typer.Typer(help="Coordinate release train workflow up to READY_FOR_BRANCH_CUT.")
console = Console()


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _build_workflow() -> tuple[ReleaseWorkflow, int, int, Path]:
    ensure_env_vars()
    config = load_config()
    runtime = load_runtime_config(config)
    policy = PolicyConfig.from_dict(config.get("policies", {}))
    slack_gateway = SlackGateway(
        bot_token=runtime.slack_bot_token,
        events_path=Path(runtime.slack_events_path),
    )
    memory = SQLiteMemory(db_path=runtime.memory_db_path)
    crew_runtime = CrewRuntimeCoordinator(
        policy=policy,
        slack_gateway=slack_gateway,
        memory_db_path=runtime.memory_db_path,
    )
    workflow = ReleaseWorkflow(
        config=runtime,
        memory=memory,
        slack_gateway=slack_gateway,
        crew_runtime=crew_runtime,
    )
    return (
        workflow,
        runtime.heartbeat_active_minutes,
        runtime.heartbeat_idle_minutes,
        Path(runtime.agent_pid_path),
    )


@app.command()
def tick() -> None:
    """Execute one heartbeat tick."""
    _configure_logging()
    workflow, _active, _idle, _pid_path = _build_workflow()
    state = workflow.tick()
    console.print(f"[green]Tick complete.[/] Current step: {state.step.value}")


@app.command("run-heartbeat")
def run_heartbeat(iterations: int = typer.Option(0, help="0 means infinite loop.")) -> None:
    """Run the heartbeat loop."""
    _configure_logging()
    workflow, active_minutes, idle_minutes, agent_pid_path = _build_workflow()
    trigger_event = False

    def _signal_handler(_signum, _frame) -> None:
        nonlocal trigger_event
        trigger_event = True

    signal.signal(signal.SIGUSR1, _signal_handler)
    agent_pid_path.parent.mkdir(parents=True, exist_ok=True)
    agent_pid_path.write_text(str(os.getpid()), encoding="utf-8")
    logger = logging.getLogger("release_agent")
    logger.info("agent pid registered: %s", agent_pid_path)

    run_count = 0
    try:
        while True:
            state = workflow.tick()
            run_count += 1
            console.print(f"[cyan]Heartbeat tick #{run_count} step={state.step.value}")

            if iterations and run_count >= iterations:
                break

            delay_minutes = idle_minutes if state.step == ReleaseStep.IDLE else active_minutes
            delay_seconds = delay_minutes * 60
            slept = 0.0
            while slept < delay_seconds:
                if trigger_event:
                    trigger_event = False
                    logger.info("received immediate trigger via SIGUSR1")
                    break
                chunk = min(1.0, delay_seconds - slept)
                time.sleep(chunk)
                slept += chunk
    finally:
        if agent_pid_path.exists():
            agent_pid_path.unlink()


@app.command("run-slack-webhook")
def run_slack_webhook(
    host: str = typer.Option("0.0.0.0", help="Host for webhook HTTP server."),
    port: int = typer.Option(8080, help="Port for webhook HTTP server."),
) -> None:
    """Run minimal Slack webhook ingress server."""
    _configure_logging()
    config = load_config()
    runtime = load_runtime_config(config)
    agent_pid_path = Path(runtime.agent_pid_path)
    logger = logging.getLogger("slack_ingress")

    def _deliver_event_to_agent() -> None:
        if not agent_pid_path.exists():
            logger.warning("agent pid file not found: %s", agent_pid_path)
            return
        raw_pid = agent_pid_path.read_text(encoding="utf-8").strip()
        if not raw_pid:
            logger.warning("agent pid file is empty: %s", agent_pid_path)
            return
        try:
            os.kill(int(raw_pid), signal.SIGUSR1)
            logger.info("delivered event trigger to agent pid=%s", raw_pid)
        except (ValueError, ProcessLookupError):
            logger.warning("failed to deliver event, stale pid=%s", raw_pid)
        except PermissionError:
            logger.warning("failed to deliver event, no permission for pid=%s", raw_pid)

    cfg = build_ingress_config(config)
    console.print(
        f"[cyan]Slack ingress listening on http://{host}:{port} "
        f"(events file: {cfg.events_path}; mode: signal crew process)[/]"
    )
    run_slack_ingress(
        host=host,
        port=port,
        cfg=cfg,
        on_event_persisted=_deliver_event_to_agent,
    )


if __name__ == "__main__":
    app()
