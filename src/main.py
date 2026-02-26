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
from .crew_memory import CrewAIMemory
from .crew_runtime import CrewRuntimeCoordinator
from .policies import PolicyConfig
from .release_workflow import ReleaseWorkflow
from .slack_ingress import build_ingress_config, run_slack_ingress
from .tools.slack_tools import SlackGateway
from .workflow_state import ReleaseStep

app = typer.Typer(help="Coordinate release train workflow up to READY_FOR_BRANCH_CUT.")
console = Console()
MAX_SIGNAL_DRAIN_TICKS = 3


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
    memory = CrewAIMemory(db_path=runtime.memory_db_path)
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


def _pending_events_count(workflow: ReleaseWorkflow) -> int | None:
    gateway = getattr(workflow, "slack_gateway", None)
    counter = getattr(gateway, "pending_events_count", None)
    if not callable(counter):
        return None
    try:
        return int(counter())
    except Exception:  # pragma: no cover - defensive logging helper
        return None


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
    pending_trigger_count = 0
    last_signal_monotonic: float | None = None

    def _signal_handler(_signum, _frame) -> None:
        nonlocal pending_trigger_count, last_signal_monotonic
        pending_trigger_count += 1
        last_signal_monotonic = time.monotonic()

    signal.signal(signal.SIGUSR1, _signal_handler)
    agent_pid_path.parent.mkdir(parents=True, exist_ok=True)
    agent_pid_path.write_text(str(os.getpid()), encoding="utf-8")
    logger = logging.getLogger("release_agent")
    logger.info("agent pid registered: %s", agent_pid_path)

    run_count = 0
    try:
        while True:
            trigger_reason = "heartbeat_timer"
            signal_lag_seconds: float | None = None
            if pending_trigger_count > 0:
                pending_trigger_count -= 1
                trigger_reason = "signal_trigger"
                if last_signal_monotonic is not None:
                    signal_lag_seconds = max(0.0, time.monotonic() - last_signal_monotonic)
                logger.info(
                    "starting triggered tick pending_after_dequeue=%s signal_lag_seconds=%s",
                    pending_trigger_count,
                    f"{signal_lag_seconds:.3f}" if signal_lag_seconds is not None else "unknown",
                )

            state = workflow.tick(trigger_reason=trigger_reason)
            run_count += 1
            console.print(f"[cyan]Heartbeat tick #{run_count} step={state.step.value}")
            queue_remaining = _pending_events_count(workflow)
            logger.info(
                "tick complete reason=%s run=%s step=%s queue_remaining=%s pending_triggers=%s",
                trigger_reason,
                run_count,
                state.step.value,
                queue_remaining if queue_remaining is not None else "unknown",
                pending_trigger_count,
            )

            if trigger_reason == "signal_trigger":
                drain_runs = 0
                while drain_runs < MAX_SIGNAL_DRAIN_TICKS:
                    queue_remaining = _pending_events_count(workflow)
                    if queue_remaining is None or queue_remaining <= 0:
                        break
                    if iterations and run_count >= iterations:
                        break
                    drain_runs += 1
                    logger.info(
                        "signal drain tick=%s/%s queue_remaining_before=%s",
                        drain_runs,
                        MAX_SIGNAL_DRAIN_TICKS,
                        queue_remaining,
                    )
                    state = workflow.tick(trigger_reason="signal_drain")
                    run_count += 1
                    console.print(f"[cyan]Heartbeat tick #{run_count} step={state.step.value}")
                    queue_after_drain = _pending_events_count(workflow)
                    logger.info(
                        "signal drain complete run=%s step=%s queue_remaining=%s",
                        run_count,
                        state.step.value,
                        queue_after_drain if queue_after_drain is not None else "unknown",
                    )

            if iterations and run_count >= iterations:
                break

            if pending_trigger_count > 0:
                logger.info(
                    "skip heartbeat sleep due pending triggers=%s",
                    pending_trigger_count,
                )
                continue

            delay_minutes = idle_minutes if state.step == ReleaseStep.IDLE else active_minutes
            delay_seconds = delay_minutes * 60
            slept = 0.0
            while slept < delay_seconds:
                if pending_trigger_count > 0:
                    logger.info(
                        "received immediate trigger via SIGUSR1 pending_triggers=%s",
                        pending_trigger_count,
                    )
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
