from __future__ import annotations

import signal
from pathlib import Path
from types import SimpleNamespace

from src import main as main_module
from src.workflow_state import ReleaseStep


class _FakeWorkflow:
    def __init__(self, *, on_first_tick, pending_count_fn):
        self._on_first_tick = on_first_tick
        self._pending_count_fn = pending_count_fn
        self.tick_reasons: list[str] = []
        self.slack_gateway = self

    def tick(self, *, trigger_reason: str = "heartbeat_timer", now=None):  # noqa: ANN001
        self.tick_reasons.append(trigger_reason)
        if len(self.tick_reasons) == 1:
            self._on_first_tick()
        return SimpleNamespace(step=ReleaseStep.IDLE)

    def pending_events_count(self) -> int:
        return int(self._pending_count_fn())


def test_signal_arriving_during_busy_tick_starts_next_tick_immediately(tmp_path: Path, monkeypatch) -> None:
    signal_handler: dict[str, object] = {}
    sleep_calls: list[float] = []

    def _fake_signal(_sig, handler):  # noqa: ANN001
        signal_handler["handler"] = handler

    def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    workflow = _FakeWorkflow(
        on_first_tick=lambda: signal_handler["handler"](signal.SIGUSR1, None),
        pending_count_fn=lambda: 0,
    )
    pid_path = tmp_path / "agent.pid"

    monkeypatch.setattr(main_module.signal, "signal", _fake_signal)
    monkeypatch.setattr(main_module.time, "sleep", _fake_sleep)
    monkeypatch.setattr(
        main_module,
        "_build_workflow",
        lambda: (workflow, 240, 240, pid_path),
    )

    main_module.run_heartbeat(iterations=2)

    assert workflow.tick_reasons == ["heartbeat_timer", "signal_trigger"]
    assert sleep_calls == []
    assert not pid_path.exists()


def test_signal_trigger_runs_bounded_drain_ticks(tmp_path: Path, monkeypatch) -> None:
    signal_handler: dict[str, object] = {}
    sleep_calls: list[float] = []

    def _fake_signal(_sig, handler):  # noqa: ANN001
        signal_handler["handler"] = handler

    def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    workflow = _FakeWorkflow(
        on_first_tick=lambda: signal_handler["handler"](signal.SIGUSR1, None),
        pending_count_fn=lambda: (
            0
            if not any(reason in {"signal_trigger", "signal_drain"} for reason in workflow.tick_reasons)
            else 10
        ),
    )
    pid_path = tmp_path / "agent.pid"

    monkeypatch.setattr(main_module.signal, "signal", _fake_signal)
    monkeypatch.setattr(main_module.time, "sleep", _fake_sleep)
    monkeypatch.setattr(
        main_module,
        "_build_workflow",
        lambda: (workflow, 240, 240, pid_path),
    )

    main_module.run_heartbeat(iterations=5)

    assert workflow.tick_reasons == [
        "heartbeat_timer",
        "signal_trigger",
        "signal_drain",
        "signal_drain",
        "signal_drain",
    ]
    assert sleep_calls == []
    assert not pid_path.exists()
