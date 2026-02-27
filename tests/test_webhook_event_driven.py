from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from src import main as main_module
from src.slack_ingress import SlackIngressConfig


@dataclass
class _FakeWorkflow:
    tick_reasons: list[str]

    def tick(self, *, trigger_reason: str = "event_trigger", now=None):  # noqa: ANN001
        _ = now
        self.tick_reasons.append(trigger_reason)
        return SimpleNamespace(step=SimpleNamespace(value="IDLE"))


def test_run_slack_webhook_processes_event_via_direct_tick(monkeypatch, tmp_path: Path) -> None:
    workflow = _FakeWorkflow(tick_reasons=[])
    cfg = SlackIngressConfig(
        signing_secret="secret",
        announce_channel_id="C_RELEASE",
        events_path=tmp_path / "events.jsonl",
    )

    monkeypatch.setattr(main_module, "_build_workflow", lambda: workflow)
    monkeypatch.setattr(main_module, "build_ingress_config", lambda _: cfg)  # noqa: ARG005

    observed: dict[str, object] = {}

    def _fake_run_slack_ingress(*, host: str, port: int, cfg: SlackIngressConfig, on_event_persisted):  # noqa: ANN001
        observed["host"] = host
        observed["port"] = port
        observed["cfg"] = cfg
        on_event_persisted()
        on_event_persisted()

    monkeypatch.setattr(main_module, "run_slack_ingress", _fake_run_slack_ingress)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("workflow: {}\n", encoding="utf-8")
    monkeypatch.setattr(main_module, "CONFIG_PATH", config_path)

    main_module.run_slack_webhook(host="127.0.0.1", port=8081)

    assert observed["host"] == "127.0.0.1"
    assert observed["port"] == 8081
    assert observed["cfg"] == cfg
    assert workflow.tick_reasons == ["webhook_event", "webhook_event"]
