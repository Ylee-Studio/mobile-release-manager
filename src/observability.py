"""Crew runtime observability callbacks."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class CrewRuntimeCallbacks:
    """Best-effort Crew callback adapter with stable logging."""

    logger_name: str = "crew_runtime.callbacks"

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.logger_name)

    def on_step(self, *args: Any, **kwargs: Any) -> None:
        self.logger.info(
            "crew_step at=%s args=%s kwargs=%s",
            datetime.now(timezone.utc).isoformat(),
            len(args),
            sorted(kwargs.keys()),
        )

    def on_task(self, *args: Any, **kwargs: Any) -> None:
        self.logger.info(
            "crew_task at=%s args=%s kwargs=%s",
            datetime.now(timezone.utc).isoformat(),
            len(args),
            sorted(kwargs.keys()),
        )
