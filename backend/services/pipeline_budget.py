"""Helpers for tracking request budget and per-stage timings."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter


@dataclass(slots=True)
class RequestBudget:
    """Track remaining time for a warm-path request."""

    total_seconds: float
    started_at: float = field(default_factory=perf_counter)
    stage_durations: dict[str, float] = field(default_factory=dict)

    def remaining_seconds(self) -> float:
        return max(0.0, self.total_seconds - (perf_counter() - self.started_at))

    def has_time_for(self, minimum_remaining_seconds: float) -> bool:
        return self.remaining_seconds() >= minimum_remaining_seconds

    def record_stage(self, stage_name: str, stage_started: float) -> float:
        duration = perf_counter() - stage_started
        self.stage_durations[stage_name] = round(duration, 3)
        return duration
