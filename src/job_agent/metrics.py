"""Metrics helpers for the job application agent."""

from __future__ import annotations

from time import perf_counter


class Stopwatch:
    """Simple stopwatch for request metrics."""

    def __init__(self) -> None:
        """Start the stopwatch."""
        self.start = perf_counter()

    def ms(self) -> float:
        """Return elapsed milliseconds."""
        return round((perf_counter() - self.start) * 1000, 3)
