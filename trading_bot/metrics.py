"""Lightweight, thread-safe runtime telemetry.

Tracks two live rates for the GUI:
  * **Alpaca API calls per second** -- every REST request the SDK makes (each
    Alpaca client's ``requests.Session`` is hooked, so pagination and retries
    are counted accurately).
  * **LLM tokens per minute** -- token usage reported by the LLM/Copilot
    endpoint on each Orchestrator query.

A single process-wide :data:`METRICS` instance is shared by the Alpaca client,
the LLM client, and the GUI. Events are kept in short rolling windows (pruned
on access) so memory stays bounded regardless of runtime.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

# Keep at most this many seconds of event history (covers the 60s token window
# plus headroom for the calls/sec averaging window).
_RETENTION_SECONDS = 120.0


@dataclass
class MetricsSnapshot:
    """An immutable read of the current telemetry, for the GUI."""

    api_calls_per_second: float
    api_calls_total: int
    tokens_per_minute: int
    tokens_total: int


class RuntimeMetrics:
    """Thread-safe rolling counters for API calls and LLM token usage."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._api_calls: Deque[float] = deque()        # monotonic timestamps
        self._tokens: Deque[Tuple[float, int]] = deque()  # (timestamp, count)
        self._api_total = 0
        self._tokens_total = 0

    # ---- Recording (called from worker / network threads) ---------------

    def record_api_call(self, n: int = 1) -> None:
        if n <= 0:
            return
        now = time.monotonic()
        with self._lock:
            for _ in range(n):
                self._api_calls.append(now)
            self._api_total += n
            self._prune(now)

    def record_tokens(self, count: int) -> None:
        if count <= 0:
            return
        now = time.monotonic()
        with self._lock:
            self._tokens.append((now, int(count)))
            self._tokens_total += int(count)
            self._prune(now)

    # ---- Reading (called from the GUI main loop) ------------------------

    def api_calls_per_second(self, window: float = 5.0) -> float:
        """Average calls/second over the trailing ``window`` seconds."""
        window = max(1.0, float(window))
        now = time.monotonic()
        cutoff = now - window
        with self._lock:
            self._prune(now)
            n = sum(1 for t in self._api_calls if t >= cutoff)
        return n / window

    def tokens_per_minute(self) -> int:
        """LLM tokens consumed in the trailing 60 seconds."""
        now = time.monotonic()
        cutoff = now - 60.0
        with self._lock:
            self._prune(now)
            return int(sum(c for t, c in self._tokens if t >= cutoff))

    def snapshot(self, calls_window: float = 5.0) -> MetricsSnapshot:
        """Return all four figures in one consistent read."""
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            cw = max(1.0, float(calls_window))
            calls_cut = now - cw
            tok_cut = now - 60.0
            cps = sum(1 for t in self._api_calls if t >= calls_cut) / cw
            tpm = int(sum(c for t, c in self._tokens if t >= tok_cut))
            return MetricsSnapshot(
                api_calls_per_second=cps,
                api_calls_total=self._api_total,
                tokens_per_minute=tpm,
                tokens_total=self._tokens_total,
            )

    def reset(self) -> None:
        with self._lock:
            self._api_calls.clear()
            self._tokens.clear()
            self._api_total = 0
            self._tokens_total = 0

    # ---- Internal --------------------------------------------------------

    def _prune(self, now: float) -> None:
        cutoff = now - _RETENTION_SECONDS
        calls = self._api_calls
        while calls and calls[0] < cutoff:
            calls.popleft()
        toks = self._tokens
        while toks and toks[0][0] < cutoff:
            toks.popleft()


# Process-wide shared instance.
METRICS = RuntimeMetrics()
