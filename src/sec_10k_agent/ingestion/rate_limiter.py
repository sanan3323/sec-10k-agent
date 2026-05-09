"""Rate limiter for outbound HTTP.

SEC EDGAR's hard limit is 10 requests/second. We default to 5 to leave
headroom for parallel work and to be a polite citizen. The limiter is
synchronous and process-local — that's fine for our single-process batch
ingestion. If we ever go async or multi-process, this becomes the single
chokepoint to replace.
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """Spaces out calls so no more than `rate_per_sec` happen per second.

    Implementation: simple per-call sleep based on the timestamp of the
    previous call. Good enough for low single-digit RPS and one process.
    NOT a real token bucket — bursts above the rate are not allowed at all.
    That's a feature here: SEC penalizes bursts, not steady traffic.

    Thread-safe via a lock. Tests rely on this.
    """

    def __init__(self, rate_per_sec: float) -> None:
        if rate_per_sec <= 0:
            raise ValueError(f"rate_per_sec must be > 0, got {rate_per_sec}")
        self._min_interval = 1.0 / rate_per_sec
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until the next call is allowed, then mark it as taken."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()
