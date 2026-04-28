"""In-memory TTL cache for financial data."""

import threading
import time
from typing import Any, Callable, Optional


class DataCache:
    DEFAULT_TTLS = {
        "stock_prices": 300,   # 5 minutes
        "economic": 86400,     # 24 hours
        "news": 3600,          # 1 hour
        "ticker_info": 3600,   # 1 hour
    }

    def __init__(self):
        self._store: dict[str, tuple[Any, float]] = {}  # key -> (value, expiry)
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._store:
                value, expiry = self._store[key]
                if time.time() < expiry:
                    return value
                del self._store[key]
            return None

    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        with self._lock:
            self._store[key] = (value, time.time() + ttl)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def get_or_fetch(self, key: str, fetcher: Callable, ttl: int = 300) -> Any:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = fetcher()
        self.set(key, value, ttl)
        return value


_cache = DataCache()


def get_cache() -> DataCache:
    return _cache
