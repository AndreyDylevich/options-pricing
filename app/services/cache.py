"""Кэш значений с ограниченным сроком годности. От обычного отличается тем,
что просроченную запись не удаляет: источник данных неофициальный и
временами падает, и тогда устаревший ответ лучше, чем никакого.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class Entry[T]:
    value: T
    stored_at: datetime
    is_fresh: bool

    @property
    def age(self) -> timedelta:
        return _utc_now() - self.stored_at


class TTLCache[T]:
    # Часами управляет вызывающий: иначе протухание не проверить тестами.
    def __init__(
        self, ttl: timedelta, clock: Callable[[], datetime] = _utc_now
    ) -> None:
        self._ttl = ttl
        self._clock = clock

        # Синхронные эндпоинты FastAPI идут в пуле потоков: два запроса придут в разных.
        self._lock = threading.Lock()
        self._items: dict[str, tuple[T, datetime]] = {}

    # Протухшая запись отдаётся с пометкой: годится ли она, решает вызывающий.
    def get(self, key: str) -> Entry[T] | None:
        with self._lock:
            item = self._items.get(key)

        if item is None:
            return None

        value, stored_at = item
        return Entry(
            value=value,
            stored_at=stored_at,
            is_fresh=self._clock() - stored_at < self._ttl,
        )

    # stored_at нужен при разогреве из базы: иначе просроченный снимок притворится свежим.
    def set(self, key: str, value: T, stored_at: datetime | None = None) -> None:
        with self._lock:
            self._items[key] = (value, stored_at or self._clock())

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
