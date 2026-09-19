"""Кэш с ограниченным сроком годности.

Отдельный модуль, а не словарь внутри market_data: логику протухания
нужно проверять тестами, не выходя в сеть, а для этого часами должен
управлять вызывающий.

Отличие от обычного TTL-кэша одно, но важное: просроченная запись не
удаляется. Источник данных неофициальный и время от времени падает, и
тогда устаревший ответ лучше, чем никакого — лишь бы пользователь видел,
что он устаревший.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Generic, TypeVar

T = TypeVar("T")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class Entry(Generic[T]):
    """Значение вместе с моментом, когда его положили."""

    value: T
    stored_at: datetime
    is_fresh: bool

    @property
    def age(self) -> timedelta:
        return _utc_now() - self.stored_at


class TTLCache(Generic[T]):
    """Значения по ключу, живущие заданное время.

    Блокировка нужна потому, что синхронные эндпоинты FastAPI выполняются
    в пуле потоков: два одновременных запроса по разным тикерам придут в
    разных потоках.
    """

    def __init__(
        self, ttl: timedelta, clock: Callable[[], datetime] = _utc_now
    ) -> None:
        self._ttl = ttl
        self._clock = clock
        self._lock = threading.Lock()
        self._items: dict[str, tuple[T, datetime]] = {}

    def get(self, key: str) -> Entry[T] | None:
        """Запись по ключу, даже если она протухла.

        Свежесть возвращается отдельным полем: решение, годится ли
        устаревшее значение, принимает вызывающий, а не кэш.
        """
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

    def set(self, key: str, value: T, stored_at: datetime | None = None) -> None:
        """Положить значение.

        stored_at нужен при разогреве из базы: там запись сделана раньше,
        и если проставить текущее время, просроченный снимок притворится
        свежим.
        """
        with self._lock:
            self._items[key] = (value, stored_at or self._clock())

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
