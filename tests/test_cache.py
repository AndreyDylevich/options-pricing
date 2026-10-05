"""Кэш рыночных данных: проверяем протухание по часам, которыми управляет
тест, и что просроченная запись остаётся доступной для отдачи при сбое.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.services.cache import TTLCache


class FakeClock:
    # Часы, которые двигает тест.

    def __init__(self) -> None:
        self.now = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def cache(clock: FakeClock) -> TTLCache[str]:
    return TTLCache(ttl=timedelta(minutes=15), clock=clock)


def test_missing_key_returns_none(cache: TTLCache[str]) -> None:
    assert cache.get("AAPL") is None


def test_stored_value_is_fresh(cache: TTLCache[str]) -> None:
    cache.set("AAPL", "снимок")
    entry = cache.get("AAPL")
    assert entry.value == "снимок"
    assert entry.is_fresh


def test_value_goes_stale_after_ttl(cache: TTLCache[str], clock: FakeClock) -> None:
    cache.set("AAPL", "снимок")
    clock.advance(minutes=14)
    assert cache.get("AAPL").is_fresh

    clock.advance(minutes=2)
    assert not cache.get("AAPL").is_fresh


def test_stale_value_is_not_dropped(cache: TTLCache[str], clock: FakeClock) -> None:
    # Просроченная запись остаётся: она пригодится, если источник упал.
    cache.set("AAPL", "снимок")
    clock.advance(days=3)

    entry = cache.get("AAPL")
    assert entry is not None
    assert entry.value == "снимок"
    assert not entry.is_fresh


def test_set_refreshes_timestamp(cache: TTLCache[str], clock: FakeClock) -> None:
    cache.set("AAPL", "старое")
    clock.advance(minutes=20)
    cache.set("AAPL", "новое")

    entry = cache.get("AAPL")
    assert entry.value == "новое"
    assert entry.is_fresh


def test_keys_are_independent(cache: TTLCache[str], clock: FakeClock) -> None:
    cache.set("AAPL", "яблоко")
    clock.advance(minutes=20)
    cache.set("KO", "кола")

    assert not cache.get("AAPL").is_fresh
    assert cache.get("KO").is_fresh


def test_clear_removes_everything(cache: TTLCache[str]) -> None:
    cache.set("AAPL", "снимок")
    cache.clear()
    assert cache.get("AAPL") is None
