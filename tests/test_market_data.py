"""Кэш рыночных данных поверх yfinance — без обращений в сеть."""

from datetime import datetime, timezone

import pytest

from app.schemas.instruments import MarketSnapshot
from app.services import market_data


def make_snapshot(spot: float) -> MarketSnapshot:
    now = datetime.now(timezone.utc)
    return MarketSnapshot(
        ticker="KO", name="Coca-Cola", currency="USD", spot=spot,
        sigma_30d=0.26, sigma_90d=0.22, sigma_252d=0.18,
        div_yield=0.024, risk_free_rate=0.037,
        expirations=["2026-09-18"], as_of=now, fetched_at=now,
    )


@pytest.fixture(autouse=True)
def clean_cache():
    market_data.clear_cache()
    yield
    market_data.clear_cache()


@pytest.fixture
def source(monkeypatch):
    """Подменяем поход в сеть счётчиком вызовов."""
    calls = {"count": 0, "fail": False}

    def fake_fetch(instrument):
        calls["count"] += 1
        if calls["fail"]:
            raise market_data.MarketDataUnavailable("источник недоступен")
        return make_snapshot(80.0 + calls["count"])

    monkeypatch.setattr(market_data, "_fetch_snapshot", fake_fetch)
    return calls


def test_first_call_goes_to_source(source) -> None:
    snapshot = market_data.get_market_snapshot("KO")
    assert source["count"] == 1
    assert snapshot.cached is False
    assert snapshot.stale is False


def test_second_call_comes_from_cache(source) -> None:
    first = market_data.get_market_snapshot("KO")
    second = market_data.get_market_snapshot("KO")

    assert source["count"] == 1, "источник дёрнули дважды — кэш не сработал"
    assert second.spot == first.spot
    assert second.cached is True


def test_refresh_bypasses_cache(source) -> None:
    market_data.get_market_snapshot("KO")
    refreshed = market_data.get_market_snapshot("KO", refresh=True)

    assert source["count"] == 2
    assert refreshed.cached is False


def test_stale_value_served_when_source_fails(source) -> None:
    """Ради этого кэш и заводился: Yahoo лёг, а страница работает."""
    first = market_data.get_market_snapshot("KO")
    source["fail"] = True

    served = market_data.get_market_snapshot("KO", refresh=True)

    assert served.spot == first.spot
    assert served.stale is True
    assert served.cached is True


def test_failure_without_cache_raises(source) -> None:
    source["fail"] = True
    with pytest.raises(market_data.MarketDataUnavailable):
        market_data.get_market_snapshot("KO")


def test_unknown_ticker_never_touches_source(source) -> None:
    with pytest.raises(market_data.InstrumentNotFound):
        market_data.get_market_snapshot("ZZZZ")
    assert source["count"] == 0


def test_different_tickers_cached_separately(source) -> None:
    market_data.get_market_snapshot("KO")
    market_data.get_market_snapshot("AAPL")
    assert source["count"] == 2


# --- реестр инструментов ---


def test_registry_starts_empty_without_database() -> None:
    """Пустой справочник — законное состояние до первой синхронизации."""
    market_data.set_instruments([])
    assert market_data.list_instruments() == []
    assert market_data.find_instrument("AAPL") is None


def test_registry_replaced_from_database() -> None:
    from app.schemas.instruments import Instrument

    original = market_data.list_instruments()
    try:
        market_data.set_instruments(
            [Instrument(ticker="TEST", name="Тест", exchange="NYSE")]
        )
        assert [i.ticker for i in market_data.list_instruments()] == ["TEST"]
        assert market_data.find_instrument("test").name == "Тест"
        assert market_data.find_instrument("AAPL") is None
    finally:
        market_data.set_instruments(original)


def test_prime_cache_keeps_original_timestamp(source) -> None:
    """Снимок из базы не должен притворяться свежим."""
    from datetime import timedelta

    old_snapshot = make_snapshot(50.0)
    stale_time = datetime.now(timezone.utc) - timedelta(hours=5)
    aged = old_snapshot.model_copy(update={"fetched_at": stale_time})

    market_data.prime_cache([aged])

    # Пятичасовой снимок протух, поэтому пойдём в источник.
    served = market_data.get_market_snapshot("KO")
    assert source["count"] == 1
    assert served.spot != 50.0


def test_prime_cache_serves_recent_snapshot(source) -> None:
    """А свежий — переживает перезапуск и экономит поход в сеть."""
    market_data.prime_cache([make_snapshot(50.0)])

    served = market_data.get_market_snapshot("KO")
    assert source["count"] == 0
    assert served.spot == 50.0
    assert served.cached is True


def test_prime_cache_rebuilds_effective_div_yield(source) -> None:
    """q по экспирациям восстанавливается из графика выплат, а не из базы.

    В базе хранится только график отсечек: готовое q зависит от
    сегодняшней даты и назавтра уже неверно. Без этого пересчёта снимок,
    поднятый из базы, отдавал бы пустую карту, расчёт молча откатывался
    бы к трейлингу за 12 месяцев, и вся работа по датам отсечек
    переставала бы действовать до первого ручного обновления.
    """
    from datetime import date, timedelta

    from app.schemas.instruments import DividendPayment

    expiry = str(date.today() + timedelta(days=40))
    ex_date = date.today() + timedelta(days=10)

    # Так снимок и приходит из базы: график есть, карта пустая.
    from_database = make_snapshot(100.0).model_copy(
        update={
            "expirations": [expiry],
            "div_schedule": [
                DividendPayment(ex_date=ex_date, amount=1.0, estimated=False)
            ],
            "div_yield_by_expiry": {},
        }
    )

    market_data.prime_cache([from_database])
    served = market_data.get_market_snapshot("KO")

    assert source["count"] == 0
    # Отсечка внутри срока, значит q считается по ней, а не по трейлингу.
    assert served.div_yield_by_expiry[expiry] > from_database.div_yield


def test_prime_cache_without_schedule_yields_nothing(source) -> None:
    """Бумага без дивидендов: пустой график даёт нулевое q, а не трейлинг."""
    from datetime import date, timedelta

    expiry = str(date.today() + timedelta(days=40))
    from_database = make_snapshot(100.0).model_copy(
        update={"expirations": [expiry], "div_schedule": [],
                "div_yield_by_expiry": {}}
    )

    market_data.prime_cache([from_database])
    served = market_data.get_market_snapshot("KO")

    assert served.div_yield_by_expiry[expiry] == 0.0


# --- список бумаг из скринера ---


def test_screener_filters_and_marks_options(monkeypatch) -> None:
    """Оставляем только акции и фонды в долларах, помечаем наличие опционов."""
    quotes = [
        {"symbol": "AAPL", "shortName": "Apple Inc.",
         "fullExchangeName": "NasdaqGS", "quoteType": "EQUITY", "currency": "USD"},
        {"symbol": "SPY", "shortName": "SPDR S&P 500",
         "fullExchangeName": "NYSEArca", "quoteType": "ETF", "currency": "USD"},
        {"symbol": "EURUSD=X", "shortName": "EUR/USD",
         "fullExchangeName": "CCY", "quoteType": "CURRENCY", "currency": "USD"},
        {"symbol": "SAP.DE", "shortName": "SAP SE",
         "fullExchangeName": "XETRA", "quoteType": "EQUITY", "currency": "EUR"},
        {"symbol": "NOOPT", "shortName": "Без опционов",
         "fullExchangeName": "NYSE", "quoteType": "EQUITY", "currency": "USD"},
    ]
    monkeypatch.setattr(
        market_data.yf, "screen", lambda name, count: {"quotes": quotes}
    )
    monkeypatch.setattr(
        market_data,
        "_option_liquidity",
        lambda ticker: (False, 0) if ticker == "NOOPT" else (True, 75),
    )

    items = market_data.fetch_screener_instruments(count=5)
    by_ticker = {item.ticker: item for item in items}

    assert set(by_ticker) == {"AAPL", "SPY", "NOOPT"}, "валюта и не-долларовая бумага должны отсеяться"
    assert by_ticker["AAPL"].has_options is True
    assert by_ticker["AAPL"].quoted_strikes == 75
    assert by_ticker["NOOPT"].has_options is False
    assert by_ticker["NOOPT"].quoted_strikes == 0


def test_unknown_screener_is_rejected() -> None:
    with pytest.raises(ValueError, match="Неизвестный скринер"):
        market_data.fetch_screener_instruments(screener="выдуманный")


def test_screener_failure_is_wrapped(monkeypatch) -> None:
    def boom(name, count):
        raise RuntimeError("сеть недоступна")

    monkeypatch.setattr(market_data.yf, "screen", boom)
    with pytest.raises(market_data.MarketDataUnavailable):
        market_data.fetch_screener_instruments()


def test_liquidity_check_treats_errors_as_no_options(monkeypatch) -> None:
    class Broken:
        @property
        def options(self):
            raise RuntimeError("нет данных")

    monkeypatch.setattr(market_data.yf, "Ticker", lambda ticker: Broken())
    assert market_data._option_liquidity("ANY") == (False, 0)


def test_liquidity_zero_when_no_expirations(monkeypatch) -> None:
    class Empty:
        options = ()

    monkeypatch.setattr(market_data.yf, "Ticker", lambda ticker: Empty())
    assert market_data._option_liquidity("ANY") == (False, 0)
