"""Кэш рыночных данных поверх yfinance: проверяем свежесть, отдачу
устаревшего при сбое источника и восстановление q из графика выплат.
Обращений в сеть нет.
"""

from datetime import UTC, datetime

import pytest

from app.schemas.instruments import MarketSnapshot
from app.services import market_data


def make_snapshot(spot: float) -> MarketSnapshot:
    now = datetime.now(UTC)
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
    # Подменяем поход в сеть счётчиком вызовов.
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
    # Ради этого кэш и заводился: Yahoo лёг, а страница работает.
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
    # Пустой справочник — законное состояние до первой синхронизации.
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
    # Снимок из базы не должен притворяться свежим.
    from datetime import timedelta

    old_snapshot = make_snapshot(50.0)
    stale_time = datetime.now(UTC) - timedelta(hours=5)
    aged = old_snapshot.model_copy(update={"fetched_at": stale_time})

    market_data.prime_cache([aged])

    # Пятичасовой снимок протух, поэтому пойдём в источник.
    served = market_data.get_market_snapshot("KO")
    assert source["count"] == 1
    assert served.spot != 50.0


def test_prime_cache_serves_recent_snapshot(source) -> None:
    # А свежий — переживает перезапуск и экономит поход в сеть.
    market_data.prime_cache([make_snapshot(50.0)])

    served = market_data.get_market_snapshot("KO")
    assert source["count"] == 0
    assert served.spot == 50.0
    assert served.cached is True


def test_prime_cache_rebuilds_effective_div_yield(source) -> None:
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
    # Бумага без дивидендов: пустой график даёт нулевое q, а не трейлинг.
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
    # Оставляем только акции и фонды в долларах, помечаем наличие
    # опционов.
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


def test_failed_liquidity_check_is_not_an_answer(monkeypatch) -> None:
    # Сбой проверки обязан отличаться от ответа «опционов нет»: результат
    # уходит в базу, и записать туда таймаут как факт нельзя — бумага
    # выпала бы из списка насовсем. Проверено на живом запуске: Yahoo
    # таймаутил, и все пятьдесят бумаг легли с has_options = false.
    class Broken:
        @property
        def options(self):
            raise RuntimeError("нет данных")

    monkeypatch.setattr(market_data.yf, "Ticker", lambda ticker: Broken())
    assert market_data._option_liquidity("ANY") is None


def test_liquidity_zero_when_no_expirations(monkeypatch) -> None:
    # А вот это настоящий ответ источника: цепочек нет. Его записывать можно.
    class Empty:
        options = ()

    monkeypatch.setattr(market_data.yf, "Ticker", lambda ticker: Empty())
    assert market_data._option_liquidity("ANY") == (False, 0)


def test_unchecked_instruments_are_skipped(monkeypatch) -> None:
    # Непроверенная бумага не попадает в выдачу вовсе, а не помечается
    # «без опционов»: в базе останется прежнее значение, а не ложь.
    quotes = [
        {"symbol": "GOOD", "shortName": "Проверилась", "fullExchangeName": "NYSE",
         "quoteType": "EQUITY", "currency": "USD"},
        {"symbol": "TIMEOUT", "shortName": "Таймаут", "fullExchangeName": "NYSE",
         "quoteType": "EQUITY", "currency": "USD"},
    ]
    monkeypatch.setattr(
        market_data.yf, "screen", lambda name, count: {"quotes": quotes}
    )
    monkeypatch.setattr(
        market_data, "_option_liquidity",
        lambda ticker: None if ticker == "TIMEOUT" else (True, 60),
    )

    items = market_data.fetch_screener_instruments(count=5)

    assert [item.ticker for item in items] == ["GOOD"]


def test_closed_market_leaves_liquidity_unmeasured(monkeypatch) -> None:
    quotes = [
        {"symbol": name, "shortName": name, "fullExchangeName": "NYSE",
         "quoteType": "EQUITY", "currency": "USD"}
        for name in ("AAPL", "MSFT", "KO")
    ]
    monkeypatch.setattr(
        market_data.yf, "screen", lambda name, count: {"quotes": quotes}
    )
    leftovers = {"AAPL": 0, "MSFT": 9, "KO": 3}
    monkeypatch.setattr(
        market_data, "_option_liquidity", lambda t: (True, leftovers[t])
    )

    items = market_data.fetch_screener_instruments(count=5)

    assert len(items) == 3
    assert all(item.quoted_strikes is None for item in items), (
        "максимум ниже порога — это закрытая биржа, а не измеренный неликвид"
    )
    assert all(item.has_options for item in items), (
        "наличие цепочек измеряется надёжно и в нерабочие часы"
    )


def test_single_zero_is_real_illiquidity(monkeypatch) -> None:
    # А вот если у одной бумаги ноль, а у других котировки есть — рынок
    # открыт, и ноль у неё настоящий. Его записывать нужно.
    quotes = [
        {"symbol": name, "shortName": name, "fullExchangeName": "NYSE",
         "quoteType": "EQUITY", "currency": "USD"}
        for name in ("LIQUID", "DEAD")
    ]
    monkeypatch.setattr(
        market_data.yf, "screen", lambda name, count: {"quotes": quotes}
    )
    monkeypatch.setattr(
        market_data, "_option_liquidity",
        lambda ticker: (True, 0) if ticker == "DEAD" else (True, 80),
    )

    by_ticker = {i.ticker: i for i in market_data.fetch_screener_instruments(count=5)}

    assert by_ticker["LIQUID"].quoted_strikes == 80
    assert by_ticker["DEAD"].quoted_strikes == 0


def test_all_checks_failed_is_an_error(monkeypatch) -> None:
    # Если не проверилась ни одна, молча вернуть пустой список нельзя:
    # пользователь увидел бы «получено 0» и решил, что опционов нет ни у
    # кого. Это отказ источника, и говорить надо о нём.
    quotes = [
        {"symbol": "ONE", "shortName": "Раз", "fullExchangeName": "NYSE",
         "quoteType": "EQUITY", "currency": "USD"},
    ]
    monkeypatch.setattr(
        market_data.yf, "screen", lambda name, count: {"quotes": quotes}
    )
    monkeypatch.setattr(market_data, "_option_liquidity", lambda ticker: None)

    with pytest.raises(market_data.MarketDataUnavailable, match="ни по одной"):
        market_data.fetch_screener_instruments(count=5)
