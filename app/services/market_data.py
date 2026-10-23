"""Рыночные данные через yfinance: цена, историческая волатильность, дивиденды
по датам отсечек, безрисковая ставка, цепочки опционов и список бумаг со
скринера. Единственный модуль, знающий про источник данных и pandas.
"""

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise

import numpy as np
import pandas as pd
import yfinance as yf

from app.config import settings
from app.schemas.instruments import DividendPayment, Instrument, MarketSnapshot
from app.services.cache import TTLCache

logger = logging.getLogger(__name__)

TRADING_DAYS = 252

CALENDAR_DAYS = 365.0

VOLATILITY_WINDOWS = (30, 90, 252)

RISK_FREE_TICKER = "^IRX"

HISTORY_PERIOD = "2y"

VOLATILITY_HISTORY_PERIOD = "5y"

DEFAULT_SCREENER = "most_actives"
DEFAULT_SCREENER_COUNT = 50

TRADABLE_TYPES = {"EQUITY", "ETF"}

OPTION_CHECK_WORKERS = 8

LIQUIDITY_EXPIRY_INDEX = 4

_instruments: list[Instrument] = []
_by_ticker: dict[str, Instrument] = {}


def set_instruments(items: list[Instrument]) -> None:
    global _instruments, _by_ticker

    _instruments = list(items)
    _by_ticker = {item.ticker: item for item in _instruments}

_SNAPSHOTS: TTLCache[MarketSnapshot] = TTLCache(
    ttl=timedelta(minutes=settings.market_cache_ttl_minutes)
)


def clear_cache() -> None:
    _SNAPSHOTS.clear()


def prime_cache(snapshots: list[MarketSnapshot]) -> None:
    for snapshot in snapshots:
        restored = snapshot.model_copy(
            update={
                "div_yield_by_expiry": _yield_by_expiry(
                    snapshot.div_schedule,
                    snapshot.spot,
                    snapshot.risk_free_rate,
                    snapshot.expirations,
                )
            }
        )
        _SNAPSHOTS.set(snapshot.ticker, restored, stored_at=snapshot.fetched_at)


class InstrumentNotFound(Exception):
    pass


class MarketDataUnavailable(Exception):
    pass


def list_instruments() -> list[Instrument]:
    return _instruments


def find_instrument(ticker: str) -> Instrument | None:
    return _by_ticker.get(ticker.upper())


def fetch_screener_instruments(
    screener: str = DEFAULT_SCREENER, count: int = DEFAULT_SCREENER_COUNT
) -> list[Instrument]:
    if screener not in yf.PREDEFINED_SCREENER_QUERIES:
        raise ValueError(
            f"Неизвестный скринер «{screener}». Доступны: "
            + ", ".join(sorted(yf.PREDEFINED_SCREENER_QUERIES))
        )

    try:
        response = yf.screen(screener, count=count)
    except Exception as error:
        raise MarketDataUnavailable(f"скринер {screener}: {error}") from error

    candidates = [
        Instrument(
            ticker=quote["symbol"],
            name=(quote.get("shortName") or quote.get("displayName") or quote["symbol"])[:64],
            exchange=(quote.get("fullExchangeName") or "")[:16],
            currency=quote.get("currency", "USD"),
        )
        for quote in response.get("quotes", [])
        if quote.get("quoteType") in TRADABLE_TYPES
        and quote.get("currency") == "USD"
        and quote.get("symbol")
    ]

    with ThreadPoolExecutor(max_workers=OPTION_CHECK_WORKERS) as pool:
        measured = pool.map(_option_liquidity, (item.ticker for item in candidates))

    results = [
        (item, result)
        for item, result in zip(candidates, measured, strict=True)
        if result is not None
    ]

    # Признак закрытой биржи — максимум ниже порога, а не «все нули»: в ночном
    # замере у одиннадцати бумаг из сорока трёх осталось от одного до девяти.
    measured = [quoted for _, (has, quoted) in results if has]
    not_measurable = bool(measured) and max(measured) < settings.min_quoted_strikes

    if not_measurable:
        logger.info(
            "Лучшая из %d бумаг имеет %d котируемых страйков при пороге %d — "
            "похоже, биржа закрыта; ликвидность не записываем",
            len(measured), max(measured), settings.min_quoted_strikes,
        )

    checked = [
        item.model_copy(
            update={
                "has_options": has_options,
                "quoted_strikes": None if not_measurable else quoted,
            }
        )
        for item, (has_options, quoted) in results
    ]

    skipped = len(candidates) - len(checked)
    if skipped:
        logger.warning(
            "Опционы не удалось проверить по %d бумагам из %d — они пропущены",
            skipped, len(candidates),
        )

    if candidates and not checked:
        raise MarketDataUnavailable(
            f"скринер ответил, но проверить опционы не удалось ни по одной из "
            f"{len(candidates)} бумаг"
        )

    return checked


# None означает «проверить не удалось» и отличается от (False, 0). Ответ
# источника идёт в базу, и записать туда сбой сети как факт нельзя: бумага
# выпала бы из списка насовсем.
def _option_liquidity(ticker: str) -> tuple[bool, int] | None:
    try:
        yf_ticker = yf.Ticker(ticker)
        expirations = yf_ticker.options
        if not expirations:
            return False, 0

        chain = yf_ticker.option_chain(
            expirations[min(LIQUIDITY_EXPIRY_INDEX, len(expirations) - 1)]
        )
        quoted = sum(
            int(((frame["bid"] > 0) & (frame["ask"] > 0)).sum())
            for frame in (chain.calls, chain.puts)
        )
        return True, quoted
    except Exception:
        return None


def get_market_snapshot(ticker: str, refresh: bool = False) -> MarketSnapshot:
    instrument = find_instrument(ticker)
    if instrument is None:
        raise InstrumentNotFound(ticker)

    cached = _SNAPSHOTS.get(instrument.ticker)

    if cached is not None and cached.is_fresh and not refresh:
        return cached.value.model_copy(update={"cached": True})

    try:
        snapshot = _fetch_snapshot(instrument)
    except MarketDataUnavailable:
        if cached is None:
            raise
        return cached.value.model_copy(update={"cached": True, "stale": True})

    _SNAPSHOTS.set(instrument.ticker, snapshot)
    return snapshot


def _fetch_snapshot(instrument: Instrument) -> MarketSnapshot:
    yf_ticker = yf.Ticker(instrument.ticker)
    history = _load_history(yf_ticker, instrument.ticker)

    close = history["Close"]
    spot = float(close.iloc[-1])
    volatilities = _historical_volatilities(close)

    rate = get_risk_free_rate()
    expirations = _expirations(yf_ticker)
    schedule = _dividend_schedule(history, yf_ticker, _horizon(expirations))

    return MarketSnapshot(
        ticker=instrument.ticker,
        name=instrument.name,
        currency=instrument.currency,
        spot=spot,
        sigma_30d=volatilities[30],
        sigma_90d=volatilities[90],
        sigma_252d=volatilities[252],
        div_yield=_trailing_dividend_yield(history, spot),
        div_schedule=schedule,
        div_yield_by_expiry=_yield_by_expiry(schedule, spot, rate, expirations),
        risk_free_rate=rate,
        expirations=expirations,
        as_of=history.index[-1].to_pydatetime(),
        fetched_at=datetime.now(UTC),
    )


def _horizon(expirations: list[str]) -> date:
    if not expirations:
        return date.today()

    return date.fromisoformat(max(expirations))


def get_risk_free_rate() -> float:
    try:
        history = yf.Ticker(RISK_FREE_TICKER).history(period="5d")
    except Exception:
        return settings.default_risk_free_rate

    if history.empty:
        return settings.default_risk_free_rate

    return float(history["Close"].iloc[-1]) / 100


# auto_adjust=True принципиален: без него сплит даёт скачок вроде -50% за
# день, и волатильность оказывается завышенной на пустом месте.
def _load_history(
    yf_ticker: yf.Ticker, ticker: str, period: str = HISTORY_PERIOD
) -> pd.DataFrame:
    try:
        history = yf_ticker.history(period=period, interval="1d", auto_adjust=True)
    except Exception as error:
        raise MarketDataUnavailable(f"{ticker}: {error}") from error

    if history.empty:
        raise MarketDataUnavailable(f"{ticker}: источник вернул пустую историю")

    return history


def get_log_returns(ticker: str, period: str = VOLATILITY_HISTORY_PERIOD) -> np.ndarray:
    instrument = find_instrument(ticker)
    if instrument is None:
        raise InstrumentNotFound(ticker)

    history = _load_history(yf.Ticker(instrument.ticker), instrument.ticker, period)

    return np.log(history["Close"]).diff().dropna().to_numpy()


def _historical_volatilities(close: pd.Series) -> dict[int, float | None]:
    log_returns = np.log(close).diff().dropna()

    result: dict[int, float | None] = {}
    for window in VOLATILITY_WINDOWS:
        if len(log_returns) < window:
            result[window] = None
            continue

        daily_std = log_returns.tail(window).std(ddof=1)
        result[window] = float(daily_std * np.sqrt(TRADING_DAYS))

    return result


def _trailing_dividend_yield(history: pd.DataFrame, spot: float) -> float:
    if "Dividends" not in history.columns:
        return 0.0

    cutoff = history.index[-1] - timedelta(days=365)
    paid = history.loc[history.index > cutoff, "Dividends"].sum()

    return float(paid) / spot if spot else 0.0


MAX_PROJECTED_PAYMENTS = 16

DEFAULT_PAYMENT_PERIOD_DAYS = 365

MISSED_PAYMENT_TOLERANCE = 1.5


def _dividend_schedule(
    history: pd.DataFrame, yf_ticker: yf.Ticker, horizon: date
) -> list[DividendPayment]:
    paid = _past_dividends(history)
    if not paid:
        return []

    ex_dates = [ex_date for ex_date, _ in paid]
    amount = paid[-1][1]
    period = timedelta(days=_payment_period(ex_dates))

    today = date.today()
    announced = _announced_ex_date(yf_ticker)

    if announced is not None and announced > today:
        current, estimated = announced, False
    else:
        silent = today - ex_dates[-1] > period * MISSED_PAYMENT_TOLERANCE

        if silent or _dividend_cancelled(yf_ticker):
            return []

        current, estimated = ex_dates[-1] + period, True
        while current <= today:
            current += period

    schedule: list[DividendPayment] = []
    while current <= horizon and len(schedule) < MAX_PROJECTED_PAYMENTS:
        schedule.append(
            DividendPayment(ex_date=current, amount=amount, estimated=estimated)
        )
        current += period
        estimated = True

    return schedule


def _past_dividends(history: pd.DataFrame) -> list[tuple[date, float]]:
    if "Dividends" not in history.columns:
        return []

    paid = history.loc[history["Dividends"] > 0, "Dividends"]

    return [
        (stamp.date(), float(amount)) for stamp, amount in paid.items()
    ]


def _payment_period(ex_dates: list[date]) -> int:
    if len(ex_dates) < 2:
        return DEFAULT_PAYMENT_PERIOD_DAYS

    gaps = sorted(
        (later - earlier).days for earlier, later in pairwise(ex_dates)
    )
    median = gaps[len(gaps) // 2]

    return max(median, 1)


def _announced_ex_date(yf_ticker: yf.Ticker) -> date | None:
    try:
        calendar = yf_ticker.calendar
    except Exception:
        return None

    if not isinstance(calendar, dict):
        return None

    value = calendar.get("Ex-Dividend Date")

    if isinstance(value, datetime):
        return value.date()

    return value if isinstance(value, date) else None


# Недоступный или пустой info — это «не знаю», а не «отменили»: молчание
# Yahoo не повод обнулять дивиденд живой компании.
def _dividend_cancelled(yf_ticker: yf.Ticker) -> bool:
    try:
        info = yf_ticker.info
    except Exception:
        return False

    if not isinstance(info, dict) or "dividendRate" not in info:
        return False

    return not info["dividendRate"]


def _yield_by_expiry(
    schedule: list[DividendPayment],
    spot: float,
    rate: float,
    expirations: list[str],
) -> dict[str, float]:
    return {
        expiry: effective_div_yield(
            schedule, spot, rate, date.fromisoformat(expiry)
        )
        for expiry in expirations
    }


# S * exp(-q * T) = S - D, откуда q = -ln(1 - D/S) / T. Для европейского
# опциона это не приближение, а точная замена: цена зависит от дивидендов
# только через множитель exp(-q * T).
def effective_div_yield(
    schedule: list[DividendPayment], spot: float, rate: float, expiry: date
) -> float:
    days = (expiry - date.today()).days
    if days <= 0 or spot <= 0:
        return 0.0

    present_value = _dividends_within(schedule, rate, expiry)

    if present_value <= 0 or present_value >= spot:
        return 0.0

    return -math.log(1 - present_value / spot) / (days / CALENDAR_DAYS)


def _dividends_within(
    schedule: list[DividendPayment], rate: float, expiry: date
) -> float:
    today = date.today()

    return sum(
        payment.amount
        * math.exp(-rate * ((payment.ex_date - today).days / CALENDAR_DAYS))
        for payment in schedule
        if today < payment.ex_date <= expiry
    )


def _expirations(yf_ticker: yf.Ticker) -> list[str]:
    try:
        raw = list(yf_ticker.options)
    except Exception:
        return []

    today = datetime.now(UTC).date()
    return [value for value in raw if date.fromisoformat(value) > today]


def get_option_chain(ticker: str, expiry: str) -> tuple[list[tuple[str, dict]], int]:
    instrument = find_instrument(ticker)
    if instrument is None:
        raise InstrumentNotFound(ticker)

    try:
        chain = yf.Ticker(instrument.ticker).option_chain(expiry)
    except Exception as error:
        raise MarketDataUnavailable(f"{ticker} на {expiry}: {error}") from error

    rows: list[tuple[str, dict]] = []
    for side, frame in (("call", chain.calls), ("put", chain.puts)):
        live = frame[(frame["bid"] > 0) & (frame["ask"] > 0)]
        for record in live.to_dict("records"):
            rows.append((side, record))

    total = len(chain.calls) + len(chain.puts)
    return rows, total
