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

# Торговых дней в году: множитель для перевода дневной волатильности в
# годовую. Календарные 365 здесь не годятся — цена не меняется в выходные.
TRADING_DAYS = 252

# Календарные дни в году. Не опечатка рядом с TRADING_DAYS: дивиденд
# дисконтируется каждый день, включая выходные, а волатильность приводится
# к году по торговым. Разница намеренная.
CALENDAR_DAYS = 365.0

# Окна для исторической волатильности, в торговых днях.
VOLATILITY_WINDOWS = (30, 90, 252)

# Тикер 13-недельных казначейских векселей США. Стандартный ориентир для
# безрисковой ставки по американским опционам.
RISK_FREE_TICKER = "^IRX"

# Глубина истории. Двух лет хватает на окно в 252 дня с запасом.
HISTORY_PERIOD = "2y"

# Отдельная, более длинная история — под GARCH и EGARCH. Двух лет им мало:
# это около 500 наблюдений на четыре параметра, и оценки гуляют сильнее
# самой волатильности. Пять лет дают порядка 1250 точек. Тянется только по
# запросу, обычному расчёту эта история не нужна.
VOLATILITY_HISTORY_PERIOD = "5y"

# Скринер, из которого берётся список бумаг. most_actives — самые
# ликвидные за день: по неликвиду цепочки опционов либо нет, либо в ней
# пустые котировки.
DEFAULT_SCREENER = "most_actives"
DEFAULT_SCREENER_COUNT = 50

# Что считаем пригодным: обычные акции и биржевые фонды. Всё остальное
# (валюты, фьючерсы, индексы) наши модели не описывают.
TRADABLE_TYPES = {"EQUITY", "ETF"}

# Проверка ликвидности — два запроса на бумагу, около секунды каждый.
OPTION_CHECK_WORKERS = 8

# Какую по счёту дату экспирации брать для замера. Пятая — это примерно
# месяц: ближние недельные цепочки узкие, дальние полупустые у всех.
LIQUIDITY_EXPIRY_INDEX = 4

# Реестр в памяти: источник истины — таблица instruments, но читать её на
# каждый запрос незачем, список меняется редко. Ни одного тикера в коде
# не зашито: всё приходит из скринера Yahoo через sync.
_instruments: list[Instrument] = []
_by_ticker: dict[str, Instrument] = {}


# Пустой список — законное состояние: так выглядит база до первой
# синхронизации.
def set_instruments(items: list[Instrument]) -> None:
    global _instruments, _by_ticker

    _instruments = list(items)
    _by_ticker = {item.ticker: item for item in _instruments}

# Кэш рыночных снимков. Без него каждый выбор акции в форме — поход в
# сеть на секунду-другую, а yfinance неофициальный и имеет ограничения на
# частоту запросов.
_SNAPSHOTS: TTLCache[MarketSnapshot] = TTLCache(
    ttl=timedelta(minutes=settings.market_cache_ttl_minutes)
)


# Сбросить кэш. Нужен тестам и ручной перезагрузке данных.
def clear_cache() -> None:
    _SNAPSHOTS.clear()


# Метка времени берётся из самой записи, а не текущая: иначе снимок
# трёхдневной давности притворился бы свежим и пользователь увидел бы
# позавчерашнюю цену без всякой пометки. Эффективное q пересчитывается
# заново, а не достаётся из базы: оно зависит от сегодняшней даты. Вчерашний
# снимок видел отсечку послезавтра внутри недельного опциона; сегодня тот же
# опцион стал шестидневным, а отсечка — завтрашней. Хранить готовые числа
# значило бы раздавать вчерашние.
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


# Тикера нет в списке доступных.
class InstrumentNotFound(Exception):
    pass


# Источник данных не ответил или вернул пустоту.
class MarketDataUnavailable(Exception):
    pass


def list_instruments() -> list[Instrument]:
    return _instruments


# Инструмент по тикеру, без учёта регистра. None, если такого нет.
def find_instrument(ticker: str) -> Instrument | None:
    return _by_ticker.get(ticker.upper())


# Скринер отбирает по ликвидности и об опционах ничего не знает, поэтому по
# каждой бумаге приходится отдельно спросить цепочку экспираций. Это
# примерно секунда на тикер, поэтому проверки идут параллельно —
# последовательно полсотни бумаг заняли бы минуту.
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

    # Пока биржа закрыта, Yahoo отдаёт нулевые или почти нулевые bid и ask:
    # остаются лишь остаточные котировки по единичным контрактам. Записать
    # это как неликвид нельзя — список бумаг оказывался бы пустым каждую
    # ночь и все выходные.
    #
    # Признак — максимум ниже порога. Скринер отдаёт самые активные бумаги
    # дня, и в торговые часы такого не бывает: замер в 16:40 по Нью-Йорку
    # дал максимум 376 страйков, замер в 02:23 — девять.
    #
    # Проверять «все нули» оказалось мало: в том же ночном замере у
    # одиннадцати бумаг из сорока трёх нашлось от одного до девяти
    # котируемых страйков, и условие не срабатывало.
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

    # Непроверенные бумаги не возвращаем вовсе, а не помечаем «опционов
    # нет». Иначе временный таймаут записался бы в базу как факт, и бумага
    # выпала бы из списка насовсем — до случайно удачной синхронизации.
    # Проверено на чистом запуске: Yahoo таймаутил, и все пятьдесят бумаг
    # легли в базу с has_options = false, после чего список остался пустым.
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


# Есть ли опционы и сколько страйков реально котируется. Наличие цепочки ещё
# ничего не значит: у части бумаг она есть, но bid и ask пустые почти везде.
# Разворачивать подразумеваемую волатильность нечего, если цены нет. Мерим по
# сроку примерно через месяц: ближние недели слишком узкие, дальние всегда
# полупустые.
#
# None означает «проверить не удалось» и отличается от (False, 0), то есть
# «проверили, опционов нет». Разница принципиальна: ответ источника мы
# сохраняем в базу, и записать туда сбой сети как факт нельзя — бумага
# выпала бы из списка насовсем. То же правило, что с отменённым дивидендом.
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


# Свежий ответ живёт MARKET_CACHE_TTL_MINUTES минут. Если срок вышел, а
# источник недоступен, отдаём устаревший снимок с пометкой stale: для
# неофициального источника старые данные полезнее пустой ошибки, лишь бы
# пользователь видел, что они старые.
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


# Обращение к источнику: история цен по бумаге и котировка ^IRX.
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


# До какой даты достраивать график выплат. Дальше самой поздней экспирации
# смысла нет: выплаты за ней ни в один опцион не попадут. Берём max, а не
# последний элемент: порядок дат приходит от Yahoo, и полагаться на него
# незачем — строки в формате ISO сравниваются как даты.
def _horizon(expirations: list[str]) -> date:
    if not expirations:
        return date.today()

    return date.fromisoformat(max(expirations))


# Безрисковая ставка долей единицы. Yahoo котирует ^IRX в процентах (3.71),
# нам нужно 0.0371. Если котировка недоступна, берём значение из .env, чтобы
# недоступность одного вспомогательного тикера не роняла весь расчёт.
def get_risk_free_rate() -> float:
    try:
        history = yf.Ticker(RISK_FREE_TICKER).history(period="5d")
    except Exception:
        return settings.default_risk_free_rate

    if history.empty:
        return settings.default_risk_free_rate

    return float(history["Close"].iloc[-1]) / 100


# auto_adjust=True принципиален: без него в ряду остаются провалы в дни
# отсечки по дивидендам, а сплит даёт скачок вроде -50% за день, из-за чего
# волатильность оказывается завышенной на пустом месте.
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


# Наружу уходит именно массив, а не Series: pandas за границы этого модуля
# не выходит, а моделям волатильности в `app/pricing/` ничего кроме чисел и
# не нужно.
def get_log_returns(ticker: str, period: str = VOLATILITY_HISTORY_PERIOD) -> np.ndarray:
    instrument = find_instrument(ticker)
    if instrument is None:
        raise InstrumentNotFound(ticker)

    history = _load_history(yf.Ticker(instrument.ticker), instrument.ticker, period)

    return np.log(history["Close"]).diff().dropna().to_numpy()


# Годовая историческая волатильность по каждому окну. Стандартное отклонение
# логарифмических доходностей, умноженное на корень из числа торговых дней в
# году. Окна, на которые не хватило истории, возвращаются как None — лучше
# пустое поле, чем оценка по трём точкам.
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


# Сумма дивидендов за последние 12 месяцев, делённая на цену. Грубая оценка,
# годная как значение по умолчанию в форме, но не как итоговое q. Две
# неточности, разные по величине: 1. Смотрит назад, а модели нужна ожидаемая
# доходность вперёд. Мелочь: у Yahoo forward-оценка (info["dividendRate"])
# отличается от фактической на 2-3% относительных, на цене это копейки. 2.
# Размазывает четыре дискретные выплаты в постоянную ставку. Вот это
# существенно на коротких сроках: у опциона с экспирацией до ближайшей
# отсечки правильное q равно нулю, а у опциона, внутрь которого отсечка
# попадает, эффективное q может оказаться втрое выше среднегодового. Вторую
# лечит effective_div_yield: под конкретную экспирацию q считается по датам
# отсечек. Это значение и подставляется в форму, а трейлинг остаётся
# справочной цифрой «сколько платили за год».
def _trailing_dividend_yield(history: pd.DataFrame, spot: float) -> float:
    if "Dividends" not in history.columns:
        return 0.0

    cutoff = history.index[-1] - timedelta(days=365)
    paid = history.loc[history.index > cutoff, "Dividends"].sum()

    return float(paid) / spot if spot else 0.0


# Предел на число достроенных вперёд выплат: страховка от бесконечного
# цикла, если ритм посчитался нулевым. Шестнадцати кварталов хватает на
# самые дальние LEAPS.
MAX_PROJECTED_PAYMENTS = 16

# Шаг между отсечками, когда история дала меньше двух: считаем такую
# бумагу платящей раз в год — занизить частоту безопаснее, чем завысить.
DEFAULT_PAYMENT_PERIOD_DAYS = 365

# Сколько ритмов тишины считаем отменой дивиденда. Полтора квартала — это
# один пропущенный платёж плюс запас: объявление регулярно задерживается
# на неделю-другую, а вот пропуск целого квартала случайностью не бывает.
# Проверено на списке: у платящих бумаг тишина не превышает 56 дней, у
# отменивших (INTC, AAL) идёт годами.
MISSED_PAYMENT_TOLERANCE = 1.5


# Отсечки, которые ещё предстоят, вплоть до даты horizon. Модели просят
# непрерывную ставку q, но бумага платит четыре раза в год скачками, и на
# коротком сроке это разные вещи: опцион с экспирацией до ближайшей отсечки
# дивиденда не видит вовсе. Чтобы посчитать q под конкретную экспирацию,
# нужно знать, какие выплаты в её срок попадают, — этот график и есть ответ.
# Ближайшая дата берётся из календаря Yahoo, если компания её уже объявила;
# дальше график продолжается по среднему шагу прошлых выплат. Продолженные
# даты помечаются estimated: дальше объявленной будущих отсечек не знает
# никто — совет директоров объявляет дивиденд поквартально, и в базах этих
# дат просто нет. Отменённый дивиденд отсекается двумя проверками. Первая —
# прогноз Yahoo на год вперёд, он реагирует на объявление сразу. Вторая —
# тишина в истории: платежей нет дольше полутора ритмов. Без них компания
# вроде Intel, свернувшей дивиденд в 2024-м, ещё два года получала бы от нас
# фантомные отсечки, а вместе с ними заниженную цену колла.
def _dividend_schedule(
    history: pd.DataFrame, yf_ticker: yf.Ticker, horizon: date
) -> list[DividendPayment]:
    paid = _past_dividends(history)
    if not paid:
        return []

    ex_dates = [ex_date for ex_date, _ in paid]
    # Размер берём у последней выплаты, а не средний за год: компании
    # повышают дивиденд раз в год и дальше держат новый уровень.
    amount = paid[-1][1]
    period = timedelta(days=_payment_period(ex_dates))

    today = date.today()
    announced = _announced_ex_date(yf_ticker)

    if announced is not None and announced > today:
        current, estimated = announced, False
    else:
        silent = today - ex_dates[-1] > period * MISSED_PAYMENT_TOLERANCE

        # Спрашиваем Yahoo только здесь: если календарь уже назвал дату,
        # компания очевидно платит, и лишний запрос в сеть ни к чему.
        if silent or _dividend_cancelled(yf_ticker):
            return []

        # Календарь молчит или показывает уже прошедшую отсечку — у
        # большинства плательщиков так и есть, следующий совет ещё не
        # собирался. Продолжаем от последней фактической своим шагом.
        current, estimated = ex_dates[-1] + period, True
        while current <= today:
            current += period

    schedule: list[DividendPayment] = []
    while current <= horizon and len(schedule) < MAX_PROJECTED_PAYMENTS:
        schedule.append(
            DividendPayment(ex_date=current, amount=amount, estimated=estimated)
        )
        current += period
        # Подтверждена только первая дата, всё, что после неё, — прогноз.
        estimated = True

    return schedule


# Фактические отсечки из истории цен: дата и размер выплаты. Отдельного
# запроса не требует — колонка Dividends приходит в том же history(), что и
# цены. auto_adjust=True на неё не влияет: подгоняются цены, а размеры
# выплат остаются как были.
def _past_dividends(history: pd.DataFrame) -> list[tuple[date, float]]:
    if "Dividends" not in history.columns:
        return []

    paid = history.loc[history["Dividends"] > 0, "Dividends"]

    return [
        (stamp.date(), float(amount)) for stamp, amount in paid.items()
    ]


# Типичный шаг между отсечками в днях. Медиана, а не среднее: у KO
# промежутки гуляют от 77 до 105 дней, и один сдвинутый на праздники квартал
# не должен утаскивать оценку.
def _payment_period(ex_dates: list[date]) -> int:
    if len(ex_dates) < 2:
        return DEFAULT_PAYMENT_PERIOD_DAYS

    gaps = sorted(
        (later - earlier).days for earlier, later in pairwise(ex_dates)
    )
    median = gaps[len(gaps) // 2]

    return max(median, 1)


# Объявленная дата ближайшей отсечки из календаря Yahoo. Возвращает что
# дали, не проверяя на будущее: у AAPL в этом поле лежит уже прошедшая дата
# последней выплаты, у KO — действительно следующая. Решает вызывающий.
def _announced_ex_date(yf_ticker: yf.Ticker) -> date | None:
    try:
        calendar = yf_ticker.calendar
    except Exception:
        # Календарь — вспомогательный запрос. Его недоступность не повод
        # ронять снимок: график достроится от истории.
        return None

    if not isinstance(calendar, dict):
        return None

    value = calendar.get("Ex-Dividend Date")

    if isinstance(value, datetime):
        return value.date()

    return value if isinstance(value, date) else None


# Говорит ли Yahoo, что выплат больше не будет. Поле dividendRate — прогноз
# на год вперёд, а не факт за прошлый: у Intel, свернувшего дивиденд в
# 2024-м, там None, у MRK — 3.4. Тем оно и ценно: на объявление об отмене
# оно реагирует сразу, тогда как тишина в истории накопится только к
# следующему кварталу. Запрос тяжёлый, около 0.7 секунды, поэтому вызывается
# лишь тогда, когда календарь будущей отсечки не назвал. Недоступный или
# пустой info — это «не знаю», а не «отменили»: молчание Yahoo не повод
# обнулять дивиденд живой компании. В этом случае решение остаётся за
# проверкой по тишине.
def _dividend_cancelled(yf_ticker: yf.Ticker) -> bool:
    try:
        info = yf_ticker.info
    except Exception:
        return False

    if not isinstance(info, dict) or "dividendRate" not in info:
        return False

    # None и 0 значат одно: выплат на год вперёд не ожидается.
    return not info["dividendRate"]


# Эффективное q под каждую доступную экспирацию. Считаем сразу по всем, а не
# по запросу: выплат единицы, дат экспирации десятки, весь перебор —
# микросекунды, зато форма переключает срок без похода на сервер.
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


# Ставка q, эквивалентная дискретным выплатам внутри срока опциона.
# Складываем приведённую стоимость D всех отсечек, попадающих в срок, и
# подбираем такую непрерывную ставку, при которой актив теряет ровно столько
# же: S * exp(-q * T) = S - D, откуда q = -ln(1 - D/S) / T. Для европейского
# опциона это не приближение, а точная замена: цена зависит от дивидендов
# только через множитель exp(-q * T), и он совпадает по построению. На
# коротком сроке значение получается большим — годовая ставка, в которую
# пересчитана выплата за две недели. Это не ошибка: полторы сотни процентов
# годовых на две недели дают те же полпроцента цены.
def effective_div_yield(
    schedule: list[DividendPayment], spot: float, rate: float, expiry: date
) -> float:
    days = (expiry - date.today()).days
    if days <= 0 or spot <= 0:
        return 0.0

    present_value = _dividends_within(schedule, rate, expiry)

    # Выплаты дороже самой бумаги — данные битые, логарифм не определён.
    # Молча отдаём ноль: сорванный снимок хуже неточного q.
    if present_value <= 0 or present_value >= spot:
        return 0.0

    return -math.log(1 - present_value / spot) / (days / CALENDAR_DAYS)


# Приведённая к сегодня стоимость выплат в промежутке (сегодня, expiry].
# Отсечка ровно в день экспирации учитывается: цена падает на открытии этого
# дня, то есть до расчёта по опциону.
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


# Доступные даты экспирации опционов, у которых срок ещё не истёк.
# Сегодняшняя дата отбрасывается вместе с прошедшими: такие контракты
# реально торгуются до закрытия рынка, но для нас это T = 0, на котором все
# три модели вырождаются — цена схлопывается во внутреннюю стоимость, gamma
# и theta уходят в бесконечность. Пустой список — законный ответ: у части
# бумаг опционов нет.
def _expirations(yf_ticker: yf.Ticker) -> list[str]:
    try:
        raw = list(yf_ticker.options)
    except Exception:
        return []

    today = datetime.now(UTC).date()
    return [value for value in raw if date.fromisoformat(value) > today]


# Сырая цепочка опционов на дату: только контракты с живыми котировками.
# Контракты без bid или ask отбрасываются сразу: середина спреда по ним не
# определена, а решать обратную задачу по цене последней сделки нельзя — она
# могла состояться неделю назад при другой цене актива. Наружу отдаются
# простые словари, а не DataFrame: pandas за границы этого модуля не
# выходит. Второе значение — сколько контрактов было в цепочке всего, чтобы
# наверху можно было сказать, сколько отброшено.
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
