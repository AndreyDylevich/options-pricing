"""График дивидендных отсечек и эффективное q под конкретную экспирацию.

Сети здесь нет: история подсовывается готовым DataFrame, календарь —
заглушкой. Проверяется ровно то, ради чего всё затевалось: одного q на
все сроки не бывает.
"""

import math
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from app.pricing import black_scholes
from app.pricing.models import MAX_DIV_YIELD, OptionParams, OptionType
from app.schemas.instruments import DividendPayment
from app.services import market_data

TODAY = date.today()

# Круглые числа вместо настоящих: 100 за акцию и выплата в 1 — чтобы
# ожидаемое значение можно было посчитать в уме.
SPOT = 100.0
AMOUNT = 1.0
RATE = 0.04


def history_with(ex_dates: list[date], amount: float = AMOUNT) -> pd.DataFrame:
    """История цен с колонкой Dividends — как её отдаёт yfinance."""
    index = pd.to_datetime([str(value) for value in ex_dates])
    return pd.DataFrame({"Dividends": [amount] * len(ex_dates)}, index=index)


# Так выглядит info у компании, которая платит: прогноз на год вперёд.
PAYING = {"dividendRate": 3.4}

# А так — у свернувшей дивиденд. Ровно это отдаёт Yahoo по INTC.
CANCELLED = {"dividendRate": None}


class FakeTicker:
    """Заглушка yf.Ticker: из всего API нужны календарь и info."""

    def __init__(self, calendar=None, info=PAYING, broken: bool = False):
        self._calendar = calendar
        self._info = info
        self._broken = broken

    @property
    def calendar(self):
        if self._broken:
            raise RuntimeError("календарь недоступен")
        return self._calendar

    @property
    def info(self):
        if self._broken:
            raise RuntimeError("info недоступен")
        return self._info


def quarterly(count: int, last_days_ago: int = 10) -> list[date]:
    """Ежеквартальные отсечки, последняя — last_days_ago дней назад."""
    last = TODAY - timedelta(days=last_days_ago)
    return [last - timedelta(days=91 * step) for step in reversed(range(count))]


# --- разбор истории --------------------------------------------------


def test_past_dividends_reads_dates_and_amounts() -> None:
    dates = quarterly(4)
    paid = market_data._past_dividends(history_with(dates))

    assert [ex_date for ex_date, _ in paid] == dates
    assert all(amount == AMOUNT for _, amount in paid)


def test_past_dividends_empty_without_column() -> None:
    frame = pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2026-01-02"]))
    assert market_data._past_dividends(frame) == []


def test_past_dividends_skips_days_without_payment() -> None:
    """В истории дивиденд стоит только в дни отсечки, в остальных — ноль."""
    index = pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-04"])
    frame = pd.DataFrame({"Dividends": [0.0, 0.5, 0.0]}, index=index)

    assert market_data._past_dividends(frame) == [(date(2026, 1, 3), 0.5)]


# --- ритм выплат -----------------------------------------------------


def test_payment_period_is_median_of_gaps() -> None:
    """Один сдвинутый квартал не должен утаскивать оценку."""
    dates = [
        date(2026, 1, 1),
        date(2026, 4, 1),   # 90
        date(2026, 7, 1),   # 91
        date(2026, 12, 1),  # 153, праздничный выброс
    ]
    assert market_data._payment_period(dates) == 91


def test_payment_period_defaults_to_year_on_single_payment() -> None:
    assert market_data._payment_period([TODAY]) == 365
    assert market_data._payment_period([]) == 365


# --- календарь -------------------------------------------------------


def test_announced_ex_date_returns_date() -> None:
    ticker = FakeTicker({"Ex-Dividend Date": date(2026, 9, 15)})
    assert market_data._announced_ex_date(ticker) == date(2026, 9, 15)


def test_announced_ex_date_narrows_datetime() -> None:
    ticker = FakeTicker({"Ex-Dividend Date": datetime(2026, 9, 15, 9, 30)})
    assert market_data._announced_ex_date(ticker) == date(2026, 9, 15)


@pytest.mark.parametrize(
    "ticker",
    [
        FakeTicker({}),
        FakeTicker({"Ex-Dividend Date": None}),
        FakeTicker(None),
        FakeTicker(broken=True),
    ],
    ids=["нет поля", "поле пустое", "календаря нет", "запрос упал"],
)
def test_announced_ex_date_absent(ticker) -> None:
    """Недоступный календарь не повод ронять снимок: график достроится."""
    assert market_data._announced_ex_date(ticker) is None


# --- график выплат ---------------------------------------------------


def test_schedule_empty_without_dividends() -> None:
    frame = pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2026-01-02"]))
    schedule = market_data._dividend_schedule(
        frame, FakeTicker(), TODAY + timedelta(days=365)
    )
    assert schedule == []


def test_schedule_starts_from_announced_date() -> None:
    announced = TODAY + timedelta(days=11)
    schedule = market_data._dividend_schedule(
        history_with(quarterly(4)),
        FakeTicker({"Ex-Dividend Date": announced}),
        TODAY + timedelta(days=200),
    )

    assert schedule[0].ex_date == announced
    assert schedule[0].estimated is False
    # Подтверждена только первая дата, дальше идёт наша арифметика.
    assert all(payment.estimated for payment in schedule[1:])


def test_schedule_ignores_past_announced_date() -> None:
    """У AAPL в календаре лежит уже прошедшая отсечка — она не годится."""
    schedule = market_data._dividend_schedule(
        history_with(quarterly(4, last_days_ago=10)),
        FakeTicker({"Ex-Dividend Date": TODAY - timedelta(days=10)}),
        TODAY + timedelta(days=200),
    )

    assert schedule[0].ex_date > TODAY
    assert schedule[0].estimated is True


def test_schedule_empty_when_dividend_was_cancelled() -> None:
    """Компания платила квартально и год назад перестала — фантомов быть не должно.

    Так поступил Intel в 2024-м. Пока выплаты остаются в двухлетней
    истории, а календарь ничего не обещает, соблазн продолжить график
    велик — и мы вычитали бы из цены деньги, которых не будет.
    """
    stopped = [TODAY - timedelta(days=365 + 91 * step) for step in reversed(range(4))]
    schedule = market_data._dividend_schedule(
        history_with(stopped), FakeTicker(), TODAY + timedelta(days=200)
    )

    assert schedule == []


def test_schedule_survives_one_late_payment() -> None:
    """Объявление задержалось на месяц — это не отмена.

    По списку бумаг тишина у платящих доходит до 56 дней (ORCL, MRVL):
    совет ещё не собирался. Обрубать график на этом нельзя.
    """
    late = [TODAY - timedelta(days=120 + 91 * step) for step in reversed(range(4))]
    schedule = market_data._dividend_schedule(
        history_with(late), FakeTicker(), TODAY + timedelta(days=200)
    )

    assert schedule
    assert schedule[0].ex_date > TODAY


def test_announced_date_outweighs_long_silence() -> None:
    """Календарь пообещал отсечку — значит платят, сколько бы ни молчали."""
    announced = TODAY + timedelta(days=20)
    stopped = [TODAY - timedelta(days=500 + 91 * step) for step in reversed(range(4))]

    schedule = market_data._dividend_schedule(
        history_with(stopped),
        FakeTicker({"Ex-Dividend Date": announced}),
        TODAY + timedelta(days=200),
    )

    assert schedule[0].ex_date == announced
    assert schedule[0].estimated is False


def test_schedule_empty_when_yahoo_says_dividend_is_gone() -> None:
    """Отмену видно сразу по прогнозу, не дожидаясь пропущенного квартала.

    Выплаты шли до прошлой недели, тишины ещё нет — но dividendRate уже
    пуст, значит компания объявила об отмене.
    """
    schedule = market_data._dividend_schedule(
        history_with(quarterly(4, last_days_ago=7)),
        FakeTicker(info=CANCELLED),
        TODAY + timedelta(days=200),
    )

    assert schedule == []


@pytest.mark.parametrize(
    "info",
    [None, {}, {"dividendRate": 3.4}],
    ids=["info недоступен", "поля нет", "прогноз есть"],
)
def test_unknown_forecast_does_not_cancel_dividend(info) -> None:
    """Молчание Yahoo — это «не знаю», а не «отменили»."""
    schedule = market_data._dividend_schedule(
        history_with(quarterly(4)), FakeTicker(info=info), TODAY + timedelta(days=200)
    )

    assert schedule


def test_broken_info_does_not_cancel_dividend() -> None:
    """Упавший запрос не должен обнулять дивиденд живой компании."""
    assert market_data._dividend_cancelled(FakeTicker(broken=True)) is False


def test_forecast_not_requested_when_calendar_knows_the_date() -> None:
    """Календарь назвал отсечку — лишний запрос в сеть незачем."""

    class Watched(FakeTicker):
        asked = False

        @property
        def info(self):
            Watched.asked = True
            return PAYING

    market_data._dividend_schedule(
        history_with(quarterly(4)),
        Watched({"Ex-Dividend Date": TODAY + timedelta(days=11)}),
        TODAY + timedelta(days=200),
    )

    assert Watched.asked is False


def test_schedule_steps_over_missed_payments() -> None:
    """Отсечка пропущена и календарь молчит — шагаем до первой будущей."""
    stale = [TODAY - timedelta(days=200), TODAY - timedelta(days=109)]
    schedule = market_data._dividend_schedule(
        history_with(stale), FakeTicker(), TODAY + timedelta(days=100)
    )

    assert schedule[0].ex_date > TODAY


def test_schedule_stops_at_horizon() -> None:
    horizon = TODAY + timedelta(days=200)
    schedule = market_data._dividend_schedule(
        history_with(quarterly(4)), FakeTicker(), horizon
    )

    assert schedule
    assert all(payment.ex_date <= horizon for payment in schedule)


def test_schedule_keeps_last_amount() -> None:
    """Компании повышают дивиденд раз в год и держат новый уровень."""
    dates = quarterly(3)
    index = pd.to_datetime([str(value) for value in dates])
    frame = pd.DataFrame({"Dividends": [0.48, 0.51, 0.53]}, index=index)

    schedule = market_data._dividend_schedule(
        frame, FakeTicker(), TODAY + timedelta(days=200)
    )

    assert all(payment.amount == 0.53 for payment in schedule)


# --- эффективное q ---------------------------------------------------


def payment(days_ahead: int, amount: float = AMOUNT) -> DividendPayment:
    return DividendPayment(
        ex_date=TODAY + timedelta(days=days_ahead), amount=amount, estimated=False
    )


def test_yield_is_zero_before_first_ex_date() -> None:
    """Ровно та ошибка, ради которой всё делалось: до отсечки q равно нулю."""
    schedule = [payment(30)]
    assert market_data.effective_div_yield(
        schedule, SPOT, RATE, TODAY + timedelta(days=20)
    ) == 0.0


def test_yield_positive_when_ex_date_inside() -> None:
    schedule = [payment(10)]
    value = market_data.effective_div_yield(
        schedule, SPOT, RATE, TODAY + timedelta(days=20)
    )
    assert value > 0


def test_yield_reproduces_price_drop_exactly() -> None:
    """Смысл пересчёта: S * exp(-q * T) должно дать ровно S - D.

    Для европейского опциона дивиденды входят в цену только этим
    множителем, поэтому замена дискретных выплат на ставку q — не
    приближение, а тождество.
    """
    days = 40
    expiry = TODAY + timedelta(days=days)
    schedule = [payment(10), payment(35)]

    present_value = market_data._dividends_within(schedule, RATE, expiry)
    q = market_data.effective_div_yield(schedule, SPOT, RATE, expiry)

    tau = days / market_data.CALENDAR_DAYS
    assert SPOT * math.exp(-q * tau) == pytest.approx(SPOT - present_value)


def test_yield_falls_as_expiry_moves_away() -> None:
    """Одна и та же выплата, размазанная на больший срок, даёт ставку ниже."""
    schedule = [payment(10)]
    values = [
        market_data.effective_div_yield(
            schedule, SPOT, RATE, TODAY + timedelta(days=days)
        )
        for days in (20, 60, 180)
    ]
    assert values[0] > values[1] > values[2]


def test_yield_jumps_across_the_ex_date() -> None:
    """Экспирации в неделю друг от друга, а q различается в разы."""
    schedule = [payment(11, amount=0.53)]

    before = market_data.effective_div_yield(
        schedule, 88.0, RATE, TODAY + timedelta(days=7)
    )
    after = market_data.effective_div_yield(
        schedule, 88.0, RATE, TODAY + timedelta(days=14)
    )

    assert before == 0.0
    assert after == pytest.approx(0.157, abs=0.01)


def test_yield_approaches_annual_rate_on_long_terms() -> None:
    """На годовом сроке четыре отсечки усредняются — расхождения нет."""
    schedule = [payment(10 + 91 * step) for step in range(4)]

    value = market_data.effective_div_yield(
        schedule, SPOT, RATE, TODAY + timedelta(days=365)
    )
    # Четыре выплаты по 1 на бумагу за 100 — примерно 4% годовых,
    # минус дисконт на то, что деньги приходят не сегодня.
    assert value == pytest.approx(0.04, abs=0.002)


def test_ex_date_on_expiry_day_counts() -> None:
    """Цена падает на открытии дня отсечки, то есть до расчёта по опциону."""
    expiry = TODAY + timedelta(days=20)
    schedule = [payment(20)]

    assert market_data.effective_div_yield(schedule, SPOT, RATE, expiry) > 0


@pytest.mark.parametrize("days", [0, -5])
def test_yield_zero_on_expired_option(days: int) -> None:
    schedule = [payment(-1)]
    assert market_data.effective_div_yield(
        schedule, SPOT, RATE, TODAY + timedelta(days=days)
    ) == 0.0


def test_yield_zero_when_payments_exceed_price() -> None:
    """Битые данные: логарифм не определён, сорванный снимок хуже неточного q."""
    schedule = [payment(10, amount=SPOT * 2)]
    assert market_data.effective_div_yield(
        schedule, SPOT, RATE, TODAY + timedelta(days=20)
    ) == 0.0


# --- зачем всё это ---------------------------------------------------


def test_effective_yield_changes_the_price() -> None:
    """Ради чего затевалось: на ближней экспирации цена заметно другая.

    Двухнедельный колл на бумагу вроде KO: отсечка 0.53 попадает внутрь
    срока. Со среднегодовым q её вес размазан по всему году, с
    эффективным — учтён целиком, и колл дешевеет.
    """
    days = 14
    spot, rate = 88.0, 0.04
    schedule = [payment(11, amount=0.53)]

    trailing = 0.0236
    effective = market_data.effective_div_yield(
        schedule, spot, rate, TODAY + timedelta(days=days)
    )

    def price(q: float) -> float:
        return black_scholes.price(
            OptionParams(
                spot=spot, strike=spot, time_to_expiry=days / 365,
                rate=rate, div_yield=q, sigma=0.20,
                option_type=OptionType.CALL,
            )
        )

    assert effective > trailing
    assert price(effective) < price(trailing)
    # Порядок величины расхождения — проценты премии, а не доли процента.
    assert abs(price(effective) / price(trailing) - 1) > 0.05


def test_short_dated_yield_stays_inside_the_contract() -> None:
    """Большое q должно проходить проверки, а не отвергаться как опечатка."""
    schedule = [payment(1, amount=0.53)]
    value = market_data.effective_div_yield(
        schedule, 88.0, RATE, TODAY + timedelta(days=2)
    )

    assert value > 1.0
    assert value <= MAX_DIV_YIELD
    OptionParams(
        spot=88.0, strike=88.0, time_to_expiry=2 / 365,
        rate=RATE, div_yield=value, sigma=0.20, option_type=OptionType.CALL,
    )
