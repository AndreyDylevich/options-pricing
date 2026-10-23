"""Паритет колла и пута: собираем котировки, заведомо ему удовлетворяющие, и
требуем нулевого расхождения, а обратная задача обязана вернуть тот форвард,
из которого цены и собраны.
"""

import math

import pytest

from app.pricing import parity

SPOT = 100.0
RATE = 0.05
DIV_YIELD = 0.03
TAU = 0.25


def exact_pair(strike: float, call_mid: float, spread: float = 0.02):
    put_mid = call_mid - (
        SPOT * math.exp(-DIV_YIELD * TAU) - strike * math.exp(-RATE * TAU)
    )
    half = spread / 2
    return (
        parity.Quote(bid=call_mid - half, ask=call_mid + half),
        parity.Quote(bid=put_mid - half, ask=put_mid + half),
    )


def test_exact_parity_leaves_no_residual() -> None:
    # Котировки по паритету — расхождение ноль.
    call, put = exact_pair(100.0, call_mid=6.0)
    result = parity.check(100.0, call, put, SPOT, RATE, DIV_YIELD, TAU)

    assert result.residual == pytest.approx(0.0, abs=1e-12)
    assert result.within_quotes


def test_wrong_div_yield_shows_up_as_residual() -> None:
    # Ошибка в q — ровно то, ради чего сверка и делается. Рынок живёт при
    # q = 3%, мы считаем при 6%. Расхождение обязано вылезти и обязано
    # совпасть с разницей множителей exp(-qT).
    call, put = exact_pair(100.0, call_mid=6.0, spread=0.02)
    result = parity.check(100.0, call, put, SPOT, RATE, 0.06, TAU)

    expected_gap = SPOT * (math.exp(-DIV_YIELD * TAU) - math.exp(-0.06 * TAU))

    assert result.residual == pytest.approx(expected_gap, abs=1e-12)
    assert not result.within_quotes


def test_wide_spreads_forgive_the_same_error() -> None:
    # При широких котировках то же расхождение уже ни о чём не говорит.
    # Сверка обязана отличать «наши данные неверны» от «рынок сам не знает
    # цену с такой точностью».
    call, put = exact_pair(100.0, call_mid=6.0, spread=4.0)
    result = parity.check(100.0, call, put, SPOT, RATE, 0.06, TAU)

    assert result.within_quotes


def test_implied_forward_recovers_the_forward() -> None:
    # Обратная задача возвращает тот форвард, из которого цены собраны.
    call, put = exact_pair(95.0, call_mid=9.0)
    result = parity.check(95.0, call, put, SPOT, RATE, DIV_YIELD, TAU)

    forward = SPOT * math.exp((RATE - DIV_YIELD) * TAU)

    assert result.implied_forward == pytest.approx(forward, rel=1e-12)
    assert result.implied_div_yield == pytest.approx(DIV_YIELD, rel=1e-12)


def test_implied_forward_ignores_our_div_yield() -> None:
    # Форвард вытаскивается из цен, а не из наших предположений. Иначе
    # проверка опиралась бы на проверяемое: подставили своё q, получили
    # своё же q обратно.
    call, put = exact_pair(100.0, call_mid=6.0)

    honest = parity.check(100.0, call, put, SPOT, RATE, DIV_YIELD, TAU)
    wrong = parity.check(100.0, call, put, SPOT, RATE, 0.42, TAU)

    assert wrong.implied_forward == pytest.approx(honest.implied_forward, rel=1e-12)


@pytest.mark.parametrize("strike", [80.0, 95.0, 100.0, 105.0, 130.0])
def test_forward_is_the_same_on_every_strike(strike: float) -> None:
    # Форвард один на всю цепочку, страйк на него влиять не должен.
    call, put = exact_pair(strike, call_mid=max(SPOT - strike, 0.0) + 5.0)
    result = parity.check(strike, call, put, SPOT, RATE, DIV_YIELD, TAU)

    assert result.implied_forward == pytest.approx(
        SPOT * math.exp((RATE - DIV_YIELD) * TAU), rel=1e-12
    )


def test_consensus_takes_the_strikes_at_the_money() -> None:
    forward = SPOT * math.exp((RATE - DIV_YIELD) * TAU)
    checks = []

    for strike in (70.0, 85.0, 98.0, 99.0, 100.0, 101.0, 102.0, 120.0, 140.0):
        call, put = exact_pair(strike, call_mid=max(SPOT - strike, 0.0) + 5.0)
        if strike in (70.0, 140.0):
            # Битая котировка на краю цепочки — обычное дело.
            call = parity.Quote(bid=call.bid + 8.0, ask=call.ask + 9.0)
        checks.append(parity.check(strike, call, put, SPOT, RATE, DIV_YIELD, TAU))

    result = parity.consensus_forward(checks)

    assert result is not None
    assert result[0] == pytest.approx(forward, abs=1e-9)


def test_consensus_survives_one_bad_quote_among_the_nearest() -> None:
    # Медиана, а не среднее: один выброс у денег ответа не сдвигает.
    forward = SPOT * math.exp((RATE - DIV_YIELD) * TAU)
    checks = []

    for strike in (96.0, 98.0, 100.0, 102.0, 104.0):
        call, put = exact_pair(strike, call_mid=max(SPOT - strike, 0.0) + 5.0)
        if strike == 98.0:
            call = parity.Quote(bid=call.bid + 5.0, ask=call.ask + 5.0)
        checks.append(parity.check(strike, call, put, SPOT, RATE, DIV_YIELD, TAU))

    result = parity.consensus_forward(checks)

    assert result is not None
    assert result[0] == pytest.approx(forward, abs=1e-9)


def test_consensus_survives_a_stale_spot() -> None:
    # Съехавший спот выгоняет за границы всю цепочку — форвард обязан
    # уцелеть. Ровно это и случилось на NVDA: наш спот снят на закрытии,
    # котировки опционов свежее, и все 46 страйков разом оказались вне
    # границ. Если отбраковывать по этому признаку, пропадает единственное
    # число, которое такой разрыв и показывает, — рыночный форвард от
    # нашего спота не зависит вообще.
    forward = SPOT * math.exp((RATE - DIV_YIELD) * TAU)
    stale = SPOT + 3.0

    checks = []
    for strike in (90.0, 95.0, 100.0, 105.0, 110.0):
        call, put = exact_pair(strike, call_mid=max(SPOT - strike, 0.0) + 5.0)
        checks.append(parity.check(strike, call, put, stale, RATE, DIV_YIELD, TAU))

    assert all(item.arbitrage for item in checks), "спот должен ломать границы"

    result = parity.consensus_forward(checks)

    assert result is not None
    assert result[0] == pytest.approx(forward, abs=1e-9)


def test_consensus_returns_none_without_usable_strikes() -> None:
    assert parity.consensus_forward([]) is None


class TestArbitrage:
    # Границы, за которыми котировки не объясняются никакой моделью.

    def test_clean_quotes_report_nothing(self) -> None:
        call, put = exact_pair(100.0, call_mid=6.0)
        result = parity.check(100.0, call, put, SPOT, RATE, DIV_YIELD, TAU)

        assert result.arbitrage is None

    def test_call_too_expensive_breaks_the_upper_bound(self) -> None:
        # Колл дороже верхней границы: продай его, купи пут — и
        # заработал.
        call, put = exact_pair(100.0, call_mid=6.0)
        rich = parity.Quote(bid=call.bid + 10.0, ask=call.ask + 10.0)

        result = parity.check(100.0, rich, put, SPOT, RATE, DIV_YIELD, TAU)

        assert result.arbitrage is not None
        assert "верхней границе" in result.arbitrage

    def test_put_too_expensive_breaks_the_lower_bound(self) -> None:
        call, put = exact_pair(100.0, call_mid=6.0)
        rich = parity.Quote(bid=put.bid + 10.0, ask=put.ask + 10.0)

        result = parity.check(100.0, call, rich, SPOT, RATE, DIV_YIELD, TAU)

        assert result.arbitrage is not None
        assert "нижней границе" in result.arbitrage

    def test_bounds_are_checked_on_executable_prices(self) -> None:
        # Нарушение считается по бидам и аскам, а не по серединам.
        # Иначе «арбитраж» находился бы там, где после спреда
        # заработать уже нечего, — и отчёт заполнился бы ложными
        # тревогами.
        call, put = exact_pair(100.0, call_mid=6.0)
        # Верхняя граница по серединам чуть превышена, но купить колл
        # придётся по аску, и выгода съедается спредом.
        upper = SPOT - 100.0 * math.exp(-RATE * TAU)
        nudged = put.mid - (upper - (call.mid - put.mid)) - 0.05
        cheap_put = parity.Quote(bid=nudged - 0.5, ask=nudged + 0.5)

        result = parity.check(100.0, call, cheap_put, SPOT, RATE, DIV_YIELD, TAU)

        assert result.observed > upper
        assert result.arbitrage is None


class TestQuote:
    def test_mid_and_half_spread(self) -> None:
        quote = parity.Quote(bid=4.5, ask=5.5)

        assert quote.mid == 5.0
        assert quote.half_spread == 0.5

    @pytest.mark.parametrize(
        "bid, ask",
        [(0.0, 5.0), (4.0, 0.0), (-1.0, 5.0), (5.0, 4.0)],
        ids=["нет бида", "нет аска", "отрицательный бид", "аск ниже бида"],
    )
    def test_broken_quotes_are_refused(self, bid: float, ask: float) -> None:
        with pytest.raises(ValueError):
            parity.Quote(bid=bid, ask=ask)


class TestInputValidation:
    @pytest.mark.parametrize(
        "strike, spot, tau",
        [(0.0, SPOT, TAU), (100.0, 0.0, TAU), (100.0, SPOT, 0.0)],
        ids=["нулевой страйк", "нулевой спот", "нулевой срок"],
    )
    def test_degenerate_inputs_are_refused(
        self, strike: float, spot: float, tau: float
    ) -> None:
        call, put = exact_pair(100.0, call_mid=6.0)

        with pytest.raises(ValueError):
            parity.check(strike, call, put, spot, RATE, DIV_YIELD, tau)

    def test_implied_div_yield_undefined_at_zero_term(self) -> None:
        assert parity.implied_div_yield(100.0, SPOT, RATE, 0.0) is None
