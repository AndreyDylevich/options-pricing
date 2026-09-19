"""Численные греки: поведение на границах допустимых диапазонов."""

from dataclasses import replace

import pytest

from app.pricing import black_scholes as bs
from app.pricing import numeric_greeks
from app.pricing.models import MAX_RATE, MAX_SIGMA, MIN_RATE, OptionParams


@pytest.mark.parametrize(
    "field,value",
    [("rate", MIN_RATE), ("rate", MAX_RATE), ("sigma", MAX_SIGMA)],
)
def test_bump_at_boundary_does_not_raise(
    atm_call: OptionParams, field: str, value: float
) -> None:
    """Параметр ровно на границе диапазона.

    Сдвиг для численной производной вышел бы за границу, и валидация
    OptionParams роняла бы расчёт — при том что схема такие значения
    разрешает. Сдвиг обязан зажиматься.
    """
    edge = replace(atm_call, **{field: value})
    greeks = numeric_greeks.numeric_greeks(bs.price, edge)

    for name in ("delta", "gamma", "vega", "theta", "rho"):
        assert getattr(greeks, name) == pytest.approx(getattr(greeks, name))


def test_clamped_bump_still_close_to_analytic(atm_call: OptionParams) -> None:
    """У границы разность односторонняя, но всё ещё осмысленная."""
    edge = replace(atm_call, rate=MIN_RATE)
    numeric = numeric_greeks.rho(bs.price, edge)
    analytic = bs.greeks(edge).rho
    assert numeric == pytest.approx(analytic, rel=1e-2)


def test_theta_near_expiry_does_not_go_negative_in_time(
    atm_call: OptionParams,
) -> None:
    """Шаг по времени ограничен половиной оставшегося срока."""
    almost_expired = replace(atm_call, time_to_expiry=1 / 365 / 24)
    assert numeric_greeks.theta(bs.price, almost_expired) < 0


def test_zero_span_returns_zero(atm_call: OptionParams) -> None:
    """Если сдвинуть параметр некуда, производная считается нулевой."""
    expired = replace(atm_call, time_to_expiry=0.0)
    assert numeric_greeks.theta(bs.price, expired) == 0.0
