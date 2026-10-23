"""Блэк-Шоулз: проверяем цену против известных значений, аналитические греки
против численных и поведение в вырожденных случаях.
"""

from dataclasses import replace
from math import exp

import pytest

from app.pricing import black_scholes as bs
from app.pricing import numeric_greeks
from app.pricing.models import AmericanNotSupported, ExerciseStyle, OptionParams


def test_known_value(atm_call: OptionParams) -> None:
    # Значение из учебника Халла для S=K=100, T=1, r=5%, q=2%, sigma=20%.
    assert bs.price(atm_call) == pytest.approx(9.227006, abs=1e-6)


def test_put_call_parity(atm_call: OptionParams, atm_put: OptionParams) -> None:
    # C - P = S*exp(-qT) - K*exp(-rT). Паритет не зависит от модели: он
    # следует из отсутствия арбитража 
    tau = atm_call.time_to_expiry
    expected = atm_call.spot * exp(-atm_call.div_yield * tau) - atm_call.strike * exp(
        -atm_call.rate * tau
    )
    assert bs.price(atm_call) - bs.price(atm_put) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("greek", ["delta", "gamma", "vega", "theta", "rho"])
def test_analytic_greeks_match_numeric(atm_call: OptionParams, greek: str) -> None:

    analytic = getattr(bs.greeks(atm_call), greek)
    numeric = getattr(numeric_greeks, greek)(bs.price, atm_call)
    assert analytic == pytest.approx(numeric, rel=1e-4)


def test_at_expiry_equals_intrinsic(atm_call: OptionParams) -> None:
    expired = replace(atm_call, time_to_expiry=0.0, spot=120.0)
    assert bs.price(expired) == pytest.approx(20.0)


def test_zero_volatility_is_discounted_forward(atm_call: OptionParams) -> None:
    # При sigma = 0 цена детерминирована, случайности нет.
    certain = replace(atm_call, sigma=0.0, spot=120.0)
    forward = 120.0 * exp((certain.rate - certain.div_yield) * certain.time_to_expiry)
    expected = exp(-certain.rate * certain.time_to_expiry) * (forward - certain.strike)
    assert bs.price(certain) == pytest.approx(expected)


def test_american_is_rejected(atm_call: OptionParams) -> None:
    american = replace(atm_call, exercise=ExerciseStyle.AMERICAN)
    with pytest.raises(AmericanNotSupported):
        bs.price(american)


def test_deeper_in_the_money_is_more_expensive(atm_call: OptionParams) -> None:
    cheap = bs.price(replace(atm_call, strike=120.0))
    expensive = bs.price(replace(atm_call, strike=80.0))
    assert expensive > cheap
