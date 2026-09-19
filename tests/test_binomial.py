"""Биномиальное дерево: сходимость к Блэку-Шоулзу и раннее исполнение."""

from dataclasses import replace

import pytest

from app.pricing import binomial as bt
from app.pricing import black_scholes as bs
from app.pricing.models import ExerciseStyle, OptionParams, OptionType


def test_converges_to_black_scholes(atm_call: OptionParams) -> None:
    assert bt.price(atm_call, steps=2000) == pytest.approx(bs.price(atm_call), abs=2e-3)


def test_error_halves_when_steps_double(atm_call: OptionParams) -> None:
    """Ошибка CRR убывает как 1/steps.

    Проверяем именно порядок сходимости, а не абсолютное значение: он
    ломается при неверных u, d или p, даже если цена на глаз похожа.
    """
    exact = bs.price(atm_call)
    coarse = abs(bt.price(atm_call, steps=500) - exact)
    fine = abs(bt.price(atm_call, steps=1000) - exact)
    assert fine == pytest.approx(coarse / 2, rel=0.15)


def test_american_call_without_dividends_equals_european(
    atm_call: OptionParams,
) -> None:
    """Досрочно исполнять колл на бумагу без дивидендов невыгодно никогда.

    Строгое равенство: ветка раннего исполнения не должна ничего менять
    там, где права на него не пользуются.
    """
    european = replace(atm_call, div_yield=0.0)
    american = replace(european, exercise=ExerciseStyle.AMERICAN)
    assert bt.price(american, steps=500) == pytest.approx(
        bt.price(european, steps=500), abs=1e-12
    )


def test_american_put_costs_more(atm_put: OptionParams) -> None:
    european = bt.price(atm_put, steps=1000)
    american = bt.price(replace(atm_put, exercise=ExerciseStyle.AMERICAN), steps=1000)
    assert american > european


def test_american_never_below_intrinsic() -> None:
    deep = OptionParams(
        spot=60.0, strike=100.0, time_to_expiry=1.0, rate=0.05, div_yield=0.0,
        sigma=0.20, option_type=OptionType.PUT, exercise=ExerciseStyle.AMERICAN,
    )
    assert bt.price(deep, steps=500) >= deep.intrinsic_value


@pytest.mark.parametrize("steps", [0, -1, bt.MAX_STEPS + 1])
def test_steps_outside_limits_are_rejected(atm_call: OptionParams, steps: int) -> None:
    with pytest.raises(bt.InvalidTreeParameters):
        bt.price(atm_call, steps=steps)


def test_probability_out_of_range_is_reported(atm_call: OptionParams) -> None:
    """При крошечной волатильности снос за шаг не покрывается движениями."""
    with pytest.raises(bt.InvalidTreeParameters, match="вероятность"):
        bt.price(replace(atm_call, sigma=1e-4), steps=500)


def test_single_step_still_prices(atm_call: OptionParams) -> None:
    """Одношаговое дерево — грубо, но не должно падать."""
    result = bt.calculate(atm_call, steps=1)
    assert result.price > 0
