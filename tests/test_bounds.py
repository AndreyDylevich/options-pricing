"""Границы отсутствия арбитража: проверяем, что все три модели на любых
разумных параметрах дают цену внутри них. Единственный тест, который ловит
ошибку в самой формуле, а не во входных данных.
"""

import pytest

from app.pricing import binomial as bt
from app.pricing import black_scholes as bs
from app.pricing import bounds
from app.pricing import monte_carlo as mc
from app.pricing.models import ExerciseStyle, OptionParams

from .conftest import PARAMETER_GRID

EUROPEAN_GRID = [p for p in PARAMETER_GRID if p.exercise is ExerciseStyle.EUROPEAN]


@pytest.mark.parametrize("params", EUROPEAN_GRID, ids=str)
def test_black_scholes_within_bounds(params: OptionParams) -> None:
    assert bounds.violation(params, bs.price(params)) is None


@pytest.mark.parametrize("params", PARAMETER_GRID, ids=str)
def test_binomial_within_bounds(params: OptionParams) -> None:
    try:
        value = bt.price(params, steps=200)
    except bt.InvalidTreeParameters:
        # Решётка честно отказалась считать — это не нарушение границ.
        pytest.skip("параметры вне применимости дерева")
    assert bounds.violation(params, value) is None


@pytest.mark.parametrize("params", EUROPEAN_GRID, ids=str)
def test_monte_carlo_within_bounds_or_warns(params: OptionParams) -> None:
    # Монте-Карло вправе выйти за границы из-за шума, но обязан сказать.
    result = mc.calculate(params, paths=20_000)
    if bounds.violation(params, result.price) is not None:
        assert result.warning is not None


def test_bounds_are_tight_for_deep_in_the_money(atm_call: OptionParams) -> None:
    # Глубоко в деньгах цена прижимается к нижней границе.
    from dataclasses import replace

    # При страйке в сто раз ниже цены исполнение практически гарантировано,
    # временной стоимости не остаётся и цена совпадает с полом.
    deep = replace(atm_call, strike=1.0)
    lower, upper = bounds.price_bounds(deep)
    assert lower <= bs.price(deep) <= upper
    assert bs.price(deep) == pytest.approx(lower, rel=1e-9)


def test_violation_is_detected() -> None:
    # Сама проверка должна срабатывать, а не молчать всегда.
    from app.pricing.models import OptionType

    params = OptionParams(
        spot=100.0, strike=100.0, time_to_expiry=1.0, rate=0.05,
        div_yield=0.02, sigma=0.20, option_type=OptionType.CALL,
    )
    assert bounds.violation(params, 1000.0) is not None
    assert bounds.violation(params, -1.0) is not None
