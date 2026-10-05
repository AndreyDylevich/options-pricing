"""Обратная задача: проверяем, что из цены, собранной при известной σ,
восстанавливается ровно она, и что нерешаемые случаи объясняются внятно.
"""

from dataclasses import replace

import pytest

from app.pricing import binomial, bounds, implied_vol
from app.pricing import black_scholes as bs
from app.pricing.models import OptionParams, OptionType


@pytest.mark.parametrize("sigma", [0.05, 0.15, 0.30, 0.80, 2.0, 4.5])
@pytest.mark.parametrize("option_type", list(OptionType))
def test_round_trip(atm_call: OptionParams, sigma: float, option_type) -> None:
    # Цена по волатильности, затем волатильность по цене — то же число.
    params = replace(atm_call, option_type=option_type)
    price = bs.price(replace(params, sigma=sigma))

    result = implied_vol.solve(params, price)
    assert result.sigma == pytest.approx(sigma, abs=1e-6)


@pytest.mark.parametrize("strike", [50.0, 80.0, 100.0, 130.0, 200.0])
def test_round_trip_across_strikes(atm_call: OptionParams, strike: float) -> None:
    # Далеко от денег vega мала, но метод Брента не расходится.
    params = replace(atm_call, strike=strike)
    price = bs.price(replace(params, sigma=0.35))

    assert implied_vol.solve(params, price).sigma == pytest.approx(0.35, abs=1e-6)


def test_works_with_binomial_price(atm_call: OptionParams) -> None:
    # Функция цены — параметр: годится любая модель.
    price = binomial.price(replace(atm_call, sigma=0.4), steps=300)

    # Пол поиска поднят: при sigma порядка 0.01% решётка вырождается и
    # отказывается считать, а метод Брента обязан вычислить оба конца.
    result = implied_vol.solve(
        atm_call,
        price,
        price_fn=lambda p: binomial.price(p, steps=300),
        sigma_min=0.01,
    )
    assert result.sigma == pytest.approx(0.4, abs=1e-4)


def test_price_below_intrinsic_has_no_solution(atm_call: OptionParams) -> None:
    # Ровно так выглядят битые котировки неликвидных страйков.
    deep = replace(atm_call, strike=50.0)
    lower, _ = bounds.price_bounds(deep)

    result = implied_vol.solve(deep, lower - 2)
    assert result.sigma is None
    assert "ниже внутренней стоимости" in result.reason


def test_price_above_underlying_has_no_solution(atm_call: OptionParams) -> None:
    _, upper = bounds.price_bounds(atm_call)

    result = implied_vol.solve(atm_call, upper + 1)
    assert result.sigma is None
    assert "выше стоимости базового актива" in result.reason


def test_price_unreachable_even_at_max_sigma(atm_call: OptionParams) -> None:
    # Цена внутри границ, но требует волатильности выше потолка поиска.
    at_ceiling = bs.price(replace(atm_call, sigma=implied_vol.SIGMA_MAX))
    _, upper = bounds.price_bounds(atm_call)

    # Строго между потолком поиска и границей арбитража: иначе сработала бы
    # предварительная проверка и причина была бы другой.
    result = implied_vol.solve(atm_call, (at_ceiling + upper) / 2)
    assert result.sigma is None
    assert "недостижима" in result.reason


def test_expired_option_has_no_solution(atm_call: OptionParams) -> None:
    result = implied_vol.solve(replace(atm_call, time_to_expiry=0.0), 5.0)
    assert result.sigma is None
    assert "истёк" in result.reason


def test_zero_price_has_no_solution(atm_call: OptionParams) -> None:
    assert implied_vol.solve(atm_call, 0.0).sigma is None


def test_higher_price_means_higher_volatility(atm_call: OptionParams) -> None:
    # Монотонность — то, на чём держится единственность решения.
    cheap = implied_vol.solve(atm_call, bs.price(replace(atm_call, sigma=0.2)))
    rich = implied_vol.solve(atm_call, bs.price(replace(atm_call, sigma=0.6)))

    assert cheap.sigma < rich.sigma


def test_put_call_parity_gives_equal_volatility(atm_call: OptionParams) -> None:
    # Из паритета следует: колл и пут одного страйка дают одну IV.
    put = replace(atm_call, option_type=OptionType.PUT)
    sigma = 0.27

    from_call = implied_vol.solve(atm_call, bs.price(replace(atm_call, sigma=sigma)))
    from_put = implied_vol.solve(put, bs.price(replace(put, sigma=sigma)))

    assert from_call.sigma == pytest.approx(from_put.sigma, abs=1e-8)
