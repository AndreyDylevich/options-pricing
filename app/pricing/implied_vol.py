"""Обратная задача: по рыночной цене опциона считаем волатильность, при
которой модель даёт ровно эту цену. Решаем методом Брента из scipy —
аналитической формулы не существует.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

from scipy.optimize import brentq

from app.pricing import black_scholes, bounds
from app.pricing.models import MAX_SIGMA, OptionParams

# Нижняя не ноль: при нулевой σ цена вырождается во внутреннюю стоимость.
SIGMA_MIN = 1e-4
SIGMA_MAX = MAX_SIGMA

PRICE_TOLERANCE = 1e-9

PriceFunction = Callable[[OptionParams], float]


@dataclass(frozen=True, slots=True)
class ImpliedVolatility:
        # None, если решения нет: у неликвидных страйков это норма, а не ошибка.
    sigma: float | None
    reason: str | None = None


def solve(
    params: OptionParams,
    market_price: float,
    price_fn: PriceFunction = black_scholes.price,
    sigma_min: float = SIGMA_MIN,
    sigma_max: float = SIGMA_MAX,
) -> ImpliedVolatility:
        # Дерево при σ порядка 0.01% вырождается — подставляя его, поднимайте sigma_min.
    if params.time_to_expiry <= 0:
        return ImpliedVolatility(None, "опцион истёк: временной стоимости нет")

    if market_price <= 0:
        return ImpliedVolatility(None, "нулевая цена")

    problem = _outside_bounds(params, market_price)
    if problem is not None:
        return ImpliedVolatility(None, problem)

    def difference(sigma: float) -> float:
        return price_fn(replace(params, sigma=sigma)) - market_price

        # Брент, а не Ньютон: тот делит на vega, а она далеко от денег почти нулевая.
    at_min = difference(sigma_min)
    at_max = difference(sigma_max)

    if at_min > 0:
        return ImpliedVolatility(
            None, "цена ниже достижимой даже при нулевой волатильности"
        )
    if at_max < 0:
        return ImpliedVolatility(
            None, f"цена недостижима даже при волатильности {sigma_max:.0%}"
        )

    sigma = brentq(difference, sigma_min, sigma_max, xtol=1e-10, maxiter=200)
    return ImpliedVolatility(float(sigma))


def _outside_bounds(params: OptionParams, market_price: float) -> str | None:
    lower, upper = bounds.price_bounds(params)
    tolerance = PRICE_TOLERANCE + PRICE_TOLERANCE * params.spot

    if market_price < lower - tolerance:
        return f"цена {market_price:.4g} ниже внутренней стоимости {lower:.4g}"
    if market_price > upper + tolerance:
        return f"цена {market_price:.4g} выше стоимости базового актива {upper:.4g}"
    return None
