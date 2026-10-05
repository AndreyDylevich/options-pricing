"""Обратная задача: по рыночной цене опциона считаем волатильность, при
которой модель даёт ровно эту цену. Решаем методом Брента из scipy —
аналитической формулы не существует.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

from scipy.optimize import brentq

from app.pricing import black_scholes, bounds
from app.pricing.models import MAX_SIGMA, OptionParams

# Границы поиска. Нижняя не ноль: при нулевой волатильности цена
# вырождается в дисконтированную внутреннюю стоимость, и решать нечего.
SIGMA_MIN = 1e-4
SIGMA_MAX = MAX_SIGMA

# Допуск на арифметику при сравнении с границами арбитража.
PRICE_TOLERANCE = 1e-9

PriceFunction = Callable[[OptionParams], float]


@dataclass(frozen=True, slots=True)
class ImpliedVolatility:
    # sigma равен None, если решения не существует. Это не ошибка: у
    # неликвидных страйков рыночная цена регулярно оказывается вне
    # достижимого моделью диапазона, и причина полезнее исключения.
    sigma: float | None
    reason: str | None = None


def solve(
    params: OptionParams,
    market_price: float,
    price_fn: PriceFunction = black_scholes.price,
    sigma_min: float = SIGMA_MIN,
    sigma_max: float = SIGMA_MAX,
) -> ImpliedVolatility:
    # Поле sigma у params игнорируется: оно как раз и подбирается.
    #
    # Границы поиска вынесены в параметры не для красоты: функция цены
    # обязана быть определена на обоих концах, а биномиальное дерево при
    # волатильности порядка 0.01% вырождается — снос за шаг перестаёт
    # покрываться движениями вверх и вниз. Подставляя дерево, поднимайте
    # sigma_min.
    if params.time_to_expiry <= 0:
        return ImpliedVolatility(None, "опцион истёк: временной стоимости нет")

    if market_price <= 0:
        return ImpliedVolatility(None, "нулевая цена")

    problem = _outside_bounds(params, market_price)
    if problem is not None:
        return ImpliedVolatility(None, problem)

    def difference(sigma: float) -> float:
        return price_fn(replace(params, sigma=sigma)) - market_price

    # Цена строго возрастает по волатильности, поэтому решение единственно
    # и ищется делением отрезка. Брент, а не Ньютон: тот делит на vega, а
    # она на далёких от денег страйках почти нулевая.
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


# Цена вне границ арбитража нерешаема в принципе. Отдельная проверка нужна
# ради внятной причины: без неё пользователь получил бы одинаковое «решения
# нет» и для явного мусора в котировках, и для цены, до которой модель
# просто не дотягивается.
def _outside_bounds(params: OptionParams, market_price: float) -> str | None:
    lower, upper = bounds.price_bounds(params)
    tolerance = PRICE_TOLERANCE + PRICE_TOLERANCE * params.spot

    if market_price < lower - tolerance:
        return f"цена {market_price:.4g} ниже внутренней стоимости {lower:.4g}"
    if market_price > upper + tolerance:
        return f"цена {market_price:.4g} выше стоимости базового актива {upper:.4g}"
    return None
