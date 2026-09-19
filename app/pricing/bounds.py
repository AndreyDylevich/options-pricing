"""Границы отсутствия арбитража.

Проверка не входных данных, а результата. Никакие параметры не могут
привести к цене выше потолка или ниже пола: если это случилось, ошибка не
у пользователя, а в самой модели.

Это единственная проверка во всём проекте, которая ловит опечатку в
формуле, а не опечатку в форме.

Европейский опцион:
    колл:  max(S*exp(-qT) - K*exp(-rT), 0)  <=  C  <=  S*exp(-qT)
    пут:   max(K*exp(-rT) - S*exp(-qT), 0)  <=  P  <=  K*exp(-rT)

Американский стоит не меньше немедленного исполнения и не больше самого
актива (колл) или страйка (пут): дисконтировать нечего, исполнить можно
хоть сейчас.
"""

from math import exp

from app.pricing.models import ExerciseStyle, OptionParams, PriceOutOfBounds

# Допуск на арифметику с плавающей точкой: относительный к масштабу цены
# актива плюс абсолютный на случай нулевых цен.
RELATIVE_TOLERANCE = 1e-9
ABSOLUTE_TOLERANCE = 1e-9


def price_bounds(params: OptionParams) -> tuple[float, float]:
    """Нижняя и верхняя границы цены для этих параметров."""
    tau = params.time_to_expiry
    spot_pv = params.spot * exp(-params.div_yield * tau)
    strike_pv = params.strike * exp(-params.rate * tau)

    if params.is_call:
        lower, upper = max(spot_pv - strike_pv, 0.0), spot_pv
    else:
        lower, upper = max(strike_pv - spot_pv, 0.0), strike_pv

    if params.exercise is ExerciseStyle.AMERICAN:
        # Право исполнить сейчас поднимает пол до внутренней стоимости и
        # снимает дисконтирование с потолка.
        lower = max(lower, params.intrinsic_value)
        upper = params.spot if params.is_call else params.strike

    return lower, upper


def violation(params: OptionParams, price: float) -> str | None:
    """Описание нарушения или None, если цена в границах."""
    lower, upper = price_bounds(params)
    tolerance = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * params.spot

    if price < lower - tolerance:
        return (
            f"цена {price:.6g} ниже границы отсутствия арбитража {lower:.6g}"
        )
    if price > upper + tolerance:
        return (
            f"цена {price:.6g} выше границы отсутствия арбитража {upper:.6g}"
        )
    return None


def require_within_bounds(params: OptionParams, price: float, model: str) -> None:
    """Бросить исключение, если цена вне границ.

    Для аналитических и решёточных методов нарушение означает ошибку в
    коде, поэтому останавливаемся. Монте-Карло вызывает не это, а
    violation(): у него отклонение может быть следствием случайного шума,
    и честнее предупредить, чем упасть.
    """
    problem = violation(params, price)
    if problem is not None:
        raise PriceOutOfBounds(f"{model}: {problem}")
