"""Греки численно: считаем цену при сдвинутом параметре и делим разность на
сдвиг. Работает с любой функцией цены, поэтому этим пользуются дерево,
Монте-Карло и тесты — как эталоном для проверки аналитических формул.
"""

from collections.abc import Callable
from dataclasses import replace

from app.pricing.models import MAX_RATE, MAX_SIGMA, MIN_RATE, Greeks, OptionParams

PERCENTAGE_POINT = 100.0
DAYS_IN_YEAR = 365.0

# Размеры сдвигов. По цене — относительный, чтобы одинаково работать и на
# бумаге за 87, и на бумаге за 3000. По остальным — абсолютные: ставка и
# волатильность и так измеряются в сопоставимых величинах.
SPOT_BUMP_FRACTION = 1e-4
GAMMA_BUMP_FRACTION = 1e-3
SIGMA_BUMP = 1e-4
RATE_BUMP = 1e-4
TIME_BUMP = 1 / DAYS_IN_YEAR

PriceFunction = Callable[[OptionParams], float]


def numeric_greeks(
    price_fn: PriceFunction,
    params: OptionParams,
    gamma_step_fraction: float = GAMMA_BUMP_FRACTION,
) -> Greeks:
    return Greeks(
        delta=delta(price_fn, params),
        gamma=gamma(price_fn, params, gamma_step_fraction),
        vega=vega(price_fn, params),
        theta=theta(price_fn, params),
        rho=rho(price_fn, params),
    )


def _difference(
    price_fn: PriceFunction,
    params: OptionParams,
    field: str,
    step: float,
    low: float | None = None,
    high: float | None = None,
) -> float:
    # Везде центральная разность: погрешность порядка h**2 против h у
    # односторонней, а стоит всего один лишний вызов.
    #
    # Сдвиг зажимается в допустимый диапазон, и делим на фактическое
    # расстояние между точками, а не на предполагаемое 2 * step. Иначе
    # параметр, стоящий ровно на границе (ставка -50%, волатильность
    # 500%), выходил бы за неё, и валидация OptionParams роняла бы расчёт
    # при том что схема такие значения разрешает. У границы разность
    # становится односторонней: точность падает с h**2 до h, но результат
    # остаётся.
    value = getattr(params, field)

    upper = value + step
    lower = value - step
    if high is not None:
        upper = min(upper, high)
    if low is not None:
        lower = max(lower, low)

    span = upper - lower
    if span <= 0:
        return 0.0

    up = price_fn(replace(params, **{field: upper}))
    down = price_fn(replace(params, **{field: lower}))
    return (up - down) / span


def delta(price_fn: PriceFunction, params: OptionParams) -> float:
    # Цена базового актива положительна, а сдвиг относительный, поэтому за
    # ноль он не уйдёт: нижняя граница здесь чисто формальная.
    return _difference(
        price_fn, params, "spot", params.spot * SPOT_BUMP_FRACTION, low=0.0
    )


def gamma(
    price_fn: PriceFunction,
    params: OptionParams,
    step_fraction: float = GAMMA_BUMP_FRACTION,
) -> float:
    # Шаг крупнее, чем у delta: вторая разность делится на квадрат шага, и
    # на мелком шаге разность двух близких чисел тонет в погрешности
    # округления.
    #
    # Монте-Карло просит шаг ещё крупнее и передаёт свой: выплата опциона
    # ломается в одной точке, и вторую разность формируют только те пути,
    # чья финальная цена попала в окрестность страйка шириной 2 * step. На
    # остальных три пробные точки лежат на прямой, и вклад в gamma ровно
    # нулевой. При сдвиге 0.1% из миллиона путей работают около четырёх
    # тысяч, при 2% — восемьдесят тысяч, и оценка перестаёт скакать.
    step = params.spot * step_fraction
    up = price_fn(replace(params, spot=params.spot + step))
    mid = price_fn(params)
    down = price_fn(replace(params, spot=params.spot - step))
    return (up - 2 * mid + down) / step**2


def vega(price_fn: PriceFunction, params: OptionParams) -> float:
    derivative = _difference(
        price_fn, params, "sigma", SIGMA_BUMP, low=0.0, high=MAX_SIGMA
    )
    return derivative / PERCENTAGE_POINT


def rho(price_fn: PriceFunction, params: OptionParams) -> float:
    derivative = _difference(
        price_fn, params, "rate", RATE_BUMP, low=MIN_RATE, high=MAX_RATE
    )
    return derivative / PERCENTAGE_POINT


def theta(price_fn: PriceFunction, params: OptionParams) -> float:
    # theta = -dV/dT: время до экспирации уменьшается, поэтому знак
    # обратный. Шаг ограничен половиной оставшегося срока, чтобы у почти
    # истёкшего опциона не уйти в отрицательное T.
    step = min(TIME_BUMP, params.time_to_expiry / 2)
    if step <= 0:
        return 0.0

    derivative_per_year = _difference(
        price_fn, params, "time_to_expiry", step, low=0.0
    )
    return -derivative_per_year / DAYS_IN_YEAR
