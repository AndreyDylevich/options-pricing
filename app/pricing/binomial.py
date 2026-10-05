"""Биномиальное дерево Кокса-Росса-Рубинштейна: считаем цену обратным проходом
от выплаты в экспирации к текущему моменту. Единственная модель в проекте,
умеющая американский стиль — на каждом шаге проверяет, не выгодно ли
исполнить опцион досрочно.
"""

import time

import numpy as np

from app.pricing import bounds, numeric_greeks
from app.pricing.models import (
    ExerciseStyle,
    Greeks,
    OptionParams,
    PricingModel,
    PricingResult,
)

DEFAULT_STEPS = 500

# Один шаг — это одно движение вверх или вниз. Меньше одного дерева не
# бывает, а выше пяти тысяч расчёт упирается во время: греки требуют пяти
# построений, и каждое проходит все слои.
MIN_STEPS = 1
MAX_STEPS = 5_000

DAYS_IN_YEAR = 365.0

# Слои, которые нужно сохранить для греков: корень, первый и второй шаг.
GREEK_LAYERS = 2


# Дерево с такими параметрами не имеет смысла.
class InvalidTreeParameters(Exception):
    pass


def calculate(params: OptionParams, steps: int = DEFAULT_STEPS) -> PricingResult:
    # Всего пять построений дерева: одно основное и по два на vega и rho.
    # Delta, gamma и theta пересчёта не требуют — они снимаются с узлов,
    # которые обратный проход и так посчитал.
    started = time.perf_counter()

    tree = _run_tree(params, steps)

    def price_fn(other: OptionParams) -> float:
        return _run_tree(other, steps).price

    result_greeks = Greeks(
        delta=tree.delta if tree.delta is not None
        else numeric_greeks.delta(price_fn, params),
        gamma=tree.gamma if tree.gamma is not None
        else numeric_greeks.gamma(price_fn, params),
        vega=numeric_greeks.vega(price_fn, params),
        theta=tree.theta if tree.theta is not None
        else numeric_greeks.theta(price_fn, params),
        rho=numeric_greeks.rho(price_fn, params),
    )

    bounds.require_within_bounds(params, tree.price, "биномиальное дерево")

    elapsed = (time.perf_counter() - started) * 1000

    return PricingResult(
        price=tree.price,
        greeks=result_greeks,
        model=PricingModel.BINOMIAL,
        elapsed_ms=elapsed,
    )


def price(params: OptionParams, steps: int = DEFAULT_STEPS) -> float:
    return _run_tree(params, steps).price


# Цена в корне и греки, снятые с узлов. delta, gamma и theta равны None,
# если шагов меньше двух: узлов для разностей просто нет.
class _TreeResult:
    __slots__ = ("delta", "gamma", "price", "theta")

    def __init__(
        self,
        price: float,
        delta: float | None = None,
        gamma: float | None = None,
        theta: float | None = None,
    ) -> None:
        self.price = price
        self.delta = delta
        self.gamma = gamma
        self.theta = theta


def _run_tree(params: OptionParams, steps: int) -> _TreeResult:
    if steps < MIN_STEPS:
        raise InvalidTreeParameters(
            f"Число шагов должно быть не меньше {MIN_STEPS}"
        )
    if steps > MAX_STEPS:
        raise InvalidTreeParameters(
            f"Число шагов не больше {MAX_STEPS}: точность растёт как 1/steps, "
            f"а время — линейно, и дальше выигрыш не окупается"
        )

    if params.time_to_expiry <= 0:
        return _TreeResult(params.intrinsic_value)

    # За один шаг цена либо растёт в up раз, либо падает в down раз:
    #   u = exp(sigma * sqrt(dt)),  d = 1/u
    #   p = (exp((r - q) * dt) - d) / (u - d)
    #
    # p — вероятность риск-нейтральная, а не реальная: она подобрана так,
    # чтобы дисконтированная цена актива была мартингалом.
    dt = params.time_to_expiry / steps
    up = np.exp(params.sigma * np.sqrt(dt))
    down = 1 / up
    discount = np.exp(-params.rate * dt)
    prob_up = (np.exp((params.rate - params.div_yield) * dt) - down) / (up - down)

    _check_probability(prob_up, steps, params)

    american = params.exercise is ExerciseStyle.AMERICAN

    # Последний слой: выплата в момент экспирации.
    values = _payoff(params, _spot_layer(params.spot, up, steps))

    # Дерево не материализуется целиком: при обратном проходе нужен только
    # текущий слой, поэтому память O(steps), а не O(steps**2). Храним лишь
    # первые три слоя — с них снимаются греки.
    layers: dict[int, np.ndarray] = {}

    for step in range(steps - 1, -1, -1):
        # Ожидание по двум потомкам, приведённое к текущему моменту.
        values = discount * (prob_up * values[1:] + (1 - prob_up) * values[:-1])

        if american:
            # Право исполнить прямо сейчас: владелец выберет большее.
            values = np.maximum(values, _payoff(params, _spot_layer(params.spot, up, step)))

        if step <= GREEK_LAYERS:
            layers[step] = values

    root = float(layers[0][0])

    if steps < 2:
        return _TreeResult(root)

    return _TreeResult(
        price=root,
        delta=_delta_from_tree(layers[1], params.spot, up, down),
        gamma=_gamma_from_tree(layers[2], params.spot, up, down),
        theta=_theta_from_tree(layers[2], root, dt),
    )


# Цены акции во всех узлах слоя. В узле j было j движений вверх и step - j
# вниз, а так как d = 1/u, множитель равен u**(j - (step - j)) = u**(2j - step).
def _spot_layer(spot: float, up: float, step: int) -> np.ndarray:
    exponents = 2 * np.arange(step + 1) - step
    return spot * up**exponents


# Выплата при немедленном исполнении в каждом узле слоя.
def _payoff(params: OptionParams, spot: np.ndarray) -> np.ndarray:
    if params.is_call:
        return np.maximum(spot - params.strike, 0.0)
    return np.maximum(params.strike - spot, 0.0)


def _delta_from_tree(
    layer: np.ndarray, spot: float, up: float, down: float
) -> float:
    # Разность цен опциона на разность цен акции на первом шаге.
    return float((layer[1] - layer[0]) / (spot * up - spot * down))


def _gamma_from_tree(
    layer: np.ndarray, spot: float, up: float, down: float
) -> float:
    # Вторая разность по трём узлам второго шага. Средний из них вернулся в
    # исходную цену, потому что u * d = 1.
    spot_up, spot_mid, spot_down = spot * up**2, spot, spot * down**2

    delta_up = (layer[2] - layer[1]) / (spot_up - spot_mid)
    delta_down = (layer[1] - layer[0]) / (spot_mid - spot_down)

    return float((delta_up - delta_down) / ((spot_up - spot_down) / 2))


def _theta_from_tree(layer: np.ndarray, root: float, dt: float) -> float:
    # Средний узел второго шага стоит на той же цене акции, что и корень, но
    # до экспирации ему остаётся на 2 * dt меньше. Разность между ними и
    # есть распад во времени, без всякого пересчёта.
    per_year = (layer[1] - root) / (2 * dt)
    return float(per_year / DAYS_IN_YEAR)


# p вне [0, 1] — это уже не вероятность. Случается, когда снос за шаг
# (r - q) * dt выходит за пределы, достижимые движениями вверх и вниз, то
# есть примерно за 2 * sigma * sqrt(dt). Отсюда и оценка нужного числа шагов
# ниже: условие сводится к dt < (2 * sigma / |r - q|)**2.
def _check_probability(
    prob_up: float, steps: int, params: OptionParams
) -> None:
    if 0 <= prob_up <= 1:
        return

    drift = abs(params.rate - params.div_yield)
    if drift > 0 and params.sigma > 0:
        needed = int(params.time_to_expiry * (drift / (2 * params.sigma)) ** 2) + 1
    else:
        needed = 0

    if needed and needed <= MAX_STEPS:
        advice = f"увеличьте число шагов примерно до {needed} (сейчас {steps})"
    else:
        # Сказать «увеличьте шаги» здесь было бы бесполезным советом:
        # при такой волатильности требуемое число выходит за MAX_STEPS.
        advice = (
            f"при волатильности {params.sigma:.4%} решётке потребовалось бы "
            f"больше {MAX_STEPS} шагов — возьмите Блэка-Шоулза"
        )

    raise InvalidTreeParameters(
        f"Риск-нейтральная вероятность вышла за пределы [0, 1]: {prob_up:.4f}. "
        f"Шаг по времени слишком велик при такой волатильности: {advice}"
    )
