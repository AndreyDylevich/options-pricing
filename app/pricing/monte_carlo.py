"""Монте-Карло по геометрическому броуновскому движению.

    S_T = S * exp((r - q - sigma**2 / 2) * T + sigma * sqrt(T) * Z)

Цена опциона — дисконтированное среднее выплаты в риск-нейтральном мире.
Блэк-Шоулз берёт это матожидание формулой, здесь оно оценивается выборкой.

Траектория по дням не строится: выплата европейского опциона зависит
только от финальной цены, а формула выше даёт её распределение точно.
Поэтому ошибки дискретизации нет вовсе — в отличие от дерева, вся
погрешность здесь статистическая, и она честно возвращается в std_error.

Снижение дисперсии — антитетические переменные: на каждое Z берётся ещё
и -Z. Пара отрицательно скоррелирована, и её среднее ближе к истине, чем
два независимых розыгрыша.

Греки считаются пересчётом на одном и том же массиве Z (common random
numbers). Без этого разность двух зашумлённых оценок дала бы мусор:
у ATM-опциона сдвиг цены на 0.01% меняет результат примерно на 0.005 при
случайной ошибке около 0.016.
"""

import time

import numpy as np

from app.pricing import bounds, numeric_greeks
from app.pricing.models import (
    AmericanNotSupported,
    ExerciseStyle,
    Greeks,
    OptionParams,
    PricingModel,
    PricingResult,
)

DEFAULT_PATHS = 100_000

# Меньше тысячи путей — стандартная ошибка становится сопоставима с самой
# ценой, и результат перестаёт что-либо значить.
MIN_PATHS = 1_000

# Пять миллионов путей — это уже около 40 МБ на массив, а numeric_greeks
# вызывает симуляцию 11 раз. Верхняя граница совпадает с той, что стоит в
# схеме запроса.
MAX_PATHS = 5_000_000

# Seed фиксирован намеренно: одни и те же параметры должны давать один и
# тот же ответ, иначе пользователь увидит, как цена дрожит между
# нажатиями, и не сможет сравнить два расчёта в истории.
SEED = 20240814

# 97.5-й перцентиль стандартного нормального распределения.
CONFIDENCE_Z = 1.959964

# Сдвиг цены для gamma — 2% против 0.1% по умолчанию. Подобран замером:
# на миллионе путей ошибка падает с 1.5% до 0.1%. Причина в том, что
# вторую разность формируют только пути, попавшие к излому выплаты, и
# узкий сдвиг оставляет от них считанные сотни.
GAMMA_BUMP_FRACTION = 2e-2

# Порог, за которым оценке нельзя верить: случайная ошибка больше 5% от
# самой цены. В нормальном режиме это доли процента, а при экстремальной
# волатильности доходит до 90% — и тогда результат отличается от точного
# в разы, хотя внешне выглядит обычным числом.
NOISE_WARNING_FRACTION = 0.05


class InvalidSimulationParameters(Exception):
    """Симуляция с такими параметрами не имеет смысла."""


def calculate(params: OptionParams, paths: int = DEFAULT_PATHS) -> PricingResult:
    """Цена, доверительный интервал и греки.

    Симуляция запускается 12 раз: одна на цену и 11 на численные греки.
    Все — на общем массиве Z.
    """
    _require_european(params)
    _check_paths(paths)

    started = time.perf_counter()

    if params.time_to_expiry <= 0:
        return _at_expiry(params, started)

    normals = _draw_normals(paths)

    price_value, std_error = _simulate(params, normals)

    def price_fn(other: OptionParams) -> float:
        return _simulate(other, normals)[0]

    result_greeks = numeric_greeks.numeric_greeks(
        price_fn, params, gamma_step_fraction=GAMMA_BUMP_FRACTION
    )
    elapsed = (time.perf_counter() - started) * 1000

    margin = CONFIDENCE_Z * std_error

    return PricingResult(
        price=price_value,
        greeks=result_greeks,
        model=PricingModel.MONTE_CARLO,
        elapsed_ms=elapsed,
        std_error=std_error,
        ci95=(price_value - margin, price_value + margin),
        warning=_quality_warning(params, price_value, std_error, paths),
    )


def price(params: OptionParams, paths: int = DEFAULT_PATHS) -> float:
    _require_european(params)
    _check_paths(paths)

    if params.time_to_expiry <= 0:
        return params.intrinsic_value

    return _simulate(params, _draw_normals(paths))[0]


def estimate(params: OptionParams, paths: int) -> tuple[float, float]:
    """Только цена и её стандартная ошибка, без греков.

    calculate() запускает симуляцию двенадцать раз — одну на цену и
    одиннадцать на численные греки. Для графика сходимости греки не нужны,
    а разница в стоимости двенадцатикратная.
    """
    _require_european(params)
    _check_paths(paths)

    if params.time_to_expiry <= 0:
        return params.intrinsic_value, 0.0

    return _simulate(params, _draw_normals(paths))


def _draw_normals(paths: int) -> np.ndarray:
    """Массив Z с антитетическими парами.

    Первая половина — розыгрыш, вторая — та же с обратным знаком. При
    нечётном числе путей одна выборка отбрасывается: пары должны быть
    полными, иначе оценка разъедется.
    """
    pairs = paths // 2
    generator = np.random.default_rng(SEED)
    half = generator.standard_normal(pairs)

    return np.concatenate([half, -half])


def _simulate(params: OptionParams, normals: np.ndarray) -> tuple[float, float]:
    """Цена и её стандартная ошибка на готовом массиве Z."""
    tau = params.time_to_expiry
    drift = (params.rate - params.div_yield - params.sigma**2 / 2) * tau
    diffusion = params.sigma * np.sqrt(tau)

    terminal = params.spot * np.exp(drift + diffusion * normals)

    if params.is_call:
        payoff = np.maximum(terminal - params.strike, 0.0)
    else:
        payoff = np.maximum(params.strike - terminal, 0.0)

    discounted = np.exp(-params.rate * tau) * payoff

    # Наблюдение — это пара (Z, -Z), а не отдельный путь: внутри пары
    # выборки зависимы, и делить на корень из общего числа путей нельзя,
    # ошибка получилась бы заниженной.
    pairs = len(normals) // 2
    pair_means = (discounted[:pairs] + discounted[pairs:]) / 2

    value = float(pair_means.mean())
    std_error = float(pair_means.std(ddof=1) / np.sqrt(pairs))

    return value, std_error


def _at_expiry(params: OptionParams, started: float) -> PricingResult:
    """T = 0: разыгрывать нечего.

    Опцион истекает прямо сейчас, его цена равна внутренней стоимости, а
    случайной ошибки нет — интервал вырождается в точку.
    """
    value = params.intrinsic_value
    in_the_money = value > 0
    delta = (1.0 if params.is_call else -1.0) if in_the_money else 0.0

    return PricingResult(
        price=value,
        greeks=Greeks(delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0),
        model=PricingModel.MONTE_CARLO,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        std_error=0.0,
        ci95=(value, value),
    )


def _quality_warning(
    params: OptionParams, value: float, std_error: float, paths: int
) -> str | None:
    """Насколько результату можно верить.

    В отличие от двух других моделей Монте-Карло не падает при плохих
    параметрах, а тихо возвращает неверное число. Зато он сам измеряет
    свою погрешность, и этим стоит воспользоваться.

    Границы отсутствия арбитража проверяем предупреждением, а не
    исключением: у случайного метода выход за них означает не ошибку в
    коде, а слишком большой разброс.
    """
    problems = []

    if value > 0 and std_error / value > NOISE_WARNING_FRACTION:
        problems.append(
            f"случайная ошибка {std_error:.4g} составляет "
            f"{std_error / value:.0%} от цены"
        )

    out_of_bounds = bounds.violation(params, value)
    if out_of_bounds is not None:
        problems.append(out_of_bounds)

    if not problems:
        return None

    return (
        f"Оценке доверять нельзя: {'; '.join(problems)}. "
        f"Увеличьте число путей (сейчас {paths}) или проверьте волатильность"
    )


def _require_european(params: OptionParams) -> None:
    """Американский стиль требует алгоритма Лонгстаффа-Шварца.

    Симуляция идёт вперёд во времени, а решение «исполнять или ждать»
    в каждый момент опирается на будущее. Обычным прогоном путей это не
    считается, и молча вернуть европейскую цену нельзя — она занижена.
    """
    if params.exercise is not ExerciseStyle.EUROPEAN:
        raise AmericanNotSupported(
            "Монте-Карло в этой реализации считает только европейский стиль. "
            "Для американского используйте биномиальное дерево."
        )


def _check_paths(paths: int) -> None:
    if paths < MIN_PATHS:
        raise InvalidSimulationParameters(
            f"Путей должно быть не меньше {MIN_PATHS}: на {paths} результат "
            f"будет слишком шумным, чтобы им пользоваться"
        )
    if paths > MAX_PATHS:
        raise InvalidSimulationParameters(
            f"Путей не больше {MAX_PATHS}: дальше расчёт упрётся в память "
            f"и время, а точность растёт лишь как корень из их числа"
        )
