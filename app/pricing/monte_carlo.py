"""Монте-Карло: разыгрываем множество возможных цен актива на дату экспирации
и считаем цену опциона как дисконтированное среднее выплаты. Единственная
модель, которая сама измеряет свою погрешность и возвращает её в std_error.
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

MIN_PATHS = 1_000

MAX_PATHS = 5_000_000

SEED = 20240814

CONFIDENCE_Z = 1.959964

# Сдвиг для gamma 2% против 0.1%: вторую разность формируют только пути,
# попавшие к излому выплаты, и узкий сдвиг оставляет от них сотни.
GAMMA_BUMP_FRACTION = 2e-2

NOISE_WARNING_FRACTION = 0.05


class InvalidSimulationParameters(Exception):
    pass


def calculate(params: OptionParams, paths: int = DEFAULT_PATHS) -> PricingResult:
    _require_european(params)
    _check_paths(paths)

    started = time.perf_counter()

    if params.time_to_expiry <= 0:
        return _at_expiry(params, started)

    normals = _draw_normals(paths)

    price_value, std_error = _simulate(params, normals)

    # Греки на том же массиве Z (common random numbers): иначе разность двух
    # зашумлённых оценок дала бы мусор.
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
    _require_european(params)
    _check_paths(paths)

    if params.time_to_expiry <= 0:
        return params.intrinsic_value, 0.0

    return _simulate(params, _draw_normals(paths))


def _draw_normals(paths: int) -> np.ndarray:
    pairs = paths // 2
    generator = np.random.default_rng(SEED)
    half = generator.standard_normal(pairs)

    return np.concatenate([half, -half])


def _simulate(params: OptionParams, normals: np.ndarray) -> tuple[float, float]:
    tau = params.time_to_expiry
    drift = (params.rate - params.div_yield - params.sigma**2 / 2) * tau
    diffusion = params.sigma * np.sqrt(tau)

    # S_T = S * exp((r - q - sigma**2 / 2) * T + sigma * sqrt(T) * Z)
    # Траектория по дням не нужна: выплата зависит только от финальной цены,
    # поэтому ошибки дискретизации здесь нет вовсе, в отличие от дерева.
    terminal = params.spot * np.exp(drift + diffusion * normals)

    if params.is_call:
        payoff = np.maximum(terminal - params.strike, 0.0)
    else:
        payoff = np.maximum(params.strike - terminal, 0.0)

    discounted = np.exp(-params.rate * tau) * payoff

    # Наблюдение — это пара (Z, -Z), а не отдельный путь: делить на корень
    # из общего числа путей нельзя, ошибка вышла бы заниженной.
    pairs = len(normals) // 2
    pair_means = (discounted[:pairs] + discounted[pairs:]) / 2

    value = float(pair_means.mean())
    std_error = float(pair_means.std(ddof=1) / np.sqrt(pairs))

    return value, std_error


def _at_expiry(params: OptionParams, started: float) -> PricingResult:
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
