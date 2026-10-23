"""GARCH и EGARCH: по истории доходностей считаем волатильность с поправкой
на то, что спокойные и бурные периоды идут сгустками, и прогнозируем её
среднюю за срок опциона. Подгонку правдоподобием делает библиотека `arch`.
"""

import warnings
from dataclasses import dataclass

import numpy as np

# Единственное отступление от «в app/pricing только numpy и scipy»: arch —
# такая же расчётная библиотека. Сети и базы модуль по-прежнему не знает.
from arch.univariate import EGARCH, GARCH, ZeroMean
from arch.univariate.distribution import StudentsT

TRADING_DAYS = 252

MIN_OBSERVATIONS = 250

PERCENT = 100.0

SIMULATIONS = 2000

# Зерно задаётся самому распределению, а не аргументом random_state у
# forecast: иначе до выборки оно не доходит — два вызова давали разные числа.
SEED = 20240814


class NotEnoughHistory(Exception):
    pass


@dataclass(frozen=True, slots=True)
class VolatilityFit:
    name: str
    observations: int

    params: dict[str, float]
    log_likelihood: float
    aic: float

    next_day: float   # прогноз на один день вперёд, годовая доля единицы
    term: float       # средняя по горизонту — она и идёт в цену опциона
    horizon_days: int


# sigma^2[t] = omega + alpha * e^2[t-1] + beta * sigma^2[t-1]
def fit_garch(returns: np.ndarray, horizon_days: int) -> VolatilityFit:
    return _fit(returns, horizon_days, "GARCH(1,1)", GARCH(p=1, q=1), analytic=True)


def fit_egarch(returns: np.ndarray, horizon_days: int) -> VolatilityFit:
    return _fit(
        returns, horizon_days, "EGARCH(1,1)", EGARCH(p=1, o=1, q=1), analytic=False
    )


def _fit(
    returns: np.ndarray,
    horizon_days: int,
    name: str,
    process,
    analytic: bool,
) -> VolatilityFit:
    clean = np.asarray(returns, dtype=float)
    clean = clean[np.isfinite(clean)]

    if len(clean) < MIN_OBSERVATIONS:
        raise NotEnoughHistory(
            f"для {name} нужно хотя бы {MIN_OBSERVATIONS} наблюдений, "
            f"есть {len(clean)}"
        )
    if horizon_days < 1:
        raise ValueError("Горизонт прогноза меньше дня")

    model = ZeroMean(
        clean * PERCENT, volatility=process, distribution=StudentsT(seed=SEED)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = model.fit(disp="off", show_warning=False)

        extra = {} if analytic else {
            "method": "simulation",
            "simulations": SIMULATIONS,
        }
        forecast = fitted.forecast(horizon=horizon_days, reindex=False, **extra)

    # В цену идёт средняя по горизонту, а не прогноз на завтра: на месячном
    # сроке разница доходит до полутора процентных пунктов.
    daily_variance = np.asarray(forecast.variance)[0]

    return VolatilityFit(
        name=name,
        observations=len(clean),
        params={key: float(value) for key, value in fitted.params.items()},
        log_likelihood=float(fitted.loglikelihood),
        aic=float(fitted.aic),
        next_day=_annualize(daily_variance[0]),
        term=_annualize(daily_variance.mean()),
        horizon_days=horizon_days,
    )


def _annualize(daily_variance: float) -> float:
    return float(np.sqrt(daily_variance * TRADING_DAYS)) / PERCENT


def historical(returns: np.ndarray, window: int) -> float | None:
    clean = np.asarray(returns, dtype=float)
    clean = clean[np.isfinite(clean)]

    if len(clean) < window:
        return None

    return float(clean[-window:].std(ddof=1) * np.sqrt(TRADING_DAYS))
