"""Модели условной волатильности: GARCH и EGARCH.

Историческая волатильность считает одно число по окну и молча
предполагает, что все дни в окне равноценны. На рынке это не так:
спокойные и бурные периоды идут сгустками, за большим движением вероятнее
большое, а не среднее. GARCH описывает именно это — дисперсия сегодня
зависит от вчерашнего движения и от вчерашней же дисперсии:

    sigma^2[t] = omega + alpha * e^2[t-1] + beta * sigma^2[t-1]

EGARCH добавляет то, чего GARCH не видит: падения раскачивают
волатильность сильнее, чем подъёмы той же величины. За это отвечает
отдельный коэффициент gamma при знаковой части, а само уравнение
записано для логарифма дисперсии, поэтому положительность выходит сама и
ограничений на знаки параметров не нужно.

Модуль лежит в `app/pricing/`, то есть не знает ни про сеть, ни про базу:
на входе массив логарифмических доходностей, на выходе числа. Из этого
правила есть одно отступление — зависимость `arch` вместо только
numpy/scipy. Она такая же расчётная библиотека, как scipy, и подгонка
максимального правдоподобия для GARCH своими руками была бы переписыванием
чужого проверенного кода.

**Главное про прогноз.** Для цены опциона нужна не волатильность на
завтра, а средняя за весь срок его жизни: цена зависит от накопленной
дисперсии до экспирации. GARCH даёт прогноз на каждый день отдельно, и
кривая ползёт от текущего уровня к долгосрочному, поэтому в формулу идёт
корень из средней дневной дисперсии по горизонту. Разница с «завтрашним»
значением на месячном сроке доходит до полутора процентных пунктов, а на
годовом больше — подставив однодневное, цену получишь систематически
неверную.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from arch.univariate import EGARCH, GARCH, ZeroMean
from arch.univariate.distribution import StudentsT

# Торговых дней в году: тот же множитель, что и у исторической
# волатильности в market_data, иначе числа в одной таблице оказались бы
# приведены к году по-разному.
TRADING_DAYS = 252

# Меньше этого подгонять бессмысленно: у GARCH четыре параметра, и на
# коротком ряду оценки гуляют сильнее, чем сама волатильность.
MIN_OBSERVATIONS = 250

# Библиотека ждёт доходности в процентах. На долях единицы оптимизатор
# правдоподобия работает с числами порядка 1e-4 и сходится плохо — это
# описано в её собственной документации. Внутрь переводим, наружу отдаём
# доли, как принято во всём проекте.
PERCENT = 100.0

# Путей на симуляционный прогноз EGARCH. У него, в отличие от GARCH,
# аналитического прогноза дальше одного дня не существует: уравнение для
# логарифма дисперсии не даёт замкнутой формы для многошагового среднего.
SIMULATIONS = 2000

# Фиксированное зерно — ради воспроизводимости: два запроса с теми же
# данными должны давать то же число, иначе таблица дёргалась бы при каждом
# обновлении страницы. То же соображение, что у SEED в monte_carlo.
#
# Задаётся самому распределению, а не аргументом `random_state` у
# forecast: когда свой генератор не передан, библиотека берёт выборку из
# генератора распределения, и переданное зерно до неё не доходит. Проверено
# — два вызова подряд давали 24.4876% и 24.5104%.
SEED = 20240814


class NotEnoughHistory(Exception):
    """Ряд слишком короткий, чтобы подгонять модель."""


@dataclass(frozen=True, slots=True)
class VolatilityFit:
    """Подогнанная модель: параметры, качество и прогноз."""

    name: str
    observations: int

    params: dict[str, float]
    log_likelihood: float
    aic: float
    """Критерий Акаике: чем меньше, тем лучше модель описывает ряд.

    Сравнивать по нему GARCH с EGARCH осмысленно — ряд один и тот же, а
    разное число параметров он уже учитывает.
    """

    next_day: float
    """Прогноз на один день вперёд, годовая доля единицы."""

    term: float
    """Средняя по горизонту — то, что идёт в цену опциона."""

    horizon_days: int


def fit_garch(returns: np.ndarray, horizon_days: int) -> VolatilityFit:
    """GARCH(1,1) с распределением Стьюдента.

    Стьюдент, а не нормальное: у дневных доходностей хвосты тяжелее
    нормальных, и на нормальном предположении оценки параметров смещаются.
    Средняя доходность зафиксирована нулём — на дневных данных она
    неотличима от шума, а лишний параметр только добавляет разброса.
    """
    return _fit(returns, horizon_days, "GARCH(1,1)", GARCH(p=1, q=1), analytic=True)


def fit_egarch(returns: np.ndarray, horizon_days: int) -> VolatilityFit:
    """EGARCH(1,1) — то же плюс разная реакция на падения и подъёмы.

    Коэффициент gamma отрицателен почти всегда: это и есть рыночная
    асимметрия, падение раскачивает волатильность сильнее подъёма той же
    величины. GARCH её не видит вовсе, потому что входит в уравнение
    через квадрат.
    """
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
    """Общая часть: подгонка правдоподобием и прогноз на горизонт."""
    clean = np.asarray(returns, dtype=float)
    clean = clean[np.isfinite(clean)]

    if len(clean) < MIN_OBSERVATIONS:
        raise NotEnoughHistory(
            f"для {name} нужно хотя бы {MIN_OBSERVATIONS} наблюдений, "
            f"есть {len(clean)}"
        )
    if horizon_days < 1:
        raise ValueError("Горизонт прогноза меньше дня")

    # Собираем модель из классов, а не удобной функцией arch_model: та
    # принимает распределение только строкой, а нам нужно передать его с
    # зерном. Заодно видно состав явно: нулевое среднее, свой процесс
    # волатильности, Стьюдент.
    model = ZeroMean(
        clean * PERCENT, volatility=process, distribution=StudentsT(seed=SEED)
    )

    # Библиотека шумит предупреждениями о масштабе и о сходимости на
    # каждой подгонке. Ловим их здесь, чтобы не засорять журнал сервера:
    # качество подгонки видно по log_likelihood и AIC в ответе.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = model.fit(disp="off", show_warning=False)

        extra = {} if analytic else {
            "method": "simulation",
            "simulations": SIMULATIONS,
        }
        forecast = fitted.forecast(horizon=horizon_days, reindex=False, **extra)

    # Дисперсия в процентах в квадрате по каждому дню горизонта.
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
    """Дневная дисперсия в процентах в квадрате — в годовую долю единицы."""
    return float(np.sqrt(daily_variance * TRADING_DAYS)) / PERCENT


def historical(returns: np.ndarray, window: int) -> float | None:
    """Обычная историческая волатильность по окну — для той же таблицы.

    Считается здесь, а не берётся из снимка, чтобы все строки таблицы
    опирались на один и тот же ряд доходностей. Иначе разница между
    строками отражала бы ещё и разную длину истории, а не только метод.
    """
    clean = np.asarray(returns, dtype=float)
    clean = clean[np.isfinite(clean)]

    if len(clean) < window:
        return None

    return float(clean[-window:].std(ddof=1) * np.sqrt(TRADING_DAYS))
