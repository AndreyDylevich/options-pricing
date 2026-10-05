"""GARCH и EGARCH: по истории доходностей считаем волатильность с поправкой
на то, что спокойные и бурные периоды идут сгустками, и прогнозируем её
среднюю за срок опциона. Подгонку правдоподобием делает библиотека `arch`.
"""

import warnings
from dataclasses import dataclass

import numpy as np

# Единственное отступление от правила «в app/pricing только numpy и scipy»:
# arch — такая же расчётная библиотека, а писать подгонку максимального
# правдоподобия для GARCH руками значило бы переписывать чужой
# проверенный код. Сети и базы модуль по-прежнему не знает.
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


# Ряд слишком короткий, чтобы подгонять модель.
class NotEnoughHistory(Exception):
    pass


@dataclass(frozen=True, slots=True)
class VolatilityFit:
    name: str
    observations: int

    params: dict[str, float]
    log_likelihood: float
    # Критерий Акаике: чем меньше, тем лучше модель описывает ряд.
    # Сравнивать по нему GARCH с EGARCH осмысленно — ряд один и тот же, а
    # разное число параметров он уже учитывает.
    aic: float

    next_day: float   # прогноз на один день вперёд, годовая доля единицы
    term: float       # средняя по горизонту — она и идёт в цену опциона
    horizon_days: int


# sigma^2[t] = omega + alpha * e^2[t-1] + beta * sigma^2[t-1]
#
# Дисперсия сегодня зависит от вчерашнего движения и от вчерашней же
# дисперсии — это и описывает сгустки, которых историческая волатильность
# не видит, считая все дни в окне равноценными.
def fit_garch(returns: np.ndarray, horizon_days: int) -> VolatilityFit:
    return _fit(returns, horizon_days, "GARCH(1,1)", GARCH(p=1, q=1), analytic=True)


# То же плюс асимметрия: падения раскачивают волатильность сильнее, чем
# подъёмы той же величины. За это отвечает коэффициент gamma при знаковой
# части, и он почти всегда отрицателен. GARCH такого не видит вовсе, потому
# что шок входит в его уравнение через квадрат.
#
# Уравнение записано для логарифма дисперсии, поэтому положительность
# выходит сама и ограничений на знаки параметров не нужно.
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
    #
    # В цену опциона идёт средняя по горизонту, а не прогноз на завтра:
    # цена зависит от накопленной дисперсии до экспирации, а кривая
    # прогноза ползёт от текущего уровня к долгосрочному. На месячном
    # сроке разница доходит до полутора процентных пунктов, на годовом
    # больше — подставив однодневное, цену получишь систематически
    # неверную.
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


# Дневная дисперсия в процентах в квадрате — в годовую долю единицы.
def _annualize(daily_variance: float) -> float:
    return float(np.sqrt(daily_variance * TRADING_DAYS)) / PERCENT


# Обычная историческая волатильность по окну. Считается здесь, а не берётся
# из снимка, чтобы все строки таблицы опирались на один и тот же ряд
# доходностей: иначе разница между строками отражала бы ещё и разную длину
# истории, а не только метод.
def historical(returns: np.ndarray, window: int) -> float | None:
    clean = np.asarray(returns, dtype=float)
    clean = clean[np.isfinite(clean)]

    if len(clean) < window:
        return None

    return float(clean[-window:].std(ddof=1) * np.sqrt(TRADING_DAYS))
