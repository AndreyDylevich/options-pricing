"""Оркестрация расчёта: срок, выбор модели, сборка ответа.

Единственное место, где встречаются HTTP-схемы и математика. Сами модели
о существовании этого слоя не знают.
"""

import math
import time
from dataclasses import asdict, replace
from datetime import date

from app.pricing import binomial, black_scholes, implied_vol, monte_carlo, parity
from app.pricing.binomial import InvalidTreeParameters
from app.pricing.models import (
    AmericanNotSupported,
    OptionParams,
    OptionType,
    PriceOutOfBounds,
    PricingModel,
    PricingResult,
)
from app.pricing.monte_carlo import InvalidSimulationParameters
from app.schemas.instruments import (
    OptionChain,
    OptionQuote,
    ParityReport,
    ParityRow,
)
from app.schemas.pricing import (
    CompareRequest,
    CompareResponse,
    ConvergenceReport,
    GreeksOut,
    ModelComparison,
    MonteCarloConvergencePoint,
    OptionInputs,
    PriceMeta,
    PriceRequest,
    PriceResponse,
    TreeConvergencePoint,
)
from app.services import market_data

# Календарные, а не торговые: деньги дисконтируются каждый день, включая
# выходные. Волатильность при этом приводится к году через 252 — разница
# намеренная, см. market_data.
DAYS_IN_YEAR = 365.0


class ModelNotImplemented(Exception):
    """Модель заявлена в API, но ещё не написана."""


class UnknownExpiry(Exception):
    """Такой даты экспирации по этой бумаге нет."""


def calculate(request: PriceRequest) -> PriceResponse:
    """Полный путь от параметров формы до цены с греками."""
    if market_data.find_instrument(request.ticker) is None:
        raise market_data.InstrumentNotFound(request.ticker)

    days = (request.expiry - date.today()).days
    params = _to_params(request, days)
    result = _dispatch(request.model, params, request)

    return PriceResponse(
        price=result.price,
        greeks=GreeksOut(**asdict(result.greeks)),
        meta=PriceMeta(
            model=result.model,
            exercise=params.exercise,
            elapsed_ms=round(result.elapsed_ms, 3),
            days_to_expiry=days,
            time_to_expiry=params.time_to_expiry,
            std_error=result.std_error,
            warning=result.warning,
            ci95=result.ci95,
        ),
    )


def compare(request: CompareRequest) -> CompareResponse:
    """Все три модели на одних и тех же входных данных.

    Отдельный эндпоинт, а не три запроса с фронта: так гарантируется, что
    модели считали ровно одно и то же. Отказ одной модели не отменяет
    остальные — он приходит текстом в её строке таблицы.
    """
    if market_data.find_instrument(request.ticker) is None:
        raise market_data.InstrumentNotFound(request.ticker)

    days = (request.expiry - date.today()).days
    params = _to_params(request, days)

    return CompareResponse(
        exercise=params.exercise,
        days_to_expiry=days,
        time_to_expiry=params.time_to_expiry,
        results=[_compare_one(model, params, request) for model in PricingModel],
    )


def _compare_one(
    model: PricingModel, params: OptionParams, request: CompareRequest
) -> ModelComparison:
    """Одна строка таблицы сравнения.

    Ловим здесь то, что в обычном расчёте уходит наверх ошибкой HTTP:
    в сравнении отказ конкретной модели — это содержательный результат,
    а не сбой запроса.
    """
    try:
        result = _dispatch(model, params, request)
    except (
        AmericanNotSupported,
        InvalidTreeParameters,
        InvalidSimulationParameters,
        ModelNotImplemented,
        PriceOutOfBounds,
        ValueError,
    ) as error:
        return ModelComparison(model=model, error=str(error))

    return ModelComparison(
        model=model,
        price=result.price,
        greeks=GreeksOut(**asdict(result.greeks)),
        elapsed_ms=round(result.elapsed_ms, 3),
        std_error=result.std_error,
        ci95=result.ci95,
        warning=result.warning,
    )


def _to_params(request: OptionInputs, days: int) -> OptionParams:
    """Схема запроса — во внутренний датакласс.

    Срок считается на сервере, а не берётся из формы: браузер может стоять
    в другом часовом поясе или с переведёнными часами, и T разъедется с
    тем, что видел пользователь.
    """
    return OptionParams(
        spot=request.spot,
        strike=request.strike,
        time_to_expiry=days / DAYS_IN_YEAR,
        rate=request.rate,
        div_yield=request.div_yield,
        sigma=request.sigma,
        option_type=request.option_type,
        exercise=request.exercise,
    )


def _dispatch(
    model: PricingModel, params: OptionParams, request: OptionInputs
) -> PricingResult:
    if model is PricingModel.BLACK_SCHOLES:
        return black_scholes.calculate(params)

    if model is PricingModel.BINOMIAL:
        return binomial.calculate(params, steps=request.steps)

    if model is PricingModel.MONTE_CARLO:
        return monte_carlo.calculate(params, paths=request.paths)

    raise ModelNotImplemented(model.value)


# Ширина окна по ln(K / F). Полсотых логарифма — это страйки примерно от
# −40% до +65% относительно форварда. Дальше начинаются контракты, чей
# спред больше временной стоимости, и остатки досплитных серий: их цены
# нарушают границы арбитража, и волатильность из них не извлекается.
DEFAULT_MONEYNESS_WINDOW = 0.5


def option_chain(
    ticker: str, expiry: str, moneyness_window: float = DEFAULT_MONEYNESS_WINDOW
) -> OptionChain:
    """Цепочка опционов с подразумеваемой волатильностью по каждому страйку.

    Волатильность считается по середине bid-ask, а не по цене последней
    сделки: сделка могла быть неделю назад при другой цене актива, и
    развёрнутая из неё волатильность не имеет отношения к сегодняшнему
    рынку. Именно поэтому колонка impliedVolatility от Yahoo, посчитанная
    от lastPrice, регулярно выдаёт нули и сотни процентов — её мы кладём
    рядом справочно, для сравнения.
    """
    snapshot = market_data.get_market_snapshot(ticker)

    if expiry not in snapshot.expirations:
        raise UnknownExpiry(expiry)

    days = (date.fromisoformat(expiry) - date.today()).days
    tau = days / DAYS_IN_YEAR

    # Эффективное q под эту экспирацию, а не среднегодовое: от него зависит
    # и форвард, и волатильность, которую мы из цен разворачиваем. Снимок,
    # поднятый из базы после перезапуска, карты не содержит — тогда
    # откатываемся на трейлинг.
    div_yield = snapshot.div_yield_by_expiry.get(expiry, snapshot.div_yield)
    forward = snapshot.spot * math.exp((snapshot.risk_free_rate - div_yield) * tau)

    rows, total = market_data.get_option_chain(ticker, expiry)
    quotes = [
        _quote_with_iv(side, row, snapshot, div_yield, tau, forward)
        for side, row in rows
    ]

    if moneyness_window > 0:
        quotes = [q for q in quotes if abs(q.log_moneyness) <= moneyness_window]

    quotes.sort(key=lambda q: (q.option_type.value, q.strike))

    return OptionChain(
        ticker=snapshot.ticker,
        expiry=expiry,
        days_to_expiry=days,
        time_to_expiry=tau,
        spot=snapshot.spot,
        rate=snapshot.risk_free_rate,
        div_yield=div_yield,
        forward=forward,
        quotes=quotes,
        quoted=len(quotes),
        skipped=total - len(quotes),
        solved=sum(1 for q in quotes if q.implied_vol is not None),
    )


def _quote_with_iv(
    side: str, row: dict, snapshot, div_yield: float, tau: float, forward: float
) -> OptionQuote:
    """Один контракт: середина спреда и развёрнутая из неё волатильность."""
    strike = float(row["strike"])
    bid, ask = float(row["bid"]), float(row["ask"])
    mid = (bid + ask) / 2

    params = OptionParams(
        spot=snapshot.spot,
        strike=strike,
        time_to_expiry=tau,
        rate=snapshot.risk_free_rate,
        div_yield=div_yield,
        sigma=0.2,  # заглушка: именно это значение и подбирается
        option_type=OptionType(side),
    )
    solved = implied_vol.solve(params, mid)

    return OptionQuote(
        strike=strike,
        option_type=OptionType(side),
        bid=bid,
        ask=ask,
        mid=mid,
        volume=_as_int(row.get("volume")),
        open_interest=_as_int(row.get("openInterest")),
        implied_vol=solved.sigma,
        iv_note=solved.reason,
        yahoo_iv=_as_float(row.get("impliedVolatility")),
        log_moneyness=math.log(strike / forward),
    )


def _as_int(value) -> int | None:
    """NaN у yfinance означает «нет данных», а не ноль."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else int(number)


def _as_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


# Лестница по числу шагов дерева. Мелкий шаг в начале не для красоты: у
# CRR отклонение меняет знак при переходе с чётного числа шагов на
# нечётное, и без подряд идущих значений эта пила на графике не видна —
# кривая выглядела бы просто дёргающейся.
TREE_LADDER = (
    *range(5, 41),
    50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2000,
)

# Лестница по числу путей — геометрическая: ошибка падает как корень,
# поэтому равномерные шаги в конце ничего не показывали бы.
MONTE_CARLO_LADDER = (
    1_000, 2_000, 5_000, 10_000, 25_000, 50_000,
    100_000, 250_000, 500_000, 1_000_000,
)


def convergence(request: PriceRequest) -> ConvergenceReport:
    """Как дерево и Монте-Карло подходят к точному значению.

    Точное значение берём у Блэка-Шоулза, поэтому американский стиль сюда
    не годится: сравнивать было бы не с чем.
    """
    if market_data.find_instrument(request.ticker) is None:
        raise market_data.InstrumentNotFound(request.ticker)

    days = (request.expiry - date.today()).days
    params = _to_params(request, days)

    # Поднимется AmericanNotSupported, если стиль американский: у той же
    # проверки внутри black_scholes текст уже написан.
    reference = black_scholes.price(params)

    tree_points, skipped = _tree_convergence(params, reference)

    return ConvergenceReport(
        reference=reference,
        days_to_expiry=days,
        time_to_expiry=params.time_to_expiry,
        tree=tree_points,
        monte_carlo=_monte_carlo_convergence(params, reference),
        skipped_steps=skipped,
    )


def _tree_convergence(
    params: OptionParams, reference: float
) -> tuple[list[TreeConvergencePoint], list[int]]:
    points: list[TreeConvergencePoint] = []
    skipped: list[int] = []

    for steps in TREE_LADDER:
        started = time.perf_counter()
        try:
            value = binomial.price(params, steps=steps)
        except binomial.InvalidTreeParameters:
            # На малом числе шагов риск-нейтральная вероятность может выйти
            # за [0, 1]. Это не повод ронять весь отчёт: пропускаем точку и
            # сообщаем, какие именно не посчитались.
            skipped.append(steps)
            continue

        points.append(
            TreeConvergencePoint(
                steps=steps,
                price=value,
                error=value - reference,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        )

    return points, skipped


def _monte_carlo_convergence(
    params: OptionParams, reference: float
) -> list[MonteCarloConvergencePoint]:
    points: list[MonteCarloConvergencePoint] = []

    for paths in MONTE_CARLO_LADDER:
        started = time.perf_counter()
        value, std_error = monte_carlo.estimate(params, paths)
        margin = monte_carlo.CONFIDENCE_Z * std_error

        points.append(
            MonteCarloConvergencePoint(
                paths=paths,
                price=value,
                error=value - reference,
                std_error=std_error,
                ci95_low=value - margin,
                ci95_high=value + margin,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        )

    return points


def parity_report(ticker: str, expiry: str) -> ParityReport:
    """Сверка наших входных данных с рынком через паритет колла и пута.

    Единственная внешняя проверка в проекте. Все остальные внутренние и
    отвечают на вопрос «правильно ли посчитана формула»; эта отвечает на
    вопрос «верны ли числа, которые мы в неё подставили» — спот, ставка,
    дивидендная доходность. Волатильность в паритет не входит, поэтому
    проверка не зависит от того, как мы её оцениваем.
    """
    snapshot = market_data.get_market_snapshot(ticker)

    if expiry not in snapshot.expirations:
        raise UnknownExpiry(expiry)

    days = (date.fromisoformat(expiry) - date.today()).days
    tau = days / DAYS_IN_YEAR
    rate = snapshot.risk_free_rate
    div_yield = snapshot.div_yield_by_expiry.get(expiry, snapshot.div_yield)

    rows, _ = market_data.get_option_chain(ticker, expiry)
    checks = _parity_checks(rows, snapshot.spot, rate, div_yield, tau)

    consensus = parity.consensus_forward(checks)
    forward_market, forward_tolerance = consensus if consensus else (None, None)

    div_yield_market = (
        parity.implied_div_yield(forward_market, snapshot.spot, rate, tau)
        if forward_market is not None
        else None
    )

    agreed = sum(1 for item in checks if item.within_quotes)
    arbitrage = sum(1 for item in checks if item.arbitrage is not None)

    return ParityReport(
        ticker=snapshot.ticker,
        expiry=expiry,
        days_to_expiry=days,
        time_to_expiry=tau,
        spot=snapshot.spot,
        rate=rate,
        div_yield_ours=div_yield,
        div_yield_market=div_yield_market,
        forward_ours=snapshot.spot * math.exp((rate - div_yield) * tau),
        forward_market=forward_market,
        forward_tolerance=forward_tolerance,
        pairs=len(checks),
        agreed=agreed,
        arbitrage=arbitrage,
        verdict=_parity_verdict(
            pairs=len(checks),
            agreed=agreed,
            arbitrage=arbitrage,
            forward_ours=snapshot.spot * math.exp((rate - div_yield) * tau),
            forward_market=forward_market,
            forward_tolerance=forward_tolerance,
            div_yield_ours=div_yield,
            div_yield_market=div_yield_market,
        ),
        rows=[
            ParityRow(
                strike=item.strike,
                call_mid=item.call_mid,
                put_mid=item.put_mid,
                observed=item.observed,
                expected=item.expected,
                residual=item.residual,
                tolerance=item.tolerance,
                within_quotes=item.within_quotes,
                implied_forward=item.implied_forward,
                implied_div_yield=item.implied_div_yield,
                arbitrage=item.arbitrage,
            )
            for item in checks
        ],
    )


def _parity_checks(
    rows: list[tuple[str, dict]],
    spot: float,
    rate: float,
    div_yield: float,
    tau: float,
) -> list[parity.ParityCheck]:
    """Свести коллы и путы по страйкам и проверить каждую пару.

    Паритет говорит о паре на одном страйке, поэтому одинокий контракт
    бесполезен — таких в цепочке всегда хватает, особенно на краях.
    """
    sides: dict[str, dict[float, parity.Quote]] = {"call": {}, "put": {}}

    for side, row in rows:
        try:
            quote = parity.Quote(bid=float(row["bid"]), ask=float(row["ask"]))
        except (TypeError, ValueError):
            # Перевёрнутый или пустой спред: сверять по нему нечего.
            continue
        sides[side][float(row["strike"])] = quote

    common = sorted(set(sides["call"]) & set(sides["put"]))

    return [
        parity.check(
            strike, sides["call"][strike], sides["put"][strike],
            spot, rate, div_yield, tau,
        )
        for strike in common
    ]


# Доля страйков вне границ, после которой дело уже не в отдельных
# котировках. Границы считаются от нашего спота, так что уход за них всей
# цепочкой разом означает одно: спот не совпадает с моментом, на который
# сняты котировки опционов.
SYSTEMATIC_BREACH = 0.5


def _parity_verdict(
    *,
    pairs: int,
    agreed: int,
    arbitrage: int,
    forward_ours: float,
    forward_market: float | None,
    forward_tolerance: float | None,
    div_yield_ours: float,
    div_yield_market: float | None,
) -> str:
    """Вывод человеческим языком: сошлось или нет и в чём именно.

    Согласие решается по форварду, а не по q, хотя расхождение q читать
    нагляднее. Причина в единицах: q — ставка годовая, и один и тот же
    сдвиг форварда на месячном сроке даёт в двенадцать раз большее
    расхождение q, чем на годовом. Любой фиксированный порог по q на
    коротких сроках кричал бы впустую. Форвард же измеряется в долларах
    и сравнивается с допуском, посчитанным из тех же спредов, что и в
    построчной таблице, — вердикт и строки перестают спорить друг с другом.
    """
    if pairs == 0:
        return "Сверять нечего: нет ни одного страйка, где котируются и колл, и пут"

    share = f"{agreed} из {pairs} страйков сошлись в пределах спредов"

    if forward_market is None or forward_tolerance is None:
        return f"{share}. Рыночный форвард вытащить не удалось"

    gap = forward_market - forward_ours
    comparison = (
        f"Форвард: наш {forward_ours:.2f} против рыночного "
        f"{forward_market:.2f}, расхождение {gap:+.2f} при допуске "
        f"{forward_tolerance:.2f}"
    )

    if div_yield_market is not None:
        comparison += (
            f". В пересчёте на дивиденды — наши {div_yield_ours * 100:.2f}% "
            f"против рыночных {div_yield_market * 100:.2f}%"
        )

    if abs(gap) <= forward_tolerance:
        return f"{share}. {comparison} — данные согласованы"

    if pairs and arbitrage / pairs >= SYSTEMATIC_BREACH:
        return (
            f"{share}. {comparison}. Вне границ паритета сразу {arbitrage} "
            f"страйков из {pairs} — столько разом не бывает случайностью: "
            f"скорее всего наш спот снят не в тот момент, что котировки "
            f"опционов. Обновите рыночные данные"
        )

    return f"{share}. {comparison} — стоит разобраться"
