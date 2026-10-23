"""Оркестрация: считает срок до экспирации, собирает параметры опциона, зовёт
нужную модель и складывает ответ. Единственное место, где встречаются
HTTP-схемы и математика — сами модели об этом слое не знают.
"""

import math
import time
from dataclasses import asdict, replace
from datetime import date

from app.pricing import (
    binomial,
    black_scholes,
    implied_vol,
    monte_carlo,
    parity,
    volatility,
)
from app.pricing.binomial import InvalidTreeParameters
from app.pricing.models import (
    AmericanNotSupported,
    ExerciseStyle,
    Greeks,
    OptionParams,
    OptionType,
    PriceOutOfBounds,
    PricingModel,
    PricingResult,
)
from app.pricing.monte_carlo import InvalidSimulationParameters
from app.schemas.instruments import (
    ContractQuote,
    OptionChain,
    OptionQuote,
    ParityReport,
    ParityRow,
    VolatilityModelFit,
    VolatilityOption,
    VolatilityReport,
)
from app.schemas.pricing import (
    CompareRequest,
    CompareResponse,
    ConvergenceReport,
    GreekCurve,
    GreekProfiles,
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
# выходные. Волатильность при этом приводится к году через 252.
DAYS_IN_YEAR = 365.0


class ModelNotImplemented(Exception):
    pass


class UnknownExpiry(Exception):
    pass


def calculate(request: PriceRequest) -> PriceResponse:
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


DEFAULT_MONEYNESS_WINDOW = 0.5


def option_chain(
    ticker: str, expiry: str, moneyness_window: float = DEFAULT_MONEYNESS_WINDOW
) -> OptionChain:
    snapshot = market_data.get_market_snapshot(ticker)

    if expiry not in snapshot.expirations:
        raise UnknownExpiry(expiry)

    days = (date.fromisoformat(expiry) - date.today()).days
    tau = days / DAYS_IN_YEAR

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


# Мелкий шаг в начале не для красоты: у CRR отклонение меняет знак при
# переходе с чётного числа шагов на нечётное, и без подряд идущих значений
# эта пила на графике не видна.
TREE_LADDER = (
    *range(5, 41),
    50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2000,
)

MONTE_CARLO_LADDER = (
    1_000, 2_000, 5_000, 10_000, 25_000, 50_000,
    100_000, 250_000, 500_000, 1_000_000,
)


def convergence(request: PriceRequest) -> ConvergenceReport:
    if market_data.find_instrument(request.ticker) is None:
        raise market_data.InstrumentNotFound(request.ticker)

    days = (request.expiry - date.today()).days
    params = _to_params(request, days)

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
    sides: dict[str, dict[float, parity.Quote]] = {"call": {}, "put": {}}

    for side, row in rows:
        try:
            quote = parity.Quote(bid=float(row["bid"]), ask=float(row["ask"]))
        except (TypeError, ValueError):
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


SYSTEMATIC_BREACH = 0.5


# Согласие решается по форварду, а не по q: q — ставка годовая, и тот же
# сдвиг форварда на месячном сроке даёт в двенадцать раз большее расхождение
# q, чем на годовом. Фиксированный порог по q кричал бы впустую.
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
    if pairs == 0:
        return (
            "Сверять нечего: нет ни одного страйка, где котируются и колл, и "
            "пут. Вне торговых часов так и будет — Yahoo отдаёт нулевые bid и "
            "ask, а паритет считается по ним. Рынок работает с 16:30 до 23:00 "
            "по Москве"
        )

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


class ContractNotListed(Exception):
    pass


def contract_quote(
    ticker: str, expiry: str, strike: float, option_type: OptionType
) -> ContractQuote:
    snapshot = market_data.get_market_snapshot(ticker)

    if expiry not in snapshot.expirations:
        raise UnknownExpiry(expiry)

    rows, _ = market_data.get_option_chain(ticker, expiry)
    side = option_type.value

    for row_side, row in rows:
        if row_side != side or not math.isclose(float(row["strike"]), strike):
            continue

        bid, ask = float(row["bid"]), float(row["ask"])
        return ContractQuote(
            ticker=snapshot.ticker,
            expiry=expiry,
            strike=float(row["strike"]),
            option_type=option_type,
            bid=bid,
            ask=ask,
            mid=(bid + ask) / 2,
            spread=ask - bid,
            volume=_as_int(row.get("volume")),
            open_interest=_as_int(row.get("openInterest")),
        )

    raise ContractNotListed(_nearest_strikes_message(rows, side, strike))


NEAREST_STRIKES = 3


def _nearest_strikes_message(
    rows: list[tuple[str, dict]], side: str, strike: float
) -> str:
    listed = sorted({float(row["strike"]) for row_side, row in rows if row_side == side})

    if not listed:
        return (
            "по этой экспирации нет ни одного контракта с живыми котировками "
            "— скорее всего биржа закрыта, американский рынок работает "
            "с 16:30 до 23:00 по Москве"
        )

    nearest = sorted(listed, key=lambda value: abs(value - strike))[:NEAREST_STRIKES]
    return (
        f"контракта на страйк {strike:g} нет, ближайшие торгуемые: "
        + ", ".join(f"{value:g}" for value in sorted(nearest))
    )


PROFILE_POINTS = 120
PROFILE_POINTS_AMERICAN = 60
PROFILE_TREE_STEPS = 200

SPOT_SPAN = 0.4

# От суток, а не от нуля: в нуле тета у опциона на деньгах уходит в бесконечность.
PROFILE_MIN_DAYS = 1
PROFILE_MAX_DAYS = 365


# Модель выбирается по стилю исполнения, а не по выбору пользователя: для
# европейского Блэк-Шоулз даёт точное значение, американский умеет только
# дерево, а Монте-Карло не годится вовсе — кривая вышла бы дёрганой от шума.
def greek_profiles(request: PriceRequest) -> GreekProfiles:
    if market_data.find_instrument(request.ticker) is None:
        raise market_data.InstrumentNotFound(request.ticker)

    days = (request.expiry - date.today()).days
    params = _to_params(request, days)

    american = params.exercise is ExerciseStyle.AMERICAN
    model = PricingModel.BINOMIAL if american else PricingModel.BLACK_SCHOLES
    points = PROFILE_POINTS_AMERICAN if american else PROFILE_POINTS

    def greeks_at(other: OptionParams) -> Greeks | None:
        try:
            if american:
                return binomial.calculate(other, steps=PROFILE_TREE_STEPS).greeks
            return black_scholes.calculate(other).greeks
        except (InvalidTreeParameters, PriceOutOfBounds, ValueError):
            return None

    here = greeks_at(params)
    if here is None:
        raise ValueError("Не удалось посчитать греки в заданной точке")

    spots = _spot_axis(params, points)
    along_spot = [(value, greeks_at(replace(params, spot=value))) for value in spots]

    terms = _days_axis(days, points)
    along_time = [
        (day, greeks_at(replace(params, time_to_expiry=day / DAYS_IN_YEAR)))
        for day in terms
    ]

    return GreekProfiles(
        model=model,
        exercise=params.exercise,
        strike=params.strike,
        days_to_expiry=days,
        delta_vs_spot=_curve(along_spot, "delta", params.spot, here.delta),
        gamma_vs_spot=_curve(along_spot, "gamma", params.spot, here.gamma),
        theta_vs_days=_curve(along_time, "theta", float(days), here.theta),
    )


def _spot_axis(params: OptionParams, points: int) -> list[float]:
    low = min(params.spot, params.strike) * (1 - SPOT_SPAN)
    high = max(params.spot, params.strike) * (1 + SPOT_SPAN)
    step = (high - low) / (points - 1)

    return [low + step * index for index in range(points)]


def _days_axis(days: int, points: int) -> list[float]:
    high = float(max(PROFILE_MAX_DAYS, days))
    step = (high - PROFILE_MIN_DAYS) / (points - 1)

    return [PROFILE_MIN_DAYS + step * index for index in range(points)]


def _curve(
    samples: list[tuple[float, Greeks | None]],
    greek: str,
    current_x: float,
    current_y: float,
) -> GreekCurve:
    usable = [(x, getattr(g, greek)) for x, g in samples if g is not None]

    return GreekCurve(
        x=[x for x, _ in usable],
        y=[y for _, y in usable],
        current_x=current_x,
        current_y=current_y,
    )



# Окно одно, годовое. Короткие дают не «волатильность изменилась», а
# нехватку наблюдений: соседние короткие окна на тех же данных давали
# 12.45% и 22.34%. Тридцать и девяносто дней остаются в карточке рынка.
HISTORY_WINDOWS = ((252, "год"),)


def volatility_report(ticker: str, expiry: str) -> VolatilityReport:
    snapshot = market_data.get_market_snapshot(ticker)

    if expiry not in snapshot.expirations:
        raise UnknownExpiry(expiry)

    days = (date.fromisoformat(expiry) - date.today()).days
    returns = market_data.get_log_returns(ticker)

    options: list[VolatilityOption] = []
    fits: list[VolatilityModelFit] = []
    troubles: list[str] = []

    for window, period_name in HISTORY_WINDOWS:
        value = volatility.historical(returns, window)
        if value is not None:
            options.append(VolatilityOption(
                key=f"hist{window}",
                label=f"Историческая, {window} дн.",
                sigma=value,
                note=f"разброс дневных движений за {period_name}, все дни равноценны",
            ))

    for fit_model, note in (
        (volatility.fit_garch,
         "учитывает, что бурные дни идут сгустками; среднее по сроку опциона"),
        (volatility.fit_egarch,
         "то же плюс асимметрия: падения раскачивают сильнее подъёмов"),
    ):
        try:
            fit = fit_model(returns, days)
        except (volatility.NotEnoughHistory, ValueError) as error:
            troubles.append(str(error))
            continue

        options.append(VolatilityOption(
            key=fit.name.split("(")[0].lower(),
            label=f"{fit.name}, прогноз на {days} дн.",
            sigma=fit.term,
            note=note,
        ))
        fits.append(VolatilityModelFit(
            name=fit.name,
            params=fit.params,
            log_likelihood=fit.log_likelihood,
            aic=fit.aic,
            next_day=fit.next_day,
            term=fit.term,
        ))

    return VolatilityReport(
        ticker=snapshot.ticker,
        expiry=expiry,
        days_to_expiry=days,
        observations=len(returns),
        options=options,
        fits=fits,
        warning="; ".join(troubles) if troubles else None,
    )
