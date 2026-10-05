"""Блэк-Шоулз-Мертон: считаем цену европейского опциона по формуле и греки
аналитическими производными, без пересчёта цены. Для европейского стиля это
точное значение, к которому две остальные модели лишь сходятся.
"""

import time
from math import exp, log, sqrt

from scipy.stats import norm

from app.pricing import bounds
from app.pricing.models import (
    AmericanNotSupported,
    ExerciseStyle,
    Greeks,
    OptionParams,
    PricingModel,
    PricingResult,
)

# Пересчёт греков в терминальные единицы: vega и rho на процентный пункт,
# theta на календарный день.
PERCENTAGE_POINT = 100.0
DAYS_IN_YEAR = 365.0


def calculate(params: OptionParams) -> PricingResult:
    _require_european(params)

    started = time.perf_counter()
    result_price = price(params)
    result_greeks = greeks(params)
    elapsed = (time.perf_counter() - started) * 1000

    bounds.require_within_bounds(params, result_price, "Блэк-Шоулз")

    return PricingResult(
        price=result_price,
        greeks=result_greeks,
        model=PricingModel.BLACK_SCHOLES,
        elapsed_ms=elapsed,
    )


# call = S*exp(-qT)*N(d1) - K*exp(-rT)*N(d2)
# put  = K*exp(-rT)*N(-d2) - S*exp(-qT)*N(-d1)
def price(params: OptionParams) -> float:
    _require_european(params)

    if _is_degenerate(params):
        return _degenerate_price(params)

    d1, d2 = _d1_d2(params)
    discount_spot = params.spot * exp(-params.div_yield * params.time_to_expiry)
    discount_strike = params.strike * exp(-params.rate * params.time_to_expiry)

    if params.is_call:
        return discount_spot * norm.cdf(d1) - discount_strike * norm.cdf(d2)
    return discount_strike * norm.cdf(-d2) - discount_spot * norm.cdf(-d1)


def greeks(params: OptionParams) -> Greeks:
    _require_european(params)

    if _is_degenerate(params):
        return _degenerate_greeks(params)

    d1, d2 = _d1_d2(params)
    spot, strike = params.spot, params.strike
    tau, sigma = params.time_to_expiry, params.sigma
    rate, div = params.rate, params.div_yield

    div_discount = exp(-div * tau)
    rate_discount = exp(-rate * tau)
    pdf_d1 = norm.pdf(d1)

    gamma = div_discount * pdf_d1 / (spot * sigma * sqrt(tau))
    vega = spot * div_discount * pdf_d1 * sqrt(tau)

    # Общий для call и put член временного распада.
    decay = -spot * pdf_d1 * sigma * div_discount / (2 * sqrt(tau))

    if params.is_call:
        delta = div_discount * norm.cdf(d1)
        theta = (
            decay
            + div * spot * div_discount * norm.cdf(d1)
            - rate * strike * rate_discount * norm.cdf(d2)
        )
        rho = strike * tau * rate_discount * norm.cdf(d2)
    else:
        delta = -div_discount * norm.cdf(-d1)
        theta = (
            decay
            - div * spot * div_discount * norm.cdf(-d1)
            + rate * strike * rate_discount * norm.cdf(-d2)
        )
        rho = -strike * tau * rate_discount * norm.cdf(-d2)

    return Greeks(
        delta=delta,
        gamma=gamma,
        vega=vega / PERCENTAGE_POINT,
        theta=theta / DAYS_IN_YEAR,
        rho=rho / PERCENTAGE_POINT,
    )


# d1 = (ln(S/K) + (r - q + sigma**2 / 2) * T) / (sigma * sqrt(T))
# d2 = d1 - sigma * sqrt(T)
def _d1_d2(params: OptionParams) -> tuple[float, float]:
    tau, sigma = params.time_to_expiry, params.sigma
    drift = params.rate - params.div_yield + sigma**2 / 2

    d1 = (log(params.spot / params.strike) + drift * tau) / (sigma * sqrt(tau))
    return d1, d1 - sigma * sqrt(tau)


# Американский стиль молча считать европейским нельзя: цена получилась бы
# заниженной без единого предупреждения, ведь право досрочного исполнения
# стоит неотрицательно. Пусть лучше упадёт с внятной ошибкой.
def _require_european(params: OptionParams) -> None:
    if params.exercise is not ExerciseStyle.EUROPEAN:
        raise AmericanNotSupported(
            "Блэк-Шоулз применим только к европейскому стилю. "
            "Для американского используйте биномиальное дерево."
        )


# Случаи, в которых основная формула делит на ноль.
def _is_degenerate(params: OptionParams) -> bool:
    return params.time_to_expiry <= 0 or params.sigma <= 0


def _degenerate_price(params: OptionParams) -> float:
    # T = 0: опцион истекает прямо сейчас, стоит ровно внутреннюю стоимость.
    if params.time_to_expiry <= 0:
        return params.intrinsic_value

    # sigma = 0: цена актива идёт детерминированно к форварду, выплата
    # известна заранее и просто дисконтируется.
    tau = params.time_to_expiry
    forward = params.spot * exp((params.rate - params.div_yield) * tau)

    payoff = (
        max(forward - params.strike, 0.0)
        if params.is_call
        else max(params.strike - forward, 0.0)
    )
    return exp(-params.rate * tau) * payoff


# В вырожденных случаях все производные по волатильности и времени равны
# нулю или не определены, а delta ступенчатая: единица, если в деньгах.
def _degenerate_greeks(params: OptionParams) -> Greeks:
    in_the_money = params.intrinsic_value > 0
    delta = (1.0 if params.is_call else -1.0) if in_the_money else 0.0

    return Greeks(delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)
