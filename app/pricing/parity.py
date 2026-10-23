"""Паритет колла и пута: по рыночным ценам пары на одном страйке считаем
форвард и дивидендную доходность и сверяем их с нашими. Единственная
проверка в проекте, которая проверяет входные данные, а не формулы.
"""

import math
from dataclasses import dataclass
from statistics import median

CONSENSUS_STRIKES = 5


@dataclass(frozen=True, slots=True)
class Quote:
    bid: float
    ask: float

    def __post_init__(self) -> None:
        if self.bid <= 0 or self.ask <= 0:
            raise ValueError("Котировка без живых bid и ask не годится для сверки")
        if self.ask < self.bid:
            raise ValueError(f"Аск {self.ask:g} ниже бида {self.bid:g}")

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def half_spread(self) -> float:
        return (self.ask - self.bid) / 2


@dataclass(frozen=True, slots=True)
class ParityCheck:
    strike: float
    call_mid: float
    put_mid: float

    observed: float = 0.0        # C - P по рынку
    expected: float = 0.0        # S*exp(-qT) - K*exp(-rT) по нашим данным
    residual: float = 0.0        # расхождение, ради него всё и затевалось
    tolerance: float = 0.0       # насколько оно объяснимо шириной спредов

    implied_forward: float = 0.0    # форвард, каким его видит рынок
    forward_tolerance: float = 0.0  # та же ширина спредов в единицах форварда
    implied_div_yield: float | None = None   # q из того же форварда

    arbitrage: str | None = None

    @property
    def within_quotes(self) -> bool:
        return abs(self.residual) <= self.tolerance


# C - P = exp(-rT) * (F - K), откуда F = K + (C - P) * exp(rT). Ни
# волатильности, ни дивидендов здесь нет — рынок сам говорит, где видит актив.
def implied_forward(
    strike: float, call_mid: float, put_mid: float, rate: float, tau: float
) -> float:
    return strike + (call_mid - put_mid) * math.exp(rate * tau)


def implied_div_yield(
    forward: float, spot: float, rate: float, tau: float
) -> float | None:
    if tau <= 0 or spot <= 0 or forward <= 0:
        return None
    return rate - math.log(forward / spot) / tau


def check(
    strike: float,
    call: Quote,
    put: Quote,
    spot: float,
    rate: float,
    div_yield: float,
    tau: float,
) -> ParityCheck:
    if strike <= 0:
        raise ValueError("Страйк должен быть больше нуля")
    if spot <= 0:
        raise ValueError("Цена базового актива должна быть больше нуля")
    if tau <= 0:
        raise ValueError("Сверка требует ненулевого срока до экспирации")

    observed = call.mid - put.mid
    expected = spot * math.exp(-div_yield * tau) - strike * math.exp(-rate * tau)

    forward = implied_forward(strike, call.mid, put.mid, rate, tau)

    return ParityCheck(
        strike=strike,
        call_mid=call.mid,
        put_mid=put.mid,
        observed=observed,
        expected=expected,
        residual=observed - expected,
        tolerance=call.half_spread + put.half_spread,
        implied_forward=forward,
        forward_tolerance=(call.half_spread + put.half_spread) * math.exp(rate * tau),
        implied_div_yield=implied_div_yield(forward, spot, rate, tau),
        arbitrage=_arbitrage(strike, call, put, spot, rate, div_yield, tau),
    )


def _arbitrage(
    strike: float,
    call: Quote,
    put: Quote,
    spot: float,
    rate: float,
    div_yield: float,
    tau: float,
) -> str | None:
    #   S*exp(-qT) - K  <=  C - P  <=  S - K*exp(-rT)
    # Считаем по исполнимым ценам, а не по серединам: по серединам
    # «нарушение» нашлось бы там, где заработать нечего.
    lower = spot * math.exp(-div_yield * tau) - strike
    upper = spot - strike * math.exp(-rate * tau)

    if call.bid - put.ask > upper:
        return (
            f"продажа колла с покупкой пута даёт {call.bid - put.ask:.2f} "
            f"при верхней границе {upper:.2f}"
        )

    if call.ask - put.bid < lower:
        return (
            f"покупка колла с продажей пута даёт {call.ask - put.bid:.2f} "
            f"при нижней границе {lower:.2f}"
        )

    return None


def consensus_forward(
    checks: list[ParityCheck], count: int = CONSENSUS_STRIKES
) -> tuple[float, float] | None:
    if not checks:
        return None

        # «У денег» — по наименьшему |C - P|, без единого нашего числа: иначе
        # проверка опиралась бы на проверяемое.
    nearest = sorted(checks, key=lambda item: abs(item.observed))[:count]

    # Медиана, а не среднее. Отбраковывать по выходу за границы нельзя:
    # границы считаются от нашего спота, а форвард от него не зависит совсем.
    forward = median(item.implied_forward for item in nearest)
    spread = median(item.forward_tolerance for item in nearest)

    return forward, spread
