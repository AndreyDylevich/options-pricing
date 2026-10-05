"""Паритет колла и пута: по рыночным ценам пары на одном страйке считаем
форвард и дивидендную доходность и сверяем их с нашими. Единственная
проверка в проекте, которая проверяет входные данные, а не формулы.
"""

import math
from dataclasses import dataclass
from statistics import median

# Сколько страйков у денег берётся на итоговый форвард. Пять — это
# компромисс: одного мало, одна битая котировка испортила бы ответ, а
# широкий захват тянет в выборку дальние страйки, где спред шире
# временной стоимости и премия за досрочное исполнение уже заметна.
CONSENSUS_STRIKES = 5


@dataclass(frozen=True, slots=True)
class Quote:
    # Только bid и ask, то есть цены, по которым реально можно торговать.
    # Цена последней сделки сюда намеренно не входит: сделка могла
    # состояться неделю назад при другой цене актива, и паритет по ней
    # выполняться не обязан.
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

    # На сколько середина может отстоять от любой исполнимой цены.
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

    # Нарушены границы для американского стиля — это уже не про нас.
    arbitrage: str | None = None

    # Можно ли подобрать цены внутри спредов, точно дающие паритет. Если
    # да, рынок с нашими данными согласен: расхождение середин целиком
    # объясняется тем, что середина — не цена, а точка между двумя ценами.
    @property
    def within_quotes(self) -> bool:
        return abs(self.residual) <= self.tolerance


# Форвард из пары колл-пут. Паритет даёт C - P = exp(-rT) * (F - K), откуда
# F = K + (C - P) * exp(rT). Ни волатильности, ни дивидендов здесь нет:
# рынок сам говорит, где он видит актив на дату экспирации.
def implied_forward(
    strike: float, call_mid: float, put_mid: float, rate: float, tau: float
) -> float:
    return strike + (call_mid - put_mid) * math.exp(rate * tau)


# Обращение F = S * exp((r - q) * T) — второй, независимый способ получить
# q: не по датам отсечек из истории, а по тому, сколько за дивиденды платит
# рынок. Расхождение двух оценок прямо измеряет, насколько наш график
# выплат близок к действительности.
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
    # Равенство C - P = S*exp(-qT) - K*exp(-rT) доказано для европейских
    # опционов, а на американские акции торгуются американские. Для них
    # верно только неравенство — его проверяет _arbitrage. Само равенство
    # остаётся хорошим приближением у денег, где премия за досрочное
    # исполнение почти нулевая, поэтому итоговый форвард и собирается по
    # страйкам у денег.
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
        # Середина гуляет внутри своего спреда у обоих контрактов, и
        # ошибки складываются: разность C - P известна с точностью до
        # суммы полуспредов.
        tolerance=call.half_spread + put.half_spread,
        implied_forward=forward,
        # F = K + (C - P) * exp(rT): страйк точен, вся неопределённость
        # приходит из разности цен и растёт вместе с ней.
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
    # Для американского стиля паритет растягивается в неравенство:
    #   S*exp(-qT) - K  <=  C - P  <=  S - K*exp(-rT)
    #
    # Выход за эти границы моделью не объясняется никак: это либо разрыв в
    # котировках, либо остаток досплитной серии, либо настоящий арбитраж,
    # который рынок закроет за секунды.
    #
    # Считаем по исполнимым ценам, а не по серединам: чтобы получить
    # разность C - P сверху, колл надо продать по биду, а пут купить по
    # аску. По серединам «нарушение» нашлось бы там, где заработать нечего.
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

    # «У денег» определяется без единого нашего числа: из равенства
    # C - P = exp(-rT) * (F - K) видно, что разность цен обращается в ноль
    # ровно на форварде, значит страйки с наименьшим |C - P| и есть
    # ближайшие к нему. Отбирать их по нашему расчётному форварду было бы
    # подгонкой: проверка опиралась бы на проверяемое.
    nearest = sorted(checks, key=lambda item: abs(item.observed))[:count]

    # Медиана, а не среднее: одна пустая котировка в цепочке встречается
    # регулярно, и сдвигать ей весь ответ незачем. Она же заменяет
    # отбраковку по выходу за границы — отбрасывать по нему нельзя.
    # Границы считаются от нашего спота, форвард от него не зависит
    # совсем, и стоит споту разойтись с моментом котировок, как за границы
    # уходит вся цепочка разом; отбраковка убрала бы ровно то число, по
    # которому этот разрыв и виден.
    forward = median(item.implied_forward for item in nearest)
    spread = median(item.forward_tolerance for item in nearest)

    return forward, spread
