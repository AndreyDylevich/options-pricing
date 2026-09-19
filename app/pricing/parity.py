"""Паритет колла и пута: сверка наших входных данных с рынком.

Единственная проверка в проекте, которая смотрит наружу. Все остальные
внутренние: дерево сходится к Блэку-Шоулзу, численные греки совпадают с
аналитическими, цена лежит в границах отсутствия арбитража. Они отвечают
на вопрос «правильно ли посчитана формула», но молчат о том, верны ли
числа, которые мы в неё подставили.

Паритет отвечает именно на это. Для колла и пута на одном страйке и
сроке рынок обязан держать

    C - P = S * exp(-q * T) - K * exp(-r * T)

Слева стоят чистые рыночные цены, никакой модели. Справа — наши спот,
ставка и дивидендная доходность. Не сходится — значит ошибка во входных
данных: устарел спот, промахнулись с дивидендом, взяли не ту ставку.

Волатильность в равенство не входит вовсе, поэтому проверка не зависит
от того, как мы её оцениваем, и не может выродиться в тавтологию — в
отличие от сравнения нашей цены с рыночной через подразумеваемую
волатильность, которая из этой же цены и получена.

Оговорка про американский стиль: равенство доказано для европейских
опционов, а на американские акции торгуются американские. Для них верно
только неравенство, оно проверяется отдельно в `_arbitrage`. Само
равенство остаётся хорошим приближением у денег, где премия за досрочное
исполнение почти нулевая, — поэтому итоговый форвард и собирается по
страйкам у денег.
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
    """Котировка одного контракта: то, по чему реально можно торговать.

    Цена последней сделки сюда намеренно не входит — сделка могла
    состояться неделю назад при другой цене актива, и паритет по ней не
    обязан выполняться.
    """

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
        """На сколько середина может отстоять от любой исполнимой цены."""
        return (self.ask - self.bid) / 2


@dataclass(frozen=True, slots=True)
class ParityCheck:
    """Результат сверки по одному страйку."""

    strike: float
    call_mid: float
    put_mid: float

    observed: float = 0.0
    """C - P по рынку."""

    expected: float = 0.0
    """S * exp(-qT) - K * exp(-rT) по нашим данным."""

    residual: float = 0.0
    """Расхождение. Ради него всё и затевалось."""

    tolerance: float = 0.0
    """Насколько расхождение объяснимо шириной спредов."""

    implied_forward: float = 0.0
    """Форвард, каким его видит рынок на этом страйке."""

    forward_tolerance: float = 0.0
    """Та же ширина спредов, пересчитанная в неопределённость форварда."""

    implied_div_yield: float | None = None
    """Дивидендная доходность, вытащенная из того же форварда."""

    arbitrage: str | None = None
    """Нарушены границы для американского стиля — это уже не про нас."""

    @property
    def within_quotes(self) -> bool:
        """Можно ли подобрать цены внутри спредов, точно дающие паритет.

        Если да, рынок с нашими данными согласен: расхождение середин
        целиком объясняется тем, что середина — это не цена, а просто
        точка между двумя ценами.
        """
        return abs(self.residual) <= self.tolerance


def implied_forward(
    strike: float, call_mid: float, put_mid: float, rate: float, tau: float
) -> float:
    """Форвардная цена актива, вытащенная из пары колл-пут.

    Из паритета C - P = exp(-rT) * (F - K) сразу следует
    F = K + (C - P) * exp(rT). Ни волатильности, ни дивидендов здесь нет:
    рынок сам говорит, где он видит актив на дату экспирации.
    """
    return strike + (call_mid - put_mid) * math.exp(rate * tau)


def implied_div_yield(
    forward: float, spot: float, rate: float, tau: float
) -> float | None:
    """Дивидендная доходность, при которой наш спот даёт рыночный форвард.

    Обращение F = S * exp((r - q) * T). Это второй, независимый способ
    получить q: не по датам отсечек из истории, а по тому, сколько за
    дивиденды платит рынок. Расхождение двух оценок — прямая мера того,
    насколько наш график выплат близок к действительности.
    """
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
    """Сверка одного страйка: что обещает паритет против того, что на рынке."""
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
    """Нарушение границ паритета для американского стиля.

    Равенство держится только у европейских опционов; у американских
    досрочное исполнение растягивает его в неравенство

        S * exp(-qT) - K  <=  C - P  <=  S - K * exp(-rT)

    Выход за эти границы моделью не объясняется никак: это либо разрыв в
    котировках, либо остаток досплитной серии, либо настоящая
    возможность арбитража, которую рынок закроет за секунды.

    Считаем по исполнимым ценам, а не по серединам: чтобы получить
    разность C - P сверху, колл надо продать по биду, а пут купить по
    аску. По серединам «нарушение» нашлось бы там, где заработать нечего.
    """
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
    """Форвард по страйкам у денег и его неопределённость от спредов.

    «У денег» здесь определяется без единого нашего числа: из равенства
    C - P = exp(-rT) * (F - K) видно, что разность цен обращается в ноль
    ровно на форварде. Значит страйки с наименьшим |C - P| и есть
    ближайшие к нему — какой бы он ни был. Отбирать их по нашему
    расчётному форварду было бы подгонкой: проверка опиралась бы на
    проверяемое.

    Берём медиану, а не среднее: одна пустая котировка в цепочке
    встречается регулярно, и сдвигать ей весь ответ незачем. Она же
    заменяет отбраковку по признаку выхода за границы — отбрасывать по
    нему нельзя. Границы считаются от нашего спота, форвард от него не
    зависит совсем, и стоит споту разойтись с моментом котировок, как за
    границы уходит вся цепочка разом. Отбраковка убрала бы тогда именно
    то число, по которому этот разрыв и виден.
    """
    if not checks:
        return None

    nearest = sorted(checks, key=lambda item: abs(item.observed))[:count]

    forward = median(item.implied_forward for item in nearest)
    spread = median(item.forward_tolerance for item in nearest)

    return forward, spread
