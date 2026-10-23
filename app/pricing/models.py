"""Общий контракт всех моделей: параметры опциона на входе, цена и греки на
выходе. Ничего не считает сам — задаёт типы, единицы измерения и проверки,
которыми пользуются остальные модули пакета.
"""

from dataclasses import dataclass
from enum import StrEnum

# Защита от опечатки в форме: 2800% волатильности почти наверняка значит,
# что ввели 28 вместо 0.28.
MAX_SIGMA = 5.0
MAX_RATE = 1.0
MIN_RATE = -0.5

# q — пересчёт дискретных выплат в ставку: квартальный дивиденд, попавший
# в двухнедельный опцион, даёт под 150% годовых, и это верное число.
MAX_DIV_YIELD = 5.0


class OptionType(StrEnum):
    CALL = "call"
    PUT = "put"


class ExerciseStyle(StrEnum):
    EUROPEAN = "european"
    AMERICAN = "american"


class PricingModel(StrEnum):
    BLACK_SCHOLES = "black_scholes"
    BINOMIAL = "binomial"
    MONTE_CARLO = "monte_carlo"


class AmericanNotSupported(Exception):
    pass


class PriceOutOfBounds(Exception):
    pass


@dataclass(frozen=True, slots=True)
class OptionParams:
    # Ставки и волатильность — доли единицы (0.0412, не 4.12), срок — в годах.
    spot: float
    strike: float
    time_to_expiry: float
    rate: float
    div_yield: float
    sigma: float
    option_type: OptionType
    exercise: ExerciseStyle = ExerciseStyle.EUROPEAN

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError("Цена базового актива должна быть больше нуля")
        if self.strike <= 0:
            raise ValueError("Страйк должен быть больше нуля")
        if self.time_to_expiry < 0:
            raise ValueError("Срок до экспирации не может быть отрицательным")
        if self.sigma < 0:
            raise ValueError("Волатильность не может быть отрицательной")
        if self.sigma > MAX_SIGMA:
            raise ValueError(
                f"Волатильность {self.sigma:.2f} выглядит опечаткой: "
                f"ожидается доля единицы не больше {MAX_SIGMA}"
            )
        if not MIN_RATE <= self.rate <= MAX_RATE:
            raise ValueError(
                f"Безрисковая ставка {self.rate:.2f} вне разумного диапазона "
                f"[{MIN_RATE}, {MAX_RATE}]"
            )
        if self.div_yield < 0:
            raise ValueError("Дивидендная доходность не может быть отрицательной")
        if self.div_yield > MAX_DIV_YIELD:
            raise ValueError(
                f"Дивидендная доходность {self.div_yield:.2f} выглядит опечаткой: "
                f"ожидается доля единицы не больше {MAX_DIV_YIELD}"
            )

    @property
    def is_call(self) -> bool:
        return self.option_type is OptionType.CALL

    @property
    def intrinsic_value(self) -> float:
        if self.is_call:
            return max(self.spot - self.strike, 0.0)
        return max(self.strike - self.spot, 0.0)


@dataclass(frozen=True, slots=True)
class Greeks:
    # Единицы как в терминалах, а не сырые производные: vega и rho на
    # процентный пункт, theta на календарный день.
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


@dataclass(frozen=True, slots=True)
class PricingResult:
    price: float
    greeks: Greeks
    model: PricingModel
    elapsed_ms: float

    std_error: float | None = None
    ci95: tuple[float, float] | None = None

    warning: str | None = None
