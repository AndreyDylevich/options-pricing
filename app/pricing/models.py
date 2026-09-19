"""Общие типы для всех моделей ценообразования.

Пакет app.pricing не зависит ни от FastAPI, ни от БД, ни от источника
рыночных данных: на вход датакласс с параметрами, на выход цена и греки.
"""

from dataclasses import dataclass
from enum import StrEnum

# Верхние границы санитарной проверки. Не запрет на экзотику, а защита от
# опечатки в форме: 2800% волатильности почти наверняка означает, что
# пользователь ввёл 28 вместо 0.28, перепутав проценты с долями.
MAX_SIGMA = 5.0
MAX_RATE = 1.0
MIN_RATE = -0.5

# Дивидендной доходности потолок нужен выше, чем кажется. q здесь — не
# рыночная величина, а пересчёт дискретных выплат в непрерывную ставку:
# квартальный дивиденд, попавший в двухнедельный опцион, даёт под 150%
# годовых, и это правильное число, а не опечатка.
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
    """Модель умеет только европейский стиль.

    Общая для Блэка-Шоулза и Монте-Карло: обе идут вперёд во времени и не
    могут решить, выгодно ли исполнить опцион досрочно.
    """


class PriceOutOfBounds(Exception):
    """Цена нарушила границы отсутствия арбитража.

    В отличие от остальных исключений это не ошибка пользователя, а
    признак ошибки в самой модели: никакие входные данные не могут
    привести к цене выше потолка или ниже пола.
    """


@dataclass(frozen=True, slots=True)
class OptionParams:
    """Полный набор входных данных для любой из трёх моделей.

    Ставки и волатильность — доли единицы (0.0412, а не 4.12), срок — в
    годах. Перевод из процентов и дат делает слой выше.
    """

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
        """Стоимость немедленного исполнения. Нижняя граница цены."""
        if self.is_call:
            return max(self.spot - self.strike, 0.0)
        return max(self.strike - self.spot, 0.0)


@dataclass(frozen=True, slots=True)
class Greeks:
    """Чувствительности цены опциона.

    Единицы как в торговых терминалах, а не сырые производные:
      delta — на 1 единицу цены базового актива
      gamma — изменение delta на 1 единицу цены
      vega  — на 1 процентный пункт волатильности
      theta — на 1 календарный день
      rho   — на 1 процентный пункт ставки
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


@dataclass(frozen=True, slots=True)
class PricingResult:
    """Что возвращает любая из моделей.

    std_error и ci95 заполняет только Монте-Карло: у аналитических и
    решёточных методов случайной ошибки нет.
    """

    price: float
    greeks: Greeks
    model: PricingModel
    elapsed_ms: float
    std_error: float | None = None
    ci95: tuple[float, float] | None = None

    # Расчёт прошёл, но доверять ему не стоит. Заполняет Монте-Карло,
    # когда случайная ошибка велика относительно самой цены.
    warning: str | None = None
