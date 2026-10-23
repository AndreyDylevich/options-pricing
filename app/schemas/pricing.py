"""Pydantic-контракты расчёта: что приходит в запросе и что уходит в ответе.
Здесь же вся проверка входных данных — ни одно значение не доходит до моделей,
не пройдя границы разумного.
"""

from datetime import date, datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.pricing.models import (
    MAX_DIV_YIELD,
    MAX_RATE,
    MAX_SIGMA,
    MIN_RATE,
    ExerciseStyle,
    OptionType,
    PricingModel,
)

Positive = Annotated[float, Field(gt=0)]


class OptionInputs(BaseModel):
    """Параметры опциона без выбора модели."""

    # У pydantic зарезервирован префикс model_; снимаем предупреждение явно.
    model_config = ConfigDict(protected_namespaces=())

    ticker: str = Field(min_length=1, max_length=10)

    spot: Positive = Field(description="Цена базового актива S")
    strike: Positive = Field(description="Страйк K")
    expiry: date = Field(description="Дата экспирации")

    option_type: OptionType
    exercise: ExerciseStyle = ExerciseStyle.EUROPEAN

    sigma: float = Field(gt=0, le=MAX_SIGMA, description="Волатильность, доля единицы")
    rate: float = Field(ge=MIN_RATE, le=MAX_RATE, description="Безрисковая ставка")
    # Потолок высокий намеренно: q приходит пересчитанным под экспирацию,
    # см. MAX_DIV_YIELD.
    div_yield: float = Field(
        ge=0, le=MAX_DIV_YIELD, description="Дивидендная доходность"
    )

    steps: int = Field(default=500, ge=1, le=5000, description="Шагов дерева")
    paths: int = Field(default=100_000, ge=1_000, le=5_000_000, description="Путей МК")

    market_as_of: datetime | None = Field(
        default=None, description="На какой момент актуальны рыночные данные"
    )

    @model_validator(mode="after")
    def check_expiry_in_future(self) -> "OptionInputs":
        if self.expiry <= date.today():
            raise ValueError(
                "Дата экспирации должна быть в будущем: при нулевом сроке "
                "опцион стоит ровно внутреннюю стоимость, а греки не определены"
            )
        return self

    @model_validator(mode="after")
    def check_strike_is_sane(self) -> "OptionInputs":
        # Страйк в сто раз дальше цены — почти наверняка опечатка в разряде.
        ratio = self.strike / self.spot
        if not 0.01 <= ratio <= 100:
            raise ValueError(
                f"Страйк {self.strike:g} отличается от цены {self.spot:g} "
                "более чем в сто раз — проверьте, не опечатка ли это"
            )
        return self


class PriceRequest(OptionInputs):
    """Расчёт одной выбранной моделью."""

    model: PricingModel

    @model_validator(mode="after")
    def check_model_supports_exercise(self) -> "PriceRequest":
        if (
            self.exercise is ExerciseStyle.AMERICAN
            and self.model is not PricingModel.BINOMIAL
        ):
            raise ValueError(
                "Американский стиль умеет считать только биномиальное дерево. "
                "Выберите его или переключитесь на европейский стиль"
            )
        return self


class CompareRequest(OptionInputs):
    """Расчёт всеми моделями сразу."""


class GreeksOut(BaseModel):
    """Греки в терминальных единицах."""

    delta: float = Field(description="На 1 единицу цены базового актива")
    gamma: float = Field(description="Изменение delta на 1 единицу цены")
    vega: float = Field(description="На 1 процентный пункт волатильности")
    theta: float = Field(description="На 1 календарный день")
    rho: float = Field(description="На 1 процентный пункт ставки")


class PriceMeta(BaseModel):
    """Как считали: полезно для сравнения моделей между собой."""

    model_config = ConfigDict(protected_namespaces=())

    model: PricingModel
    exercise: ExerciseStyle
    elapsed_ms: float
    days_to_expiry: int
    time_to_expiry: float = Field(description="Срок в годах, T")

    std_error: float | None = Field(default=None, description="Только Монте-Карло")
    ci95: tuple[float, float] | None = Field(default=None, description="Только Монте-Карло")

    warning: str | None = Field(
        default=None,
        description="Расчёт прошёл, но результату доверять нельзя",
    )


class PriceResponse(BaseModel):
    price: float
    greeks: GreeksOut
    meta: PriceMeta

    calculation_id: int | None = None


class ModelComparison(BaseModel):
    """Строка таблицы сравнения: либо результат, либо причина отказа."""

    model_config = ConfigDict(protected_namespaces=())

    model: PricingModel
    price: float | None = None
    greeks: GreeksOut | None = None
    elapsed_ms: float | None = None

    std_error: float | None = None
    ci95: tuple[float, float] | None = None
    warning: str | None = None

    error: str | None = None


class CompareResponse(BaseModel):
    """Все модели на одних и тех же входных данных."""

    exercise: ExerciseStyle
    days_to_expiry: int
    time_to_expiry: float
    results: list[ModelComparison]


class TreeConvergencePoint(BaseModel):
    """Одна точка на графике сходимости дерева."""

    steps: int
    price: float
    error: float = Field(description="Отклонение от точного значения")
    elapsed_ms: float


class MonteCarloConvergencePoint(BaseModel):
    """Одна точка на графике сходимости Монте-Карло."""

    paths: int
    price: float
    error: float
    std_error: float
    ci95_low: float
    ci95_high: float
    elapsed_ms: float


class ConvergenceReport(BaseModel):
    """Как две приближённые модели подходят к точному значению."""

    model_config = ConfigDict(protected_namespaces=())

    reference: float = Field(description="Точная цена по Блэку-Шоулзу")
    days_to_expiry: int
    time_to_expiry: float

    tree: list[TreeConvergencePoint]
    monte_carlo: list[MonteCarloConvergencePoint]

    skipped_steps: list[int] = Field(
        default_factory=list,
        description="Число шагов, при котором дерево отказалось считать",
    )


# Массивами, а не списком точек: Plotly принимает именно массивы.
class GreekCurve(BaseModel):
    """Одна кривая профиля: как грек меняется вдоль одного параметра."""

    x: list[float]
    y: list[float]

    current_x: float = Field(description="Где на кривой текущие параметры")
    current_y: float = Field(description="И значение грека в этой точке")


class GreekProfiles(BaseModel):
    """Профили греков вокруг заданных параметров."""

    model_config = ConfigDict(protected_namespaces=())

    model: PricingModel = Field(description="Чем считались кривые")
    exercise: ExerciseStyle
    strike: float
    days_to_expiry: int

    delta_vs_spot: GreekCurve
    gamma_vs_spot: GreekCurve
    theta_vs_days: GreekCurve = Field(description="По дням до экспирации")
