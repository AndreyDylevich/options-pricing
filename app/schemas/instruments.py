"""Pydantic-контракты справочника и рыночных данных: список бумаг, снимок по
одной, цепочка опционов, котировка контракта, отчёты по паритету и
волатильности.
"""

from datetime import date, datetime

from pydantic import BaseModel, Field

from app.pricing.models import OptionType


class Instrument(BaseModel):
    """Базовый актив в списке выбора."""

    ticker: str = Field(examples=["AAPL"])
    name: str = Field(examples=["Apple"])
    exchange: str = Field(examples=["NASDAQ"])
    currency: str = Field(default="USD", examples=["USD"])
    has_options: bool = Field(
        default=True,
        description="Есть ли по бумаге торгуемые опционы — иначе считать нечего",
    )
    quoted_strikes: int | None = Field(
        default=None,
        description="Страйков месячной цепочки с живыми котировками; "
        "null — не измерено, биржа была закрыта",
    )


class DividendPayment(BaseModel):
    """Одна предстоящая дивидендная отсечка."""

    ex_date: date = Field(
        description="Дата отсечки: с этого дня цена торгуется уже без дивиденда"
    )
    amount: float = Field(description="Выплата на акцию, в валюте бумаги")
    estimated: bool = Field(
        description="Дата не объявлена компанией, а продолжена по ритму прошлых выплат"
    )


# Волатильности — годовые, доли единицы (0.284 = 28.4%). Окна, для которых
# не хватило истории, приходят как null.
class MarketSnapshot(BaseModel):
    """Рыночные данные, из которых собираются параметры модели."""

    ticker: str
    name: str
    currency: str

    spot: float = Field(description="Цена базового актива S")
    sigma_30d: float | None = Field(description="Историческая волатильность, 30 торговых дней")
    sigma_90d: float | None = Field(description="То же, 90 дней")
    sigma_252d: float | None = Field(description="То же, 252 дня (год)")
    div_yield: float = Field(
        description="Дивидендная доходность за 12 мес — справочно, не для расчёта"
    )
    div_schedule: list[DividendPayment] = Field(
        default_factory=list,
        description="Предстоящие отсечки, по которым считается q",
    )
    div_yield_by_expiry: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Эффективное q под каждую дату экспирации, доля единицы. "
            "Именно оно идёт в расчёт: до ближайшей отсечки q равно нулю, "
            "а если отсечка попадает в срок, q заметно выше среднегодового"
        ),
    )
    risk_free_rate: float = Field(description="Безрисковая ставка r из ^IRX")

    expirations: list[str] = Field(description="Доступные даты экспирации, YYYY-MM-DD")

    as_of: datetime = Field(description="Дата последней свечи в истории")
    fetched_at: datetime = Field(description="Когда данные были запрошены у источника")

    cached: bool = Field(
        default=False, description="Ответ отдан из кэша, без обращения к источнику"
    )
    stale: bool = Field(
        default=False,
        description="Кэш просрочен, но источник недоступен — данные устарели",
    )


class OptionQuote(BaseModel):
    """Один контракт из цепочки вместе с посчитанной волатильностью."""

    strike: float
    option_type: OptionType

    bid: float
    ask: float
    mid: float = Field(description="Середина bid-ask: по ней и решается обратная задача")

    volume: int | None = None
    open_interest: int | None = None

    implied_vol: float | None = Field(
        default=None, description="Наша подразумеваемая волатильность, доля единицы"
    )
    iv_note: str | None = Field(
        default=None, description="Почему волатильность не определилась"
    )
    yahoo_iv: float | None = Field(
        default=None, description="Что по этому же контракту отдаёт Yahoo"
    )

    log_moneyness: float = Field(
        description="ln(K / F) — ось, на которой строится улыбка"
    )


# Параметры, при которых считалась волатильность, лежат рядом с
# котировками: без них числа невоспроизводимы.
class OptionChain(BaseModel):
    """Цепочка опционов на одну дату экспирации."""

    ticker: str
    expiry: str
    days_to_expiry: int
    time_to_expiry: float

    spot: float
    rate: float
    div_yield: float = Field(description="Эффективное q под эту экспирацию")
    forward: float = Field(description="S * exp((r - q) * T)")

    quotes: list[OptionQuote]

    quoted: int = Field(description="Контрактов с живыми bid и ask")
    skipped: int = Field(description="Контрактов без котировок — отброшены")
    solved: int = Field(description="Для скольких волатильность нашлась")


class ParityRow(BaseModel):
    """Сверка паритета по одному страйку."""

    strike: float
    call_mid: float
    put_mid: float

    observed: float = Field(description="C - P по рыночным котировкам")
    expected: float = Field(description="S*exp(-qT) - K*exp(-rT) по нашим данным")
    residual: float = Field(description="Расхождение: observed - expected")
    tolerance: float = Field(description="Сколько из него объяснимо шириной спредов")
    within_quotes: bool = Field(description="Укладывается ли расхождение в спреды")

    implied_forward: float = Field(description="Форвард, вытащенный из этой пары")
    implied_div_yield: float | None = Field(
        default=None, description="И дивидендная доходность из него же"
    )

    arbitrage: str | None = Field(
        default=None,
        description="Котировки вне границ паритета — данные битые или это арбитраж",
    )


# Проверяются не модели, а то, что в них подставляется: спот, ставка и
# дивидендная доходность. Волатильность в паритет не входит.
class ParityReport(BaseModel):
    """Сверка наших входных данных с рынком по паритету колла и пута."""

    ticker: str
    expiry: str
    days_to_expiry: int
    time_to_expiry: float

    spot: float
    rate: float

    div_yield_ours: float = Field(description="Наше q, посчитанное по датам отсечек")
    div_yield_market: float | None = Field(
        default=None, description="q, каким его видит рынок через форвард"
    )

    forward_ours: float = Field(description="S*exp((r-q)T) на нашем q")
    forward_market: float | None = Field(
        default=None, description="Медиана форвардов по страйкам у денег"
    )
    forward_tolerance: float | None = Field(
        default=None, description="Неопределённость рыночного форварда от спредов"
    )

    pairs: int = Field(description="Страйков, где есть и колл, и пут с котировками")
    agreed: int = Field(description="Из них сошлись в пределах спредов")
    arbitrage: int = Field(description="Из них вышли за границы паритета")

    verdict: str = Field(description="Вывод по-русски: сходится или нет и насколько")

    rows: list[ParityRow]


# Цена последней сделки сюда намеренно не входит: сделка могла состояться
# неделю назад при другой цене актива, и сравнивать с ней расчёт
# бессмысленно. По той же причине и волатильность в проекте считается от
# середины спреда.
class ContractQuote(BaseModel):
    """Рыночная цена конкретного контракта — то, за что его торгуют сейчас."""

    ticker: str
    expiry: str
    strike: float
    option_type: OptionType

    bid: float
    ask: float
    mid: float = Field(description="Середина спреда — с ней и сравнивается расчёт")
    spread: float = Field(description="ask - bid, ширина неопределённости рынка")

    volume: int | None = Field(default=None, description="Сделок за день")
    open_interest: int | None = Field(default=None, description="Открытых позиций")


class VolatilityOption(BaseModel):
    """Одна строка таблицы выбора: чем посчитана σ и сколько получилось."""

    key: str = Field(description="Идентификатор для кнопки выбора")
    label: str = Field(description="Название способа по-русски")
    sigma: float = Field(description="Годовая волатильность, доля единицы")
    note: str = Field(description="Чем этот способ отличается от остальных")


# Нужна не для выбора, а для понимания: по коэффициентам видно, почему
# прогноз получился таким, а по AIC — какая из двух моделей лучше
# описывает этот конкретный ряд.
class VolatilityModelFit(BaseModel):
    """Подогнанная модель: параметры и качество."""

    name: str
    params: dict[str, float]
    log_likelihood: float
    aic: float = Field(description="Критерий Акаике: чем меньше, тем лучше")

    next_day: float = Field(description="Прогноз на завтра, годовая доля")
    term: float = Field(description="Средняя по сроку опциона — она и идёт в цену")


# Историческая смотрит назад одним окном и считает все дни равноценными.
# GARCH и EGARCH смотрят туда же, но учитывают, что спокойные и бурные
# периоды идут сгустками, и дают прогноз на срок опциона. Подразумеваемой
# здесь нет: подставив её в модель, получаешь рыночную цену обратно — она
# из этой цены и выведена. Её место в цепочке опционов, рядом с улыбкой.
class VolatilityReport(BaseModel):
    """Оценки волатильности, посчитанные по истории цен."""

    ticker: str
    expiry: str
    days_to_expiry: int
    observations: int = Field(description="Дней истории, на которых подгонялись модели")

    options: list[VolatilityOption]
    fits: list[VolatilityModelFit]

    warning: str | None = Field(
        default=None, description="Что из таблицы не удалось посчитать и почему"
    )
