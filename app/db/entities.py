"""Таблицы PostgreSQL: справочник бумаг, рыночные снимки и история расчётов.
Описывают схему, ничего не считают.
"""

from datetime import date, datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


# Тикер — первичный ключ: он и так уникален, а по базе удобнее ходить
# глазами, когда во внешних ссылках стоит AAPL, а не число 7.
class Instrument(Base):

    __tablename__ = "instruments"

    ticker: Mapped[str] = mapped_column(String(10), primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    exchange: Mapped[str] = mapped_column(String(16))
    currency: Mapped[str] = mapped_column(String(3))

    # Скринер отбирает бумаги по ликвидности и про опционы ничего не знает,
    # а бумага без цепочки опционов нам бесполезна. Проверяется один раз
    # при синхронизации, а не при каждой загрузке страницы.
    has_options: Mapped[bool] = mapped_column(Boolean, default=True)

    # Сколько страйков месячной цепочки имеют одновременно bid и ask.
    # Наличие опционов ещё не значит, что ими торгуют: у части бумаг
    # цепочка есть, а котировок в ней почти нет.
    quoted_strikes: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# Пишется при каждом обращении к источнику. Две задачи: пережить перезапуск
# приложения, чтобы кэш в памяти не начинал с нуля, и дать воспроизводимость
# — по market_as_of из расчёта видно, какими данными он пользовался.
class MarketSnapshot(Base):

    __tablename__ = "market_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    ticker: Mapped[str] = mapped_column(
        String(10), ForeignKey("instruments.ticker", ondelete="CASCADE")
    )

    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    spot: Mapped[float]
    sigma_30d: Mapped[float | None]
    sigma_90d: Mapped[float | None]
    sigma_252d: Mapped[float | None]
    div_yield: Mapped[float]
    risk_free_rate: Mapped[float]

    # Список дат экспирации: хранить их отдельной таблицей незачем, они
    # меняются целиком вместе со снимком и запросами не разбираются.
    expirations: Mapped[list] = mapped_column(JSONB, default=list)

    # График предстоящих дивидендных отсечек. Хранится он, а не готовое
    # q по экспирациям: q из графика выводится однозначно, а вот обратно
    # — уже нет. Без этой колонки после перезапуска сервера кэш
    # поднимался бы из базы с пустым графиком, и весь расчёт q по датам
    # отсечек молча откатывался бы к трейлингу за 12 месяцев.
    div_schedule: Mapped[list] = mapped_column(JSONB, default=list)

    __table_args__ = (
        # Читается всегда «последний снимок по бумаге».
        Index("ix_market_snapshots_ticker_fetched", "ticker", fetched_at.desc()),
    )


# Греки лежат отдельными колонками, а не одним JSON: по ним нужно строить
# графики и сортировать («покажи delta по всем страйкам»), а доставать это
# из JSON неудобно и медленно. В JSON уходят только параметры конкретных
# моделей, у которых нет общего набора полей: steps у дерева, paths у Монте-
# Карло, и что бы там ни появилось у будущих моделей.
class Calculation(Base):

    __tablename__ = "calculations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # --- что считали ---
    ticker: Mapped[str] = mapped_column(String(10))
    model: Mapped[str] = mapped_column(String(32))
    option_type: Mapped[str] = mapped_column(String(8))
    exercise: Mapped[str] = mapped_column(String(16))

    # --- входные параметры ---
    spot: Mapped[float]
    strike: Mapped[float]
    expiry: Mapped[date] = mapped_column(Date)
    days_to_expiry: Mapped[int]
    time_to_expiry: Mapped[float]
    rate: Mapped[float]
    div_yield: Mapped[float]
    sigma: Mapped[float]

    # Параметры конкретной модели: steps, paths и что появится дальше.
    model_params: Mapped[dict] = mapped_column(JSONB, default=dict)

    # Момент, на который были актуальны рыночные данные. Отличается от
    # created_at: посчитать можно сегодня по вчерашнему закрытию.
    market_as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # --- результат ---
    price: Mapped[float]
    delta: Mapped[float]
    gamma: Mapped[float]
    vega: Mapped[float]
    theta: Mapped[float]
    rho: Mapped[float]

    # Только Монте-Карло, у остальных моделей случайной ошибки нет.
    std_error: Mapped[float | None]
    ci95_low: Mapped[float | None]
    ci95_high: Mapped[float | None]

    elapsed_ms: Mapped[float]

    __table_args__ = (
        # История читается «последние N» — индекс по убыванию времени.
        Index("ix_calculations_created_at", created_at.desc()),
        # И «покажи всё, что считали по этой бумаге этой моделью».
        Index("ix_calculations_ticker_model", "ticker", "model"),
    )
