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


class Instrument(Base):

    __tablename__ = "instruments"

    ticker: Mapped[str] = mapped_column(String(10), primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    exchange: Mapped[str] = mapped_column(String(16))
    currency: Mapped[str] = mapped_column(String(3))

    has_options: Mapped[bool] = mapped_column(Boolean, default=True)

    # NULL значит «не измерено», и это не ноль: пока биржа закрыта, Yahoo
    # отдаёт нулевые bid и ask по всем страйкам подряд.
    quoted_strikes: Mapped[int | None] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


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

    expirations: Mapped[list] = mapped_column(JSONB, default=list)

    # Хранится график выплат, а не готовое q: из графика q выводится
    # однозначно, а обратно — уже нет.
    div_schedule: Mapped[list] = mapped_column(JSONB, default=list)

    __table_args__ = (
        Index("ix_market_snapshots_ticker_fetched", "ticker", fetched_at.desc()),
    )


# Греки отдельными колонками, а не одним JSON: по ним нужно сортировать и
# строить графики. В JSON уходит только то, у чего нет общего набора полей.
class Calculation(Base):

    __tablename__ = "calculations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    ticker: Mapped[str] = mapped_column(String(10))
    model: Mapped[str] = mapped_column(String(32))
    option_type: Mapped[str] = mapped_column(String(8))
    exercise: Mapped[str] = mapped_column(String(16))

    spot: Mapped[float]
    strike: Mapped[float]
    expiry: Mapped[date] = mapped_column(Date)
    days_to_expiry: Mapped[int]
    time_to_expiry: Mapped[float]
    rate: Mapped[float]
    div_yield: Mapped[float]
    sigma: Mapped[float]

    model_params: Mapped[dict] = mapped_column(JSONB, default=dict)

    # Отличается от created_at: посчитать можно сегодня по вчерашнему закрытию.
    market_as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    price: Mapped[float]
    delta: Mapped[float]
    gamma: Mapped[float]
    vega: Mapped[float]
    theta: Mapped[float]
    rho: Mapped[float]

    std_error: Mapped[float | None]
    ci95_low: Mapped[float | None]
    ci95_high: Mapped[float | None]

    elapsed_ms: Mapped[float]

    __table_args__ = (
        Index("ix_calculations_created_at", created_at.desc()),
        Index("ix_calculations_ticker_model", "ticker", "model"),
    )
