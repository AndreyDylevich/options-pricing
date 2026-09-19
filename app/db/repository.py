"""Чтение и запись справочников, рыночных снимков и истории расчётов.

Отделено от pricing_service намеренно: расчёт не должен зависеть от того,
доступна ли база. Если запись упадёт, цена всё равно вернётся пользователю.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.entities import Calculation, Instrument, MarketSnapshot
from app.schemas.instruments import Instrument as InstrumentSchema
from app.schemas.instruments import MarketSnapshot as MarketSnapshotSchema
from app.schemas.pricing import PriceRequest, PriceResponse


async def save_calculation(
    session: AsyncSession, request: PriceRequest, response: PriceResponse
) -> int:
    """Записать расчёт, вернуть его идентификатор."""
    meta, greeks = response.meta, response.greeks

    row = Calculation(
        ticker=request.ticker,
        model=meta.model.value,
        option_type=request.option_type.value,
        exercise=request.exercise.value,
        spot=request.spot,
        strike=request.strike,
        expiry=request.expiry,
        days_to_expiry=meta.days_to_expiry,
        time_to_expiry=meta.time_to_expiry,
        rate=request.rate,
        div_yield=request.div_yield,
        sigma=request.sigma,
        model_params=_model_params(request),
        market_as_of=request.market_as_of,
        price=response.price,
        delta=greeks.delta,
        gamma=greeks.gamma,
        vega=greeks.vega,
        theta=greeks.theta,
        rho=greeks.rho,
        std_error=meta.std_error,
        ci95_low=meta.ci95[0] if meta.ci95 else None,
        ci95_high=meta.ci95[1] if meta.ci95 else None,
        elapsed_ms=meta.elapsed_ms,
    )

    session.add(row)
    await session.commit()
    return row.id


async def purge_old_calculations(
    session: AsyncSession, retention_days: int | None = None
) -> int:
    """Удалить расчёты старше срока хранения. Возвращает число удалённых.

    История наружу не отдаётся — её смотрят прямо в PostgreSQL, и держать
    там месяцы отладочных прогонов незачем: таблица растёт от каждого
    нажатия «Рассчитать» и от каждого прогона тестов.

    Отсчёт ведём от текущего момента в UTC, как и `created_at`: возьми мы
    naive-время, сравнение с колонкой timestamptz упало бы.
    """
    days = (
        settings.calculation_retention_days if retention_days is None else retention_days
    )
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    result = await session.execute(
        delete(Calculation).where(Calculation.created_at < cutoff)
    )
    await session.commit()

    return result.rowcount or 0


def _model_params(request: PriceRequest) -> dict:
    """Параметры, относящиеся только к конкретной модели.

    Сохраняем лишь то, что реально повлияло на результат: число шагов не
    имеет смысла в записи о расчёте по Блэку-Шоулзу.
    """
    if request.model.value == "binomial":
        return {"steps": request.steps}
    if request.model.value == "monte_carlo":
        return {"paths": request.paths}
    return {}


async def list_instruments(session: AsyncSession) -> list[InstrumentSchema]:
    """Бумаги с ликвидными опционами, по алфавиту.

    Мало иметь цепочку — по ней должны быть котировки. Порог задаётся
    настройкой MIN_QUOTED_STRIKES: на бумаге, где котируются четыре
    страйка, улыбку волатильности строить не по чему.
    """
    query = (
        select(Instrument)
        .where(
            Instrument.has_options.is_(True),
            Instrument.quoted_strikes >= settings.min_quoted_strikes,
        )
        .order_by(Instrument.ticker)
    )
    rows = (await session.scalars(query)).all()

    return [
        InstrumentSchema(
            ticker=row.ticker,
            name=row.name,
            exchange=row.exchange,
            currency=row.currency,
            has_options=row.has_options,
            quoted_strikes=row.quoted_strikes,
        )
        for row in rows
    ]


async def save_snapshot(
    session: AsyncSession, snapshot: MarketSnapshotSchema
) -> None:
    """Сохранить рыночный снимок.

    Пишется только то, что реально пришло от источника: повторно класть
    отданное из кэша значит засорять таблицу копиями одной и той же
    записи.
    """
    session.add(
        MarketSnapshot(
            ticker=snapshot.ticker,
            fetched_at=snapshot.fetched_at,
            as_of=snapshot.as_of,
            spot=snapshot.spot,
            sigma_30d=snapshot.sigma_30d,
            sigma_90d=snapshot.sigma_90d,
            sigma_252d=snapshot.sigma_252d,
            div_yield=snapshot.div_yield,
            risk_free_rate=snapshot.risk_free_rate,
            expirations=snapshot.expirations,
            div_schedule=[
                payment.model_dump(mode="json") for payment in snapshot.div_schedule
            ],
        )
    )
    await session.commit()


async def latest_snapshots(
    session: AsyncSession,
) -> list[MarketSnapshotSchema]:
    """По одному последнему снимку на каждую бумагу.

    Нужен при старте: кэш в памяти после перезапуска пуст, и без этого
    первый же выбор любой акции снова шёл бы в сеть.
    """
    newest = (
        select(
            MarketSnapshot.ticker,
            func.max(MarketSnapshot.fetched_at).label("fetched_at"),
        )
        .group_by(MarketSnapshot.ticker)
        .subquery()
    )

    query = (
        select(MarketSnapshot, Instrument)
        .join(
            newest,
            (MarketSnapshot.ticker == newest.c.ticker)
            & (MarketSnapshot.fetched_at == newest.c.fetched_at),
        )
        .join(Instrument, Instrument.ticker == MarketSnapshot.ticker)
    )

    return [
        MarketSnapshotSchema(
            ticker=row.ticker,
            name=instrument.name,
            currency=instrument.currency,
            spot=row.spot,
            sigma_30d=row.sigma_30d,
            sigma_90d=row.sigma_90d,
            sigma_252d=row.sigma_252d,
            div_yield=row.div_yield,
            risk_free_rate=row.risk_free_rate,
            expirations=row.expirations,
            div_schedule=row.div_schedule or [],
            as_of=row.as_of,
            fetched_at=row.fetched_at,
        )
        for row, instrument in (await session.execute(query)).all()
    ]


async def sync_instruments(
    session: AsyncSession, items: list[InstrumentSchema], replace: bool = False
) -> dict[str, int]:
    """Записать список бумаг, вернуть сводку.

    По умолчанию дополняет: скринер отдаёт самые ликвидные бумаги дня, и
    затирать ими весь список значит терять то, что было вчера. Флаг
    replace нужен, чтобы избавиться от накопившегося и от бумаг,
    прописанных в первой миграции.
    """
    known_before = set(
        (await session.scalars(select(Instrument.ticker))).all()
    )

    if replace:
        keep = {item.ticker for item in items}
        await session.execute(
            delete(Instrument).where(Instrument.ticker.notin_(keep))
        )

    now = datetime.now(timezone.utc)
    statement = insert(Instrument).values(
        [
            {
                "ticker": item.ticker,
                "name": item.name,
                "exchange": item.exchange,
                "currency": item.currency,
                "has_options": item.has_options,
                "quoted_strikes": item.quoted_strikes,
                "synced_at": now,
            }
            for item in items
        ]
    )
    # Тикер — первичный ключ, поэтому повторная синхронизация не падает, а
    # освежает название, биржу и отметку времени.
    await session.execute(
        statement.on_conflict_do_update(
            index_elements=["ticker"],
            set_={
                "name": statement.excluded.name,
                "exchange": statement.excluded.exchange,
                "currency": statement.excluded.currency,
                "has_options": statement.excluded.has_options,
                "quoted_strikes": statement.excluded.quoted_strikes,
                "synced_at": statement.excluded.synced_at,
            },
        )
    )
    await session.commit()

    fetched = {item.ticker for item in items}
    return {
        "fetched": len(items),
        "added": len(fetched - known_before),
        "updated": len(fetched & known_before),
        "removed": len(known_before - fetched) if replace else 0,
        "without_options": sum(1 for item in items if not item.has_options),
        "illiquid": sum(
            1
            for item in items
            if item.has_options and item.quoted_strikes < settings.min_quoted_strikes
        ),
    }
