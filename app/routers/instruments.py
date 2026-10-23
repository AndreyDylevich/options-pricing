"""HTTP-слой справочника и рыночных данных: список бумаг, снимок по выбранной,
цепочка опционов, котировка контракта, сверка по паритету и оценки
волатильности. Сам ничего не считает — зовёт market_data и pricing_service.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import repository
from app.db.session import get_session
from app.pricing.models import OptionType
from app.schemas.instruments import (
    ContractQuote,
    Instrument,
    MarketSnapshot,
    OptionChain,
    ParityReport,
    VolatilityReport,
)
from app.security import require_admin_token
from app.services import market_data, pricing_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/instruments", tags=["instruments"])


@router.get("", summary="Список доступных базовых активов")
async def get_instruments() -> list[Instrument]:
    return market_data.list_instruments()


@router.post(
    "/sync",
    summary="Обновить список бумаг из скринера Yahoo",
    description="Дополняет справочник тем, что сейчас отдаёт скринер. Только "
    "добавляет и освежает — удалять через HTTP нечем: замена списка целиком "
    "живёт в команде `python -m scripts.sync_instruments --replace`. "
    "Требует заголовок X-Admin-Token.",
    dependencies=[Depends(require_admin_token)],
)
async def sync_instruments(
    screener: str = Query(
        default=market_data.DEFAULT_SCREENER,
        description="Готовый набор Yahoo: most_actives, undervalued_large_caps и др.",
    ),
    count: int = Query(default=market_data.DEFAULT_SCREENER_COUNT, ge=1, le=250),
    session: AsyncSession = Depends(get_session),
) -> dict:
    # Токен нужен и без удаления: один вызов — полсотни обращений к Yahoo
    # от имени пользователя.
    try:
        items = await run_in_threadpool(
            market_data.fetch_screener_instruments, screener, count
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from None
    except market_data.MarketDataUnavailable as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Скринер недоступен ({error})",
        ) from None

    summary = await repository.sync_instruments(session, items)

    # Реестр в памяти должен сразу отражать новое состояние, иначе список
    # обновится только после перезапуска.
    market_data.set_instruments(await repository.list_instruments(session))
    summary["available"] = len(market_data.list_instruments())

    return summary


@router.get(
    "/{ticker}/chain",
    summary="Цепочка опционов с подразумеваемой волатильностью",
)
async def get_chain(
    ticker: str,
    expiry: str = Query(description="Дата экспирации, YYYY-MM-DD"),
    moneyness_window: float = Query(
        default=pricing_service.DEFAULT_MONEYNESS_WINDOW,
        ge=0,
        le=5,
        description="Ширина окна по ln(K/F); 0 — без фильтра",
    ),
) -> OptionChain:
    try:
        return await run_in_threadpool(
            pricing_service.option_chain, ticker, expiry, moneyness_window
        )
    except market_data.InstrumentNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Инструмент {ticker} не найден в списке доступных",
        ) from None
    except pricing_service.UnknownExpiry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"У {ticker} нет опционов с экспирацией {expiry}",
        ) from None
    except market_data.MarketDataUnavailable as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Источник рыночных данных недоступен ({error})",
        ) from None


@router.get(
    "/{ticker}/quote",
    summary="Рыночная цена конкретного контракта",
    description="За сколько этот опцион торгуется прямо сейчас: bid, ask и "
    "середина спреда. Цена последней сделки не отдаётся — сделка могла быть "
    "неделю назад при другой цене актива.",
)
async def get_quote(
    ticker: str,
    expiry: str = Query(description="Дата экспирации, YYYY-MM-DD"),
    strike: float = Query(gt=0, description="Страйк"),
    option_type: OptionType = Query(description="call или put"),
) -> ContractQuote:
    # Отдельно от расчёта цены намеренно: цепочка тянется из сети сотни
    # миллисекунд, а расчёт занимает единицы.
    try:
        # yfinance ходит в сеть синхронно.
        return await run_in_threadpool(
            pricing_service.contract_quote, ticker, expiry, strike, option_type
        )
    except market_data.InstrumentNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Инструмент {ticker} не найден в списке доступных",
        ) from None
    except pricing_service.UnknownExpiry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"У {ticker} нет опционов с экспирацией {expiry}",
        ) from None
    except pricing_service.ContractNotListed as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)
        ) from None
    except market_data.MarketDataUnavailable as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Источник рыночных данных недоступен ({error})",
        ) from None


@router.get(
    "/{ticker}/parity",
    summary="Сверка наших данных с рынком по паритету колла и пута",
    description="Проверяет не модели, а то, что в них подставляется: спот, "
    "ставку и дивидендную доходность. Волатильность в паритет не входит.",
)
async def get_parity(
    ticker: str,
    expiry: str = Query(description="Дата экспирации, YYYY-MM-DD"),
) -> ParityReport:
    try:
        return await run_in_threadpool(pricing_service.parity_report, ticker, expiry)
    except market_data.InstrumentNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Инструмент {ticker} не найден в списке доступных",
        ) from None
    except pricing_service.UnknownExpiry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"У {ticker} нет опционов с экспирацией {expiry}",
        ) from None
    except market_data.MarketDataUnavailable as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Источник рыночных данных недоступен ({error})",
        ) from None


@router.get("/{ticker}/market", summary="Рыночные данные по базовому активу")
async def get_market(
    ticker: str,
    refresh: bool = Query(
        default=False, description="Обойти кэш и сходить к источнику заново"
    ),
    session: AsyncSession = Depends(get_session),
) -> MarketSnapshot:
    try:
        snapshot = await run_in_threadpool(
            market_data.get_market_snapshot, ticker, refresh
        )
    except market_data.InstrumentNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Инструмент {ticker} не найден в списке доступных",
        ) from None
    except market_data.MarketDataUnavailable as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Источник рыночных данных недоступен ({error})",
        ) from None

    if not snapshot.cached:
        try:
            await repository.save_snapshot(session, snapshot)
        except Exception:
            logger.exception("Не удалось сохранить рыночный снимок")

    return snapshot


@router.get(
    "/{ticker}/volatility",
    summary="Все способы оценить волатильность для выбранного срока",
    description="Историческая, GARCH и EGARCH в одной таблице. Горизонт "
    "прогноза привязан к экспирации: цена опциона зависит от накопленной "
    "дисперсии до его конца, и прогноз на завтра дал бы систематически "
    "неверную цену.",
)
async def get_volatility(
    ticker: str,
    expiry: str = Query(description="Дата экспирации, YYYY-MM-DD"),
) -> VolatilityReport:
    try:
        return await run_in_threadpool(
            pricing_service.volatility_report, ticker, expiry
        )
    except market_data.InstrumentNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Инструмент {ticker} не найден в списке доступных",
        ) from None
    except pricing_service.UnknownExpiry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"У {ticker} нет опционов с экспирацией {expiry}",
        ) from None
    except market_data.MarketDataUnavailable as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Источник рыночных данных недоступен ({error})",
        ) from None
