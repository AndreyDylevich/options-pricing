"""POST /api/price — расчёт цены и греков выбранной моделью.

Каждый расчёт попадает в таблицу `calculations`, но наружу история не
отдаётся: смотреть её полагается прямо в PostgreSQL. Старые записи
подчищает `main.py`, см. `repository.purge_old_calculations`.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import repository
from app.db.session import get_session
from app.limits import limiter, pricing_slot
from app.pricing.binomial import InvalidTreeParameters
from app.pricing.models import AmericanNotSupported, PriceOutOfBounds
from app.pricing.monte_carlo import InvalidSimulationParameters
from app.schemas.pricing import (
    CompareRequest,
    CompareResponse,
    ConvergenceReport,
    GreekProfiles,
    PriceMeta,
    PriceRequest,
    PriceResponse,
)
from app.services import market_data, pricing_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["pricing"])


@router.post("/price", summary="Цена опциона и греки")
@limiter.limit(settings.pricing_rate_limit)
async def post_price(
    request: Request,
    body: PriceRequest,
    session: AsyncSession = Depends(get_session),
) -> PriceResponse:
    try:
        # Расчёт синхронный и упирается в процессор. У Блэка-Шоулза это
        # доли миллисекунды, но дерево и Монте-Карло будут считаться
        # сотнями миллисекунд, и держать на них event loop нельзя.
        #
        # Семафор ограничивает не частоту, а одновременность: Монте-Карло
        # на пяти миллионах путей держит 200 МБ, и сорок таких расчётов,
        # которые FastAPI пустил бы в потоки разом, съели бы всю память.
        async with pricing_slot():
            response = await run_in_threadpool(pricing_service.calculate, body)
    except market_data.InstrumentNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Инструмент {body.ticker} не найден в списке доступных",
        ) from None
    except pricing_service.ModelNotImplemented as error:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Модель «{error}» ещё не реализована",
        ) from None
    except (
        AmericanNotSupported,
        InvalidTreeParameters,
        InvalidSimulationParameters,
    ) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from None
    except PriceOutOfBounds as error:
        # Не вина пользователя: никакие входные данные не могут дать цену
        # вне границ отсутствия арбитража. Значит сломана сама модель.
        logger.error("Модель вернула невозможную цену: %s", error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка расчёта: {error}",
        ) from None
    except ValueError as error:
        # Последний рубеж: проверки внутри OptionParams. Схема должна была
        # отсечь это раньше, но библиотеку можно вызвать и мимо неё.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from None

    # Недоступная база не должна отнимать у пользователя посчитанную цену:
    # результат уже есть, история — дело второе.
    try:
        response.calculation_id = await repository.save_calculation(
            session, body, response
        )
    except Exception:
        logger.exception("Не удалось сохранить расчёт в историю")

    return response


@router.post("/price/compare", summary="Цена всеми моделями сразу")
@limiter.limit(settings.pricing_rate_limit)
async def post_compare(
    request: Request,
    body: CompareRequest,
    session: AsyncSession = Depends(get_session),
) -> CompareResponse:
    try:
        async with pricing_slot():
            response = await run_in_threadpool(pricing_service.compare, body)
    except market_data.InstrumentNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Инструмент {body.ticker} не найден в списке доступных",
        ) from None

    await _save_comparison(session, body, response)
    return response


async def _save_comparison(
    session: AsyncSession, request: CompareRequest, response: CompareResponse
) -> None:
    """Каждая успешная модель попадает в историю отдельной строкой.

    Так сравнение остаётся в базе наравне с одиночными расчётами, и его
    потом можно поднять запросом. Отказавшиеся модели не сохраняем:
    сохранять нечего.
    """
    for item in response.results:
        if item.price is None:
            continue

        single = PriceRequest(**request.model_dump(), model=item.model)
        as_response = PriceResponse(
            price=item.price,
            greeks=item.greeks,
            meta=PriceMeta(
                model=item.model,
                exercise=response.exercise,
                elapsed_ms=item.elapsed_ms,
                days_to_expiry=response.days_to_expiry,
                time_to_expiry=response.time_to_expiry,
                std_error=item.std_error,
                ci95=item.ci95,
                warning=item.warning,
            ),
        )
        try:
            await repository.save_calculation(session, single, as_response)
        except Exception:
            logger.exception("Не удалось сохранить строку сравнения")


@router.post("/price/convergence", summary="Сходимость дерева и Монте-Карло")
@limiter.limit(settings.pricing_rate_limit)
async def post_convergence(
    request: Request, body: PriceRequest
) -> ConvergenceReport:
    """Как приближённые модели подходят к точному значению.

    В историю не пишется: это не расчёт цены, а замер поведения моделей —
    полсотни промежуточных значений засорили бы таблицу.
    """
    try:
        # Полсотни построений дерева и десять симуляций: секунды процессора,
        # event loop на них держать нельзя.
        async with pricing_slot():
            return await run_in_threadpool(pricing_service.convergence, body)
    except market_data.InstrumentNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Инструмент {body.ticker} не найден в списке доступных",
        ) from None
    except AmericanNotSupported:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Сходимость строится относительно Блэка-Шоулза, а он "
            "считает только европейский стиль. Переключите стиль исполнения",
        ) from None
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from None


@router.post("/price/profiles", summary="Профили греков")
@limiter.limit(settings.pricing_rate_limit)
async def post_profiles(request: Request, body: PriceRequest) -> GreekProfiles:
    """Как греки ведут себя при большом сдвиге параметров, а не малом.

    В историю не пишется: это не расчёт цены, а картина поведения модели
    вокруг заданной точки — три сотни промежуточных значений засорили бы
    таблицу.
    """
    try:
        # Сто двадцать построений для европейского стиля или шестьдесят
        # деревьев для американского: event loop на них держать нельзя.
        async with pricing_slot():
            return await run_in_threadpool(pricing_service.greek_profiles, body)
    except market_data.InstrumentNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Инструмент {body.ticker} не найден в списке доступных",
        ) from None
    except (AmericanNotSupported, InvalidTreeParameters, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from None
