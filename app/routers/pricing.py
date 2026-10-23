"""HTTP-слой расчёта: принимает параметры опциона, зовёт pricing_service и
отдаёт цену с греками. Сам ничего не считает. Каждый расчёт попадает в
таблицу `calculations`, но наружу история не отдаётся.
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
        # Расчёт синхронный и упирается в процессор: дерево и Монте-Карло —
        # сотни миллисекунд, держать на них event loop нельзя.
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
        # Не вина пользователя: цена вне границ арбитража значит, что сломана модель.
        logger.error("Модель вернула невозможную цену: %s", error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка расчёта: {error}",
        ) from None
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from None

    # Недоступная база не должна отнимать у пользователя посчитанную цену.
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


@router.post(
    "/price/convergence",
    summary="Сходимость дерева и Монте-Карло",
    description="Как приближённые модели подходят к точному значению. "
    "В историю не пишется: это замер поведения моделей, а не расчёт цены.",
)
@limiter.limit(settings.pricing_rate_limit)
async def post_convergence(
    request: Request, body: PriceRequest
) -> ConvergenceReport:
    try:
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


@router.post(
    "/price/profiles",
    summary="Профили греков",
    description="Как греки ведут себя при большом сдвиге параметров, а не "
    "малом: дельта и гамма по цене актива, тета по сроку. В историю не "
    "пишется — это картина поведения модели, а не расчёт цены.",
)
@limiter.limit(settings.pricing_rate_limit)
async def post_profiles(request: Request, body: PriceRequest) -> GreekProfiles:
    try:
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
