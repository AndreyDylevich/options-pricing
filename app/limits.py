"""Ограничение нагрузки: частота запросов с одного адреса и число расчётов,
идущих одновременно. Ничего не считает — отказывает или ставит в очередь.
"""

import asyncio
import weakref

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from app.config import settings

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])

_pricing_slots: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
    weakref.WeakKeyDictionary()
)


# Пропуск на тяжёлый расчёт: держать на время всей работы.
def pricing_slot() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()

    slot = _pricing_slots.get(loop)
    if slot is None:
        slot = asyncio.Semaphore(settings.max_concurrent_pricing)
        _pricing_slots[loop] = slot

    return slot

def too_many_requests(_: Request, error: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "detail": f"Слишком часто. Ограничение: {error.detail}. "
            "Подождите немного и повторите"
        },
    )


def install(app: FastAPI) -> None:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, too_many_requests)
    app.add_middleware(SlowAPIMiddleware)
