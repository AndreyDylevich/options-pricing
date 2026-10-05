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

# Общий лимит на всё. Вешается middleware, а не декоратором на каждый
# эндпоинт: декоратор slowapi требует параметр `request` в сигнатуре, и
# ради общего потолка переписывать все обработчики незачем.
# Частота и одновременность ограничены по-отдельности намеренно: они
# закрывают разные дыры, и порознь ни одна не работает.
#
# Частота. Пределы на входные параметры разумны поштучно, но ничто не
# мешало слать такие запросы подряд, а monte_carlo.calculate на пяти
# миллионах путей — это 389 мс и 200 МБ. Скрипта в десять строк хватало,
# чтобы занять машину целиком.
#
# Считаем по адресу источника. Для локального приложения это вырождается в
# один общий счётчик, и это ровно то, что нужно: защищаемся не от соседа по
# сети, а от скрипта, долбящего в одну точку.
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])

# Одновременность. Лимита частоты для неё мало: «тридцать в минуту» не
# мешает всем тридцати прийти в одну секунду, а FastAPI отдаёт под
# run_in_threadpool сорок потоков — сорок тяжёлых расчётов разом это восемь
# гигабайт пика при формально соблюдённом лимите. Семафор держит потолок:
# четыре расчёта, 800 МБ. Обратное тоже верно — одним семафором не
# обойтись: он не отказывает, а ставит в очередь, и без лимита частоты
# очередь росла бы неограниченно.
#
# Семафор свой на каждый событийный цикл. Один общий, созданный при
# импорте, не годится: asyncio привязывает семафор к первому же циклу,
# который его тронул, и в любом другом падает с «bound to a different event
# loop». Uvicorn держит один цикл на весь процесс, поэтому в бою это
# всплыло бы не сразу — зато сразу везде: каждый расчёт отвечал бы
# пятисотой.
#
# Один общий, созданный при импорте, не годится: asyncio привязывает
# семафор к первому же циклу, который его тронул, и в любом другом
# падает с «bound to a different event loop». Uvicorn держит один цикл на
# весь процесс, поэтому в бою это всплыло бы не сразу — зато всплыло бы
# сразу везде: каждый расчёт отвечал бы пятисотой.
#
# Слабые ссылки на ключи: запись уходит вместе с закрытым циклом, и
# коллекция не растёт от прогона к прогону.
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


# 429 с внятным текстом вместо стандартного английского: ошибки API в этом
# проекте на русском, и пользователь должен понять, что это защита, а не
# поломка.
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
