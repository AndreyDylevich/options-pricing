"""Ограничение нагрузки: частота запросов и одновременность расчётов.

Оба параметра ограничены по-отдельности намеренно — они закрывают разные
дыры, и порознь ни один не работает.

**Частота.** Пределы на входные параметры разумны поштучно: пять тысяч
шагов дерева, пять миллионов путей Монте-Карло. Но ничто не мешало слать
такие запросы подряд, а `monte_carlo.calculate` на пяти миллионах путей —
это 389 мс и 200 МБ: симуляция запускается двенадцать раз, одна на цену и
одиннадцать на численные греки. Скрипта в десять строк хватало, чтобы
занять машину целиком.

**Одновременность.** Лимита частоты для этого мало: «тридцать в минуту»
не мешает всем тридцати прийти в одну секунду. FastAPI отдаёт под
`run_in_threadpool` сорок потоков, сорок тяжёлых расчётов разом — это
восемь гигабайт пика, то есть смерть по памяти при формально соблюдённом
лимите. Семафор держит потолок: четыре расчёта, 800 МБ.

Обратное тоже верно — одним семафором не обойтись. Он не отказывает, а
ставит в очередь, и без ограничения частоты очередь росла бы
неограниченно.

Считаем по адресу источника. Для локального приложения это вырождается в
один общий счётчик, и это ровно то, что нужно: защищаемся не от соседа по
сети, а от скрипта, долбящего в одну точку.
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
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])

# Отдельный потолок памяти — по семафору на событийный цикл.
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


def pricing_slot() -> asyncio.Semaphore:
    """Пропуск на тяжёлый расчёт. Держать на время всей работы."""
    loop = asyncio.get_running_loop()

    slot = _pricing_slots.get(loop)
    if slot is None:
        slot = asyncio.Semaphore(settings.max_concurrent_pricing)
        _pricing_slots[loop] = slot

    return slot


def too_many_requests(_: Request, error: RateLimitExceeded) -> JSONResponse:
    """429 с внятным текстом вместо стандартного английского.

    Ошибки API в этом проекте на русском, и отказ по частоте не исключение
    — пользователь должен понять, что это не поломка, а защита.
    """
    return JSONResponse(
        status_code=429,
        content={
            "detail": f"Слишком часто. Ограничение: {error.detail}. "
            "Подождите немного и повторите"
        },
    )


def install(app: FastAPI) -> None:
    """Подключить ограничение частоты к приложению."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, too_many_requests)
    app.add_middleware(SlowAPIMiddleware)
