"""Точка входа: собирает приложение, поднимает состояние из базы при старте и
отдаёт страницу. Запуск — uvicorn app.main:app --reload
"""

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import limits
from app.db import repository
from app.db.session import SessionFactory, engine
from app.routers import instruments, pricing
from app.security import admin_token
from app.services import market_data

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s:     %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"



@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        async with SessionFactory() as session:
            market_data.set_instruments(await repository.list_instruments(session))
            market_data.prime_cache(await repository.latest_snapshots(session))
    except Exception:
        # База или источник недоступны. Список бумаг останется пустым, на
        # странице будет предложено нажать синхронизацию вручную. Сами
        # расчёты от базы не зависят и продолжат работать.
        logger.exception("Не удалось поднять состояние из базы")

    tasks = [asyncio.create_task(_purge_calculations_forever())]

    # Наполнение пустого справочника 
    if not market_data.list_instruments():
        tasks.append(asyncio.create_task(_bootstrap_instruments()))

    yield

    for task in tasks:
        task.cancel()
    for task in tasks:
        with suppress(asyncio.CancelledError):
            await task

    await engine.dispose()


# Как часто подчищать историю расчётов
PURGE_INTERVAL_SECONDS = 3600


async def _purge_calculations_forever() -> None:
    while True:
        try:
            async with SessionFactory() as session:
                removed = await repository.purge_old_calculations(session)
            if removed:
                logger.info("Удалено устаревших расчётов: %d", removed)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Недоступная база не повод ронять задачу: расчёты от неё не
            # зависят, а на следующем круге чистка повторится.
            logger.exception("Не удалось почистить историю расчётов")

        await asyncio.sleep(PURGE_INTERVAL_SECONDS)


async def _bootstrap_instruments() -> None:
    logger.info("Справочник бумаг пуст, тянем список из скринера")

    try:
        items = await run_in_threadpool(market_data.fetch_screener_instruments)

        async with SessionFactory() as session:
            await repository.sync_instruments(session, items)
            market_data.set_instruments(await repository.list_instruments(session))

        available = len(market_data.list_instruments())
        if available:
            logger.info("Справочник наполнен: %d бумаг", available)
        else:
            logger.warning(
                "Справочник пуст: ни одна бумага не прошла порог "
                "ликвидности MIN_QUOTED_STRIKES"
            )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Не удалось наполнить справочник из скринера")


app = FastAPI(title="Option Pricing", version="0.1.0", lifespan=lifespan)
limits.install(app)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(instruments.router, prefix="/api")
app.include_router(pricing.router, prefix="/api")


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, error: RequestValidationError) -> JSONResponse:
    field_names = {
        "spot": "цена базового актива",
        "strike": "страйк",
        "expiry": "дата экспирации",
        "sigma": "волатильность",
        "rate": "ставка",
        "div_yield": "дивидендная доходность",
        "steps": "шагов дерева",
        "paths": "путей Монте-Карло",
    }

    messages = []
    for item in error.errors():
        location = [part for part in item["loc"] if part != "body"]
        name = field_names.get(str(location[-1]) if location else "")
        text = item["msg"].removeprefix("Value error, ")
        messages.append(f"{name}: {text}" if name else text)

    return JSONResponse(status_code=422, content={"detail": "; ".join(messages)})



TOKEN_PLACEHOLDER = '<meta name="admin-token" content="">'


@app.get("/", include_in_schema=False)
async def index() -> HTMLResponse:
    markup = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    filled = TOKEN_PLACEHOLDER.replace('content=""', f'content="{admin_token()}"')

    return HTMLResponse(markup.replace(TOKEN_PLACEHOLDER, filled, 1))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
