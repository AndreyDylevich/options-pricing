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

# Корневой логгер по умолчанию стоит на WARNING, и всё, что приложение
# пишет через logger.info, пропадает: не видно ни чистки истории, ни
# первой синхронизации справочника. Ошибки при этом проходили и раньше —
# у них уровень выше.
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s:     %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


# Список бумаг и последние рыночные снимки живут в памяти, а базой только
# подкрепляются: так после перезапуска страница не начинает с пустого списка
# и не идёт в сеть за уже известными данными. Здесь же поднимается фоновая
# чистка истории расчётов: наружу она не отдаётся, и старые записи в таблице
# никому не нужны.
@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        async with SessionFactory() as session:
            instruments = await repository.list_instruments(session)

            if not instruments:
                instruments = await _first_sync(session)

            market_data.set_instruments(instruments)
            market_data.prime_cache(await repository.latest_snapshots(session))
    except Exception:
        # База или источник недоступны. Список бумаг останется пустым, на
        # странице будет предложено нажать синхронизацию вручную. Сами
        # расчёты от базы не зависят и продолжат работать.
        logger.exception("Не удалось поднять состояние из базы")

    cleaner = asyncio.create_task(_purge_calculations_forever())

    yield

    # Отменяем явно и дожидаемся: без этого задача умирает вместе с
    # событийным циклом посреди запроса к базе, и asyncio ругается на
    # незавершённое ожидание при каждой остановке сервера.
    cleaner.cancel()
    with suppress(asyncio.CancelledError):
        await cleaner

    await engine.dispose()


# Как часто подчищать историю расчётов. Час взят с запасом: за час
# набегают десятки строк, а не десятки тысяч, и чаще ходить в базу
# незачем. Одна чистка выполняется сразу при старте — иначе сервер,
# который поднимают и гасят каждый день, не удалил бы ничего никогда.
PURGE_INTERVAL_SECONDS = 3600


# Фоновой задачей, а не по событию записи: чистка по каждому расчёту гоняла
# бы DELETE сотни раз в день ради нескольких строк, а чистка только при
# старте ничего не дала бы серверу, поднятому на неделю.
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


# Ни одного тикера в коде не зашито, поэтому на чистой базе список пустой.
# Тянем его при первом старте, иначе пользователь увидел бы пустой
# выпадающий список и не понял, что делать.
async def _first_sync(session) -> list:
    logger.info("Справочник бумаг пуст, тянем список из скринера")

    items = await run_in_threadpool(market_data.fetch_screener_instruments)
    await repository.sync_instruments(session, items)

    return await repository.list_instruments(session)


app = FastAPI(title="Option Pricing", version="0.1.0", lifespan=lifespan)

# Общий потолок частоты запросов с одного адреса. Тяжёлые расчётные
# эндпоинты ограничены отдельно и жёстче, см. app/limits.py.
limits.install(app)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(instruments.router, prefix="/api")
app.include_router(pricing.router, prefix="/api")


# По умолчанию FastAPI отдаёт массив объектов с англоязычными msg и
# техническими loc вроде ["body", "sigma"]. Показывать такое на странице
# бессмысленно, поэтому склеиваем в текст, называя поля так же, как они
# подписаны в форме.
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


# Пустой мета-тег в index.html, в который подставляется действующий токен.
TOKEN_PLACEHOLDER = '<meta name="admin-token" content="">'


# Отдаём не файлом, а строкой: странице нужен токен для POST /sync, а
# держать его в app.js нельзя — тот раздаётся как статика и попадёт в
# репозиторий. Мета-тег же виден только тому, кто может прочитать наш ответ,
# а чужой вкладке CORS этого не позволит. Чтение файла на каждый запрос
# ничего не стоит: страница одна и весит десятки килобайт.
@app.get("/", include_in_schema=False)
async def index() -> HTMLResponse:
    markup = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    filled = TOKEN_PLACEHOLDER.replace('content=""', f'content="{admin_token()}"')

    return HTMLResponse(markup.replace(TOKEN_PLACEHOLDER, filled, 1))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
