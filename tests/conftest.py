"""Общие приспособления для тестов: подмена базы на тестовую, выключенный
лимитер, реестр бумаг и наборы параметров опциона.
"""

from dataclasses import replace

import pytest

from app.pricing.models import ExerciseStyle, OptionParams, OptionType
from app.schemas.instruments import Instrument
from app.services import market_data

# В коде не зашито ни одного тикера: обычно реестр приходит из базы, а в
# тестах его задаём здесь, чтобы они не зависели от того, что сегодня
# выдал скринер.
TEST_INSTRUMENTS = [
    Instrument(ticker="AAPL", name="Apple", exchange="NASDAQ", quoted_strikes=120),
    Instrument(ticker="KO", name="Coca-Cola", exchange="NYSE", quoted_strikes=60),
]


# Имя тестовой базы получается из боевой приписыванием суффикса:
# pricing → pricing_test. Отдельная база, а не отдельная схема, потому
# что схему пришлось бы прописывать в каждой модели, а так приложение о
# подмене вообще не знает.
TEST_DATABASE_SUFFIX = "_test"


def test_database_url() -> str:
    from sqlalchemy.engine import make_url

    from app.config import settings

    url = make_url(settings.database_url)
    # render_as_string, а не str: str() подставляет вместо пароля «***»,
    # чтобы он не утёк в логи, и получившейся строкой подключиться нельзя.
    return url.set(database=f"{url.database}{TEST_DATABASE_SUFFIX}").render_as_string(
        hide_password=False
    )


async def _prepare_test_database(url: str) -> None:
    # Создать базу и таблицы, если их ещё нет. Таблицы поднимаем через
    # metadata, а не Alembic: миграции — это история изменений схемы, а
    # тесту нужна сама схема, и прогонять ради неё десяток шагов незачем.
    # Расхождение метаданных с миграциями поймает `alembic revision
    # --autogenerate` на боевой базе.
    from sqlalchemy import text
    from sqlalchemy.engine import make_url
    from sqlalchemy.ext.asyncio import create_async_engine

    from app.db import entities  # noqa: F401  регистрирует таблицы в metadata
    from app.db.base import Base

    target = make_url(url)

    # Создавать базу можно только подключившись к другой: PostgreSQL не
    # даёт выполнить CREATE DATABASE внутри транзакции к ней самой.
    admin = create_async_engine(
        target.set(database="postgres").render_as_string(hide_password=False),
        isolation_level="AUTOCOMMIT",
    )
    try:
        async with admin.connect() as connection:
            exists = await connection.scalar(
                text("select 1 from pg_database where datname = :name"),
                {"name": target.database},
            )
            if not exists:
                await connection.execute(text(f'create database "{target.database}"'))
    finally:
        await admin.dispose()

    engine = create_async_engine(url)
    try:
        async with engine.begin() as connection:
            # Пересоздаём, а не только создаём отсутствующее: create_all не
            # меняет уже существующие таблицы, и после правки схемы тестовая
            # база молча оставалась бы прежней. Один раз так и вышло —
            # quoted_strikes стал обнуляемым в моделях, а в тестовой базе
            # сохранил NOT NULL, и тест упал на вставке NULL.
            #
            # Заодно это и есть очистка перед прогоном. Перед, а не после:
            # если тест упал, содержимое таблиц остаётся под рукой для
            # разбора до следующего запуска.
            await connection.run_sync(Base.metadata.drop_all)
            await connection.run_sync(Base.metadata.create_all)
    finally:
        await engine.dispose()


# Почему подмена автоматическая и на всю сессию, а не по требованию:
# раньше она включалась на первом тесте, который её просил, а полсотни
# тестов до него успевали записать историю расчётов в боевую базу. Роутер
# пишет туда на каждый успешный POST /price, и отключить это нельзя — в
# этом и смысл эндпоинта.
_test_database: dict[str, str] = {}


@pytest.fixture(scope="session", autouse=True)
def use_test_database():
    # Увести всю запись из боевой базы в тестовую на время прогона.
    # Подменяем фабрику сессий, а не настройку: приложение создаёт движок
    # на импорте, и правка settings на него бы не подействовала. Если
    # PostgreSQL не поднять, ничего не подменяем и запоминаем причину:
    # тестам, которым база нужна, её сообщит фикстура `database`, а
    # остальные шестьсот сорок два пройдут как шли. Математика базы не
    # знает, сеть подменяется, а роутер расчёта глушит ошибки записи —
    # чтобы недоступная база не отнимала у пользователя посчитанную цену.
    import asyncio

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from app.db import session as session_module

    url = test_database_url()

    try:
        asyncio.run(_prepare_test_database(url))
    except Exception as error:
        _test_database["problem"] = f"{type(error).__name__}: {error}"
        yield
        return

    # NullPool обязателен: TestClient заводит свой событийный цикл на
    # каждый экземпляр, а пул привязывает соединения к тому циклу, в
    # котором их открыл. С обычным пулом второй клиент получал бы чужое
    # соединение и падал на «attached to a different loop». Цена нового
    # соединения на запрос в тестах несущественна.
    engine = create_async_engine(url, poolclass=NullPool)
    original_engine = session_module.engine
    original_factory = session_module.SessionFactory

    session_module.engine = engine
    session_module.SessionFactory = async_sessionmaker(engine, expire_on_commit=False)
    _test_database["url"] = url

    yield

    session_module.engine = original_engine
    session_module.SessionFactory = original_factory
    asyncio.run(engine.dispose())


@pytest.fixture
def database(use_test_database) -> str:
    # Адрес тестовой базы; пропуск теста, если её не подняли. Тот, кто
    # впервые склонировал репозиторий, запускает `pytest` раньше, чем
    # читает README, и должен увидеть зелёный прогон, а не красные строки
    # про отказ соединения.
    url = _test_database.get("url")
    if url is None:
        pytest.skip(
            "нужен PostgreSQL: поднимите его и пропишите DATABASE_URL в .env "
            f"(см. README). Причина: {_test_database.get('problem')}"
        )
    return url


@pytest.fixture(autouse=True)
def rate_limits_off():
    # Выключить ограничение частоты на время теста. Иначе набор тестов
    # упирался бы в собственную защиту: расчётных запросов за прогон около
    # тридцати, ровно на границе лимита. Держался он лишь потому, что
    # проверки на кривой ввод получают 422 раньше лимитера и в счёт не
    # идут, — то есть от каждого нового теста набор мог покраснеть без
    # всякой причины. Сам лимитер проверяется в test_limits.py, там он
    # включается обратно.
    from app import limits

    was = limits.limiter.enabled
    limits.limiter.enabled = False
    limits.limiter.reset()
    yield
    limits.limiter.enabled = was


@pytest.fixture(autouse=True)
def instrument_registry():
    market_data.set_instruments(TEST_INSTRUMENTS)
    yield
    market_data.set_instruments([])


@pytest.fixture
def atm_call() -> OptionParams:
    # Опцион на деньгах: год до экспирации, ставка 5%, дивиденды 2%.
    return OptionParams(
        spot=100.0,
        strike=100.0,
        time_to_expiry=1.0,
        rate=0.05,
        div_yield=0.02,
        sigma=0.20,
        option_type=OptionType.CALL,
    )


@pytest.fixture
def atm_put(atm_call: OptionParams) -> OptionParams:
    return replace(atm_call, option_type=OptionType.PUT)


# Набор для проверок, которые должны выполняться при любых разумных
# входных данных: деньги, срок, волатильность, ставка, стиль.
PARAMETER_GRID = [
    OptionParams(
        spot=spot,
        strike=100.0,
        time_to_expiry=tau,
        rate=rate,
        div_yield=0.02,
        sigma=sigma,
        option_type=option_type,
        exercise=exercise,
    )
    for spot in (60.0, 100.0, 160.0)
    for tau in (1 / 365, 0.25, 2.0)
    for sigma in (0.05, 0.30, 1.20)
    for rate in (-0.01, 0.05)
    for option_type in OptionType
    for exercise in ExerciseStyle
]
