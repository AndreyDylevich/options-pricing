"""Асинхронный engine и фабрика сессий SQLAlchemy."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

# pool_pre_ping проверяет соединение перед выдачей из пула: PostgreSQL
# закрывает простаивающие соединения, и без проверки первый запрос после
# паузы падал бы.
engine = create_async_engine(settings.database_url, pool_pre_ping=True)

SessionFactory = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Зависимость FastAPI: сессия на время одного запроса."""
    async with SessionFactory() as session:
        yield session
