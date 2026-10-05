"""Настройка Alembic: откуда брать строку подключения и метаданные схемы.
Строка берётся из .env, а не из alembic.ini — чтобы пароль не попадал в
репозиторий.
"""

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Строка подключения берётся из .env, а не из alembic.ini: пароль не
# должен лежать в файле, который попадает в репозиторий.
from app.config import settings  # noqa: E402

config.set_main_option("sqlalchemy.url", settings.database_url)

# Импорт entities обязателен: autogenerate сравнивает Base.metadata с
# реальной схемой, а таблицы попадают в metadata только при импорте
# модуля, который их объявляет.
from app.db.base import Base  # noqa: E402
from app.db import entities  # noqa: E402, F401

target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


# Run migrations in 'offline' mode. This configures the context with just a
# URL and not an Engine, though an Engine is acceptable here as well. By
# skipping the Engine creation we don't even need a DBAPI to be available.
# Calls to context.execute() here emit the given string to the script
# output.
def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


# Run migrations in 'online' mode.
def run_migrations_online() -> None:

    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
