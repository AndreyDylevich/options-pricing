"""Базовый класс для всех таблиц.

Alembic сравнивает Base.metadata с реальной схемой базы, поэтому все
модели должны быть импортированы к моменту сравнения — см. alembic/env.py.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
