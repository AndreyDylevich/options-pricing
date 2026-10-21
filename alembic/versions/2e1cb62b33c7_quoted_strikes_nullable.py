"""quoted_strikes становится обнуляемым: NULL значит «не измерено».

Пока биржа закрыта, Yahoo отдаёт нулевые bid и ask по всем страйкам подряд.
Прежняя схема записывала это нулём, то есть «неликвид», и список бумаг
оказывался пустым каждую ночь и все выходные — проверено на чистом запуске
в 02:23 по Нью-Йорку.

Revision ID: 2e1cb62b33c7
Revises: 7befb4bac7d8
Create Date: 2026-10-21 09:24:57.990705
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "2e1cb62b33c7"
down_revision: str | Sequence[str] | None = "7befb4bac7d8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "instruments", "quoted_strikes", existing_type=sa.INTEGER(), nullable=True
    )


def downgrade() -> None:
    # Снять NULL с заполненной колонки можно только заменив сами значения:
    # иначе ALTER падает на первой же строке с NULL. Ноль здесь означает
    # «неликвид» — ровно та потеря смысла, из-за которой колонку и сделали
    # обнуляемой, но при откате схемы выразить «не измерено» больше нечем.
    op.execute("update instruments set quoted_strikes = 0 where quoted_strikes is null")
    op.alter_column(
        "instruments", "quoted_strikes", existing_type=sa.INTEGER(), nullable=False
    )
