"""График дивидендных отсечек в снимке рыночных данных.

Без него после перезапуска сервера кэш поднимался из базы с пустым
графиком, и расчёт q по датам отсечек молча откатывался к трейлинговой
доходности за 12 месяцев — то есть вся дивидендная работа переставала
действовать до первого ручного обновления.

Revision ID: 7befb4bac7d8
Revises: ba84f1e7aa7e
Create Date: 2026-09-14 19:55:19.349350
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "7befb4bac7d8"
down_revision: Union[str, Sequence[str], None] = "ba84f1e7aa7e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Колонка NOT NULL на уже заполненной таблице проходит только со
    # значением по умолчанию: старым снимкам его взять неоткуда. Сразу
    # после заполнения умолчание снимаем — новые записи обязаны приносить
    # график явно, молчаливый пустой список замаскировал бы ошибку.
    op.add_column(
        "market_snapshots",
        sa.Column(
            "div_schedule",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.alter_column("market_snapshots", "div_schedule", server_default=None)


def downgrade() -> None:
    op.drop_column("market_snapshots", "div_schedule")
