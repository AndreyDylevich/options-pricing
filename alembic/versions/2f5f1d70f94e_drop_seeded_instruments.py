"""drop seeded instruments

Revision ID: 2f5f1d70f94e
Revises: f4cf02b52ce3
Create Date: 2026-08-19 19:33:54.037723

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2f5f1d70f94e'
down_revision: Union[str, Sequence[str], None] = 'f4cf02b52ce3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Убрать бумаги, вписанные руками в первую миграцию.

    Отличить их просто: у пришедших из скринера проставлен synced_at, у
    захардкоженных он пустой. Тикеры, которые скринер успел подтвердить,
    останутся — у них отметка уже есть.
    """
    op.execute("delete from instruments where synced_at is null")


def downgrade() -> None:
    """Откатывать нечего: список наполняется синхронизацией со скринером."""
