"""Проверка административного токена на эндпоинтах, меняющих состояние.
Ничего не считает — решает, пускать запрос или отказать.
"""

import secrets

from fastapi import Header, HTTPException, status

from app.config import settings

HEADER_NAME = "X-Admin-Token"

# Случайный токен на время жизни процесса. Пересоздаётся при каждом
# запуске, поэтому в репозиторий попасть не может по построению.
_GENERATED = secrets.token_urlsafe(32)

def admin_token() -> str:
    return settings.admin_token or _GENERATED


def require_admin_token(
    x_admin_token: str | None = Header(default=None, alias=HEADER_NAME),
) -> None:
    if x_admin_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Нужен заголовок {HEADER_NAME}",
        )
    if not secrets.compare_digest(x_admin_token, admin_token()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Неверный административный токен",
        )
