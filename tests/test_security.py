"""Защита эндпоинтов, меняющих состояние.

Поводом была настоящая дыра: `POST /api/instruments/sync` не требовал
ничего, а флаг `replace=true` удалял из справочника всё, чего нет в
новом ответе скринера, — вместе с рыночными снимками, которые уходили
каскадом по внешнему ключу. Сервер слушает localhost, но послать на него
POST может любая страница, открытая в браузере пользователя.

Тесты здесь сторожат оба принятых решения: удаления через HTTP больше
нет вовсе, а на то, что осталось, нужен токен.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.security import HEADER_NAME, admin_token

SYNC = "/api/instruments/sync"


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as instance:
        yield instance


@pytest.fixture
def screener(monkeypatch):
    """Подменяем скринер: тест про доступ, а не про поход в сеть.

    Считаем вызовы — по ним видно, дошёл ли запрос до работы или его
    отбили на входе.
    """
    from app.schemas.instruments import Instrument
    from app.services import market_data

    calls = {"count": 0}

    def fake(screener_name=None, count=None):
        calls["count"] += 1
        return [
            Instrument(
                ticker="KO", name="Coca-Cola", exchange="NYSE", quoted_strikes=60
            )
        ]

    monkeypatch.setattr(market_data, "fetch_screener_instruments", fake)
    return calls


class TestSyncNeedsToken:
    def test_without_header_is_401(self, client: TestClient, screener) -> None:
        response = client.post(SYNC)

        assert response.status_code == 401
        assert HEADER_NAME in response.json()["detail"]
        # Главное: до скринера дело не дошло. Иначе чужая вкладка могла бы
        # чужими руками выбрать лимиты Yahoo, даже получив отказ в ответ.
        assert screener["count"] == 0

    def test_wrong_token_is_403(self, client: TestClient, screener) -> None:
        # Латиницей: заголовки HTTP переносят только ASCII, кириллица в
        # значении роняет сам запрос, не дойдя до сервера.
        response = client.post(SYNC, headers={HEADER_NAME: "wrong-token"})

        assert response.status_code == 403
        assert screener["count"] == 0

    def test_right_token_passes(
        self, client: TestClient, screener, database
    ) -> None:
        response = client.post(SYNC, headers={HEADER_NAME: admin_token()})

        assert response.status_code == 200
        assert screener["count"] == 1

    def test_empty_token_is_refused(self, client: TestClient, screener) -> None:
        """Пустой заголовок — не то же самое, что верный токен.

        Сравнение идёт со случайной строкой, и пустая ей не равна, но
        проверить стоит: незаданный ADMIN_TOKEN не должен превращаться в
        «пускать всех с пустым заголовком».
        """
        response = client.post(SYNC, headers={HEADER_NAME: ""})

        assert response.status_code == 403
        assert screener["count"] == 0


class TestSyncCannotDelete:
    """Удаления через HTTP не осталось — это главная часть починки."""

    def test_replace_parameter_is_gone(
        self, client: TestClient, screener, database
    ) -> None:
        """Старая атака: ?count=1&replace=true оставляла одну бумагу.

        Лишний параметр запроса FastAPI молча игнорирует, поэтому важен
        не код ответа, а то, что до репозитория флаг не доедет.
        """
        from app.db import repository

        seen = {}

        async def spy(session, items, replace=False):
            seen["replace"] = replace
            return {"fetched": len(items), "added": 0, "updated": 0, "removed": 0}

        original = repository.sync_instruments
        repository.sync_instruments = spy
        try:
            response = client.post(
                f"{SYNC}?count=1&replace=true", headers={HEADER_NAME: admin_token()}
            )
        finally:
            repository.sync_instruments = original

        assert response.status_code == 200
        assert seen["replace"] is False

    def test_schema_does_not_advertise_replace(self, client: TestClient) -> None:
        """И в документации API его тоже нет — незачем подсказывать."""
        schema = client.get("/openapi.json").json()
        parameters = schema["paths"]["/api/instruments/sync"]["post"]["parameters"]

        assert "replace" not in {item["name"] for item in parameters}


class TestTokenReachesThePage:
    def test_page_carries_the_token(self, client: TestClient) -> None:
        """Страница получает токен мета-тегом, иначе кнопка не сработает."""
        page = client.get("/").text

        assert f'content="{admin_token()}"' in page

    def test_placeholder_is_not_served_empty(self, client: TestClient) -> None:
        """Пустая заготовка не должна доезжать до браузера."""
        page = client.get("/").text

        assert 'name="admin-token" content=""' not in page


def test_token_comes_from_settings_when_set(monkeypatch) -> None:
    """Заданный в .env токен перевешивает случайный."""
    from app.config import settings

    monkeypatch.setattr(settings, "admin_token", "постоянный-токен")

    assert admin_token() == "постоянный-токен"
