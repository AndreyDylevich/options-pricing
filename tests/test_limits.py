"""Ограничение нагрузки: проверяем, что лимит частоты отказывает на нужном
запросе, а семафор не пускает больше настроенного числа расчётов разом.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from app import limits
from app.main import app


@pytest.fixture
def rate_limits_on(rate_limits_off):
    # Включить лимитер обратно: по умолчанию в тестах он выключен.
    # Зависимость от `rate_limits_off` не декоративная — она задаёт
    # порядок. Без неё автоматическая фикстура могла бы выключить лимитер
    # после того, как эта его включила.
    limits.limiter.enabled = True
    limits.limiter.reset()
    yield
    limits.limiter.enabled = False
    limits.limiter.reset()


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as instance:
        yield instance


def test_health_is_capped_by_the_shared_limit(client, rate_limits_on) -> None:
    # Общий потолок действует на всём, даже на самом дешёвом эндпоинте.
    # Считаем по здоровью: он ничего не считает и не ходит в базу, поэтому
    # отказ здесь означает именно работу лимитера, а не что-то ещё.
    codes = [client.get("/api/health").status_code for _ in range(130)]

    assert codes.count(200) == 120
    assert codes.count(429) == 10


def test_refusal_is_explained_in_russian(client, rate_limits_on) -> None:
    # Отказ должен читаться как защита, а не как поломка.
    for _ in range(121):
        response = client.get("/api/health")

    assert response.status_code == 429
    detail = response.json()["detail"]
    assert "Слишком часто" in detail
    assert "120 per 1 minute" in detail


def test_pricing_has_its_own_stricter_limit(client, rate_limits_on) -> None:
    # Расчёты ограничены жёстче общего потока — они дороже всех прочих.
    from app.config import settings

    assert settings.pricing_rate_limit != settings.rate_limit

    schema = client.get("/openapi.json").json()
    assert "/api/price" in schema["paths"]


class TestPricingSlot:
    # Семафор: то, что удерживает память, когда лимит частоты соблюдён.

    def test_allows_only_the_configured_number_at_once(self) -> None:
        # Тридцать запросов в минуту могут прийти в одну секунду.
        # Лимитер их пропустит — он считает частоту, а не
        # одновременность. Потолок памяти держит только семафор,
        # поэтому проверяем, что дальше настроенного числа
        # одновременных расчётов не пускают.
        from app.config import settings

        peak = 0
        running = 0

        async def heavy():
            nonlocal peak, running
            async with limits.pricing_slot():
                running += 1
                peak = max(peak, running)
                # Отдаём управление, чтобы остальные успели дойти до
                # семафора: без этого задачи выполнялись бы по очереди и
                # одновременности не возникло бы даже без ограничения.
                await asyncio.sleep(0.01)
                running -= 1

        async def storm():
            await asyncio.gather(*(heavy() for _ in range(40)))

        asyncio.run(storm())

        assert peak == settings.max_concurrent_pricing

    def test_all_requests_still_complete(self) -> None:
        # Семафор ставит в очередь, а не отказывает: расчёт не
        # теряется.
        done = 0

        async def work():
            nonlocal done
            async with limits.pricing_slot():
                await asyncio.sleep(0)
                done += 1

        # gather нельзя собирать до запуска цикла — он сразу заводит
        # задачи, а цикла ещё нет.
        async def scenario():
            await asyncio.gather(*(work() for _ in range(20)))

        asyncio.run(scenario())

        assert done == 20

    def test_slot_is_released_after_a_failure(self) -> None:
        # Упавший расчёт обязан вернуть пропуск. Иначе несколько
        # ошибок подряд выбрали бы все места, и сервис встал бы
        # намертво при полностью исправном коде расчёта.
        from app.config import settings

        async def failing():
            with pytest.raises(ValueError):
                async with limits.pricing_slot():
                    raise ValueError("расчёт не удался")

        async def scenario():
            for _ in range(settings.max_concurrent_pricing * 3):
                await failing()
            # Места на месте: семафор пропускает столько же, сколько и до.
            assert not limits.pricing_slot().locked()

        asyncio.run(scenario())
