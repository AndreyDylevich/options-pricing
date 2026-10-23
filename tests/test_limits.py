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
    codes = [client.get("/api/health").status_code for _ in range(130)]

    assert codes.count(200) == 120
    assert codes.count(429) == 10


def test_refusal_is_explained_in_russian(client, rate_limits_on) -> None:
    for _ in range(121):
        response = client.get("/api/health")

    assert response.status_code == 429
    detail = response.json()["detail"]
    assert "Слишком часто" in detail
    assert "120 per 1 minute" in detail


def test_pricing_has_its_own_stricter_limit(client, rate_limits_on) -> None:
    from app.config import settings

    assert settings.pricing_rate_limit != settings.rate_limit

    schema = client.get("/openapi.json").json()
    assert "/api/price" in schema["paths"]


class TestPricingSlot:

    def test_allows_only_the_configured_number_at_once(self) -> None:
        from app.config import settings

        peak = 0
        running = 0

        async def heavy():
            nonlocal peak, running
            async with limits.pricing_slot():
                running += 1
                peak = max(peak, running)
                await asyncio.sleep(0.01)
                running -= 1

        async def storm():
            await asyncio.gather(*(heavy() for _ in range(40)))

        asyncio.run(storm())

        assert peak == settings.max_concurrent_pricing

    def test_all_requests_still_complete(self) -> None:
        done = 0

        async def work():
            nonlocal done
            async with limits.pricing_slot():
                await asyncio.sleep(0)
                done += 1

        async def scenario():
            await asyncio.gather(*(work() for _ in range(20)))

        asyncio.run(scenario())

        assert done == 20

    def test_slot_is_released_after_a_failure(self) -> None:
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
