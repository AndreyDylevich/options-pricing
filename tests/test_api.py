"""Проверки на границе HTTP: что пользователь не может ввести чепуху."""

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Один клиент на весь модуль.

    TestClient поднимает собственный event loop, а пул соединений
    SQLAlchemy привязан к тому, в котором был создан. С клиентом на каждый
    тест второй запрос к базе падал бы с «Event loop is closed», причём
    незаметно: роутер глушит ошибки записи, чтобы недоступная база не
    отнимала посчитанную цену.
    """
    with TestClient(app) as instance:
        yield instance


def request_body(**overrides) -> dict:
    body = {
        "model": "black_scholes",
        "ticker": "KO",
        "spot": 87.42,
        "strike": 87.5,
        "expiry": str(date.today() + timedelta(days=35)),
        "option_type": "call",
        "exercise": "european",
        "sigma": 0.2661,
        "rate": 0.037,
        "div_yield": 0.0238,
    }
    body.update(overrides)
    return body


@pytest.mark.parametrize(
    "overrides,fragment",
    [
        ({"spot": 0}, "greater than 0"),
        ({"spot": -1}, "greater than 0"),
        ({"strike": 0}, "greater than 0"),
        ({"sigma": 0}, "greater than 0"),
        ({"sigma": 50}, "less than or equal"),
        ({"rate": 5}, "less than or equal"),
        ({"rate": -2}, "greater than or equal"),
        ({"div_yield": -0.1}, "greater than or equal"),
        # Потолок q высокий намеренно: на коротком сроке дискретная выплата
        # даёт большую годовую ставку, см. MAX_DIV_YIELD.
        ({"div_yield": 6}, "less than or equal"),
        ({"steps": 0}, "greater than or equal"),
        ({"steps": 99999}, "less than or equal"),
        ({"paths": 10}, "greater than or equal"),
        ({"paths": 10**9}, "less than or equal"),
        ({"expiry": str(date.today())}, "должна быть в будущем"),
        ({"expiry": str(date.today() - timedelta(days=1))}, "должна быть в будущем"),
        ({"strike": 100000}, "более чем в сто раз"),
        ({"strike": 0.01}, "более чем в сто раз"),
        ({"exercise": "american"}, "только биномиальное дерево"),
        ({"option_type": "лонг"}, "Input should be"),
        ({"model": "garch"}, "Input should be"),
    ],
)
def test_invalid_input_is_rejected(
    client: TestClient, overrides: dict, fragment: str
) -> None:
    response = client.post("/api/price", json=request_body(**overrides))
    assert response.status_code == 422
    assert fragment in response.json()["detail"]


def test_unknown_ticker_is_404(client: TestClient) -> None:
    response = client.post("/api/price", json=request_body(ticker="ZZZZ"))
    assert response.status_code == 404


@pytest.mark.parametrize("model", ["black_scholes", "binomial", "monte_carlo"])
def test_all_models_price_the_same_option(client: TestClient, model: str) -> None:
    response = client.post("/api/price", json=request_body(model=model))
    assert response.status_code == 200

    body = response.json()
    assert body["price"] == pytest.approx(2.88, abs=0.05)
    assert set(body["greeks"]) == {"delta", "gamma", "vega", "theta", "rho"}
    assert body["meta"]["model"] == model


def test_american_works_with_binomial(client: TestClient) -> None:
    response = client.post(
        "/api/price", json=request_body(model="binomial", exercise="american")
    )
    assert response.status_code == 200


def test_monte_carlo_returns_interval(client: TestClient) -> None:
    meta = client.post("/api/price", json=request_body(model="monte_carlo")).json()["meta"]
    assert meta["std_error"] > 0
    assert meta["ci95"][0] < meta["ci95"][1]


def test_analytic_models_have_no_interval(client: TestClient) -> None:
    meta = client.post("/api/price", json=request_body(model="binomial")).json()["meta"]
    assert meta["std_error"] is None
    assert meta["ci95"] is None
    assert meta["warning"] is None


# --- сравнение всех моделей ---


def test_compare_returns_all_three(client: TestClient) -> None:
    body = request_body()
    body.pop("model")
    response = client.post("/api/price/compare", json=body)
    assert response.status_code == 200

    results = response.json()["results"]
    assert [item["model"] for item in results] == [
        "black_scholes",
        "binomial",
        "monte_carlo",
    ]
    assert all(item["price"] == pytest.approx(2.88, abs=0.05) for item in results)


def test_compare_uses_identical_parameters(client: TestClient) -> None:
    """Все модели должны сойтись между собой: параметры-то одни."""
    body = request_body()
    body.pop("model")
    results = client.post("/api/price/compare", json=body).json()["results"]

    prices = [item["price"] for item in results]
    assert max(prices) - min(prices) < 0.05


def test_compare_reports_refusals_instead_of_failing(client: TestClient) -> None:
    """Американский стиль: дерево считает, две другие объясняют отказ."""
    body = request_body(exercise="american")
    body.pop("model")
    response = client.post("/api/price/compare", json=body)
    assert response.status_code == 200

    by_model = {item["model"]: item for item in response.json()["results"]}
    assert by_model["binomial"]["price"] is not None
    assert by_model["black_scholes"]["price"] is None
    assert "европейск" in by_model["black_scholes"]["error"]
    assert by_model["monte_carlo"]["price"] is None


def test_compare_rejects_bad_input(client: TestClient) -> None:
    body = request_body(sigma=50)
    body.pop("model")
    assert client.post("/api/price/compare", json=body).status_code == 422


def test_compare_unknown_ticker_is_404(client: TestClient) -> None:
    body = request_body(ticker="ZZZZ")
    body.pop("model")
    assert client.post("/api/price/compare", json=body).status_code == 404


def in_own_loop(work, url):
    """Выполнить запрос к базе на одноразовом движке в своём цикле.

    Общий SessionFactory сюда не годится: его пул привязан к циклу
    TestClient, а тот между запросами закрыт. Чужой цикл роняет asyncpg
    на «attached to a different loop», поэтому движок создаётся и
    закрывается внутри той же asyncio.run.

    Адрес приходит из фикстуры `database` — это тестовая база, а не
    боевая. Брать его из настроек нельзя: там адрес приложения, и
    проверка смотрела бы не в ту таблицу, куда писала.
    """
    import asyncio

    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    async def run():
        engine = create_async_engine(url)
        try:
            # expire_on_commit=False как в SessionFactory: иначе после
            # коммита чтение row.id полезло бы в базу за обновлением и
            # упало на MissingGreenlet.
            async with AsyncSession(engine, expire_on_commit=False) as session:
                return await work(session)
        finally:
            await engine.dispose()

    return asyncio.run(run())


def test_compare_saves_every_priced_model(client: TestClient, database) -> None:
    """Сравнение кладёт в историю по строке на каждую посчитавшую модель.

    Смотрим прямо в базу: наружу история не отдаётся, эндпоинта для неё
    больше нет. Считаем по идентификаторам, а не по числу строк, —
    таблица растёт от прогона к прогону.
    """
    from sqlalchemy import select

    from app.db.entities import Calculation

    async def newest_id(session):
        query = select(Calculation.id).order_by(Calculation.id.desc()).limit(1)
        return await session.scalar(query) or 0

    last_id = in_own_loop(newest_id, database)

    body = request_body()
    body.pop("model")
    client.post("/api/price/compare", json=body)

    async def models_after(session):
        rows = await session.scalars(
            select(Calculation.model).where(Calculation.id > last_id)
        )
        return set(rows.all())

    assert in_own_loop(models_after, database) == {
        "black_scholes",
        "binomial",
        "monte_carlo",
    }


# --- цепочка опционов ---


def test_chain_requires_known_expiry(client: TestClient, monkeypatch) -> None:
    from app.services import pricing_service

    def boom(ticker, expiry, window):
        raise pricing_service.UnknownExpiry(expiry)

    monkeypatch.setattr(pricing_service, "option_chain", boom)
    response = client.get("/api/instruments/KO/chain?expiry=2030-01-18")
    assert response.status_code == 404
    assert "2030-01-18" in response.json()["detail"]


def test_chain_unknown_ticker_is_404(client: TestClient) -> None:
    response = client.get("/api/instruments/ZZZZ/chain?expiry=2026-09-18")
    assert response.status_code == 404


def test_chain_returns_solved_volatility(client: TestClient, monkeypatch) -> None:
    """Цепочка отдаётся с посчитанной волатильностью по каждому страйку."""
    from datetime import date, datetime, timedelta, timezone

    from app.schemas.instruments import MarketSnapshot
    from app.services import market_data

    expiry = str(date.today() + timedelta(days=30))
    now = datetime.now(timezone.utc)

    snapshot = MarketSnapshot(
        ticker="KO", name="Coca-Cola", currency="USD", spot=100.0,
        sigma_30d=0.25, sigma_90d=0.25, sigma_252d=0.25,
        div_yield=0.0, risk_free_rate=0.05,
        expirations=[expiry], as_of=now, fetched_at=now,
    )
    monkeypatch.setattr(
        market_data, "get_market_snapshot", lambda ticker, refresh=False: snapshot
    )

    # Цены посчитаны Блэком-Шоулзом при sigma = 30%: решатель обязан
    # вернуть ровно её.
    from dataclasses import replace

    from app.pricing import black_scholes
    from app.pricing.models import OptionParams, OptionType

    def fake_chain(ticker, expiry_date):
        rows = []
        for strike in (90.0, 100.0, 110.0):
            for side, kind in (("call", OptionType.CALL), ("put", OptionType.PUT)):
                params = OptionParams(
                    spot=100.0, strike=strike, time_to_expiry=30 / 365,
                    rate=0.05, div_yield=0.0, sigma=0.30, option_type=kind,
                )
                price = black_scholes.price(params)
                rows.append((side, {
                    "strike": strike, "bid": price - 0.001, "ask": price + 0.001,
                    "volume": 10, "openInterest": 100, "impliedVolatility": 0.0,
                }))
        return rows, len(rows) + 4

    monkeypatch.setattr(market_data, "get_option_chain", fake_chain)

    body = client.get(f"/api/instruments/KO/chain?expiry={expiry}").json()

    assert body["quoted"] == 6
    assert body["skipped"] == 4
    assert body["solved"] == 6
    for quote in body["quotes"]:
        assert quote["implied_vol"] == pytest.approx(0.30, abs=1e-4)
        assert quote["iv_note"] is None


def _parity_fixture(
    monkeypatch,
    market_div_yield: float,
    our_div_yield: float,
    half_spread: float = 0.01,
):
    """Цепочка, построенная при одном q, и снимок, уверенный в другом.

    Ровно та ситуация, ради которой сверка и написана: формулы считают
    верно, а подставленная в них дивидендная доходность неверна.
    """
    from datetime import date, datetime, timedelta, timezone

    from app.pricing import black_scholes
    from app.pricing.models import OptionParams, OptionType
    from app.schemas.instruments import MarketSnapshot
    from app.services import market_data

    expiry = str(date.today() + timedelta(days=30))
    now = datetime.now(timezone.utc)
    tau = 30 / 365

    snapshot = MarketSnapshot(
        ticker="KO", name="Coca-Cola", currency="USD", spot=100.0,
        sigma_30d=0.25, sigma_90d=0.25, sigma_252d=0.25,
        div_yield=our_div_yield, risk_free_rate=0.05,
        div_yield_by_expiry={expiry: our_div_yield},
        expirations=[expiry], as_of=now, fetched_at=now,
    )
    monkeypatch.setattr(
        market_data, "get_market_snapshot", lambda ticker, refresh=False: snapshot
    )

    def fake_chain(ticker, expiry_date):
        rows = []
        for strike in (90.0, 95.0, 100.0, 105.0, 110.0):
            for side, kind in (("call", OptionType.CALL), ("put", OptionType.PUT)):
                params = OptionParams(
                    spot=100.0, strike=strike, time_to_expiry=tau,
                    rate=0.05, div_yield=market_div_yield, sigma=0.30,
                    option_type=kind,
                )
                price = black_scholes.price(params)
                rows.append((side, {
                    "strike": strike,
                    "bid": price - half_spread,
                    "ask": price + half_spread,
                }))
        return rows, len(rows)

    monkeypatch.setattr(market_data, "get_option_chain", fake_chain)
    return expiry


def test_parity_confirms_correct_inputs(client: TestClient, monkeypatch) -> None:
    """Наше q совпадает с рыночным — сверка обязана это подтвердить."""
    expiry = _parity_fixture(monkeypatch, market_div_yield=0.03, our_div_yield=0.03)

    body = client.get(f"/api/instruments/KO/parity?expiry={expiry}").json()

    assert body["pairs"] == 5
    assert body["agreed"] == 5
    assert body["arbitrage"] == 0
    assert body["div_yield_market"] == pytest.approx(0.03, abs=1e-6)
    assert "согласованы" in body["verdict"]


def test_parity_catches_wrong_div_yield(client: TestClient, monkeypatch) -> None:
    """Рынок живёт при 3%, мы считаем при 8% — сверка обязана это поймать.

    Главный тест всего модуля: ни одна внутренняя проверка такую ошибку
    не видит, потому что формулы при неверном q считают так же безупречно.
    """
    expiry = _parity_fixture(monkeypatch, market_div_yield=0.03, our_div_yield=0.08)

    body = client.get(f"/api/instruments/KO/parity?expiry={expiry}").json()

    assert body["agreed"] == 0
    # Рыночное q достаётся из цен и на нашей ошибке не должно сказаться.
    assert body["div_yield_market"] == pytest.approx(0.03, abs=1e-6)
    assert body["div_yield_ours"] == pytest.approx(0.08)
    assert "разобраться" in body["verdict"]


def test_parity_does_not_cry_wolf_on_short_terms(
    client: TestClient, monkeypatch
) -> None:
    """Целый процентный пункт q на месячном сроке — это ещё не расхождение.

    q — ставка годовая, и на коротком сроке она пересчитывается из
    копеечного сдвига форварда: процентный пункт здесь даёт восемь центов
    на стодолларовой бумаге, вчетверо меньше спреда. Вердикт поэтому
    выносится по форварду, а не по q, — иначе он спорил бы с собственной
    построчной таблицей, где все страйки сошлись.
    """
    expiry = _parity_fixture(
        monkeypatch, market_div_yield=0.03, our_div_yield=0.04, half_spread=0.15
    )

    body = client.get(f"/api/instruments/KO/parity?expiry={expiry}").json()

    assert body["agreed"] == body["pairs"]
    assert "согласованы" in body["verdict"]


def test_parity_rejects_unknown_expiry(client: TestClient, monkeypatch) -> None:
    _parity_fixture(monkeypatch, market_div_yield=0.03, our_div_yield=0.03)

    response = client.get("/api/instruments/KO/parity?expiry=2030-01-18")

    assert response.status_code == 404


def test_parity_rejects_unknown_ticker(client: TestClient) -> None:
    response = client.get("/api/instruments/NOSUCH/parity?expiry=2030-01-18")

    assert response.status_code == 404


# --- очистка истории расчётов ---


def test_purge_removes_only_stale_rows(client: TestClient, database) -> None:
    """Старое удаляется, свежее остаётся.

    История наружу не отдаётся, поэтому и проверяется прямо в базе.
    Кладём две записи по разные стороны срока хранения и смотрим, какая
    переживёт чистку.
    """
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import select

    from app.config import settings
    from app.db import repository
    from app.db.entities import Calculation

    retention = settings.calculation_retention_days
    now = datetime.now(timezone.utc)

    def sample(created_at: datetime) -> Calculation:
        return Calculation(
            created_at=created_at, ticker="KO", model="black_scholes",
            option_type="call", exercise="european",
            spot=100.0, strike=100.0, expiry=date.today() + timedelta(days=30),
            days_to_expiry=30, time_to_expiry=30 / 365,
            sigma=0.2, rate=0.05, div_yield=0.0,
            price=5.0, delta=0.5, gamma=0.02, vega=0.1, theta=-0.01, rho=0.04,
            elapsed_ms=1.0, model_params={},
        )

    async def insert_pair(session):
        stale = sample(now - timedelta(days=retention, hours=1))
        fresh = sample(now - timedelta(hours=1))
        session.add_all([stale, fresh])
        await session.commit()
        return stale.id, fresh.id

    stale_id, fresh_id = in_own_loop(insert_pair, database)

    async def purge(session):
        return await repository.purge_old_calculations(session)

    assert in_own_loop(purge, database) >= 1

    async def survivors(session):
        rows = await session.scalars(
            select(Calculation.id).where(Calculation.id.in_([stale_id, fresh_id]))
        )
        return set(rows.all())

    assert in_own_loop(survivors, database) == {fresh_id}


def test_purge_honours_an_explicit_retention(
    client: TestClient, database
) -> None:
    """Срок хранения можно задать явно — на нём и держатся тесты."""
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import select

    from app.db import repository
    from app.db.entities import Calculation

    now = datetime.now(timezone.utc)

    async def insert_one(session):
        row = Calculation(
            created_at=now - timedelta(days=400), ticker="KO",
            model="black_scholes", option_type="call", exercise="european",
            spot=100.0, strike=100.0, expiry=date.today() + timedelta(days=30),
            days_to_expiry=30, time_to_expiry=30 / 365,
            sigma=0.2, rate=0.05, div_yield=0.0,
            price=5.0, delta=0.5, gamma=0.02, vega=0.1, theta=-0.01, rho=0.04,
            elapsed_ms=1.0, model_params={},
        )
        session.add(row)
        await session.commit()
        return row.id

    row_id = in_own_loop(insert_one, database)

    # Записи 400 дней. Со сроком в 1000 дней она пережидает чистку, со
    # сроком в 300 — удаляется. Нулевой срок сюда ставить нельзя: тесты
    # ходят в ту же базу, что и приложение, и «удалить всё старше сего
    # момента» снесло бы настоящую историю расчётов.
    async def keep(session):
        return await repository.purge_old_calculations(session, retention_days=1000)

    in_own_loop(keep, database)

    async def exists(session):
        return await session.scalar(
            select(Calculation.id).where(Calculation.id == row_id)
        )

    assert in_own_loop(exists, database) == row_id

    async def drop(session):
        return await repository.purge_old_calculations(session, retention_days=300)

    in_own_loop(drop, database)
    assert in_own_loop(exists, database) is None
