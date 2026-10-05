"""Профили греков: проверяем не наличие точек, а форму кривых — у каждого
грека она известна из теории, и по ней видно, не съехала ли развёртка.
"""

from datetime import date, timedelta
from itertools import pairwise

import pytest

from app.schemas.pricing import PriceRequest
from app.services import pricing_service

SPOT = 100.0
STRIKE = 100.0


def request_for(**overrides) -> PriceRequest:
    body = {
        "model": "black_scholes",
        "ticker": "KO",
        "spot": SPOT,
        "strike": STRIKE,
        "expiry": date.today() + timedelta(days=30),
        "option_type": "call",
        "exercise": "european",
        "sigma": 0.25,
        "rate": 0.04,
        "div_yield": 0.0,
    }
    body.update(overrides)
    return PriceRequest(**body)


@pytest.fixture
def profiles():
    return pricing_service.greek_profiles(request_for())


class TestDeltaVsSpot:
    def test_runs_from_zero_to_one_for_a_call(self, profiles) -> None:
        # Дельта колла зажата между нулём и единицей и оба края
        # достигает. Если развёртка съедет по оси или перепутает
        # грека, это сразу нарушится: ни у одного другого грека таких
        # границ нет.
        values = profiles.delta_vs_spot.y

        assert values[0] == pytest.approx(0.0, abs=1e-3)
        assert values[-1] == pytest.approx(1.0, abs=1e-3)
        assert all(0.0 <= value <= 1.0 for value in values)

    def test_grows_monotonically(self, profiles) -> None:
        # Дороже актив — дороже колл, значит дельта только растёт.
        values = profiles.delta_vs_spot.y

        assert all(b >= a for a, b in pairwise(values))

    def test_put_delta_is_negative(self) -> None:
        # У пута дельта от минус единицы до нуля — знак перевёрнут.
        values = pricing_service.greek_profiles(
            request_for(option_type="put")
        ).delta_vs_spot.y

        assert values[0] == pytest.approx(-1.0, abs=1e-3)
        assert values[-1] == pytest.approx(0.0, abs=1e-3)

    def test_crosses_half_near_the_strike(self, profiles) -> None:
        # Перелом дельты приходится на страйк, а не на середину
        # графика. Проверка того, что ось построена от цены актива, а
        # не от номера точки: при съехавшей сетке половина оказалась
        # бы где угодно.
        curve = profiles.delta_vs_spot
        crossing = min(range(len(curve.y)), key=lambda i: abs(curve.y[i] - 0.5))

        assert curve.x[crossing] == pytest.approx(STRIKE, rel=0.05)


class TestGammaVsSpot:
    def test_peaks_near_the_strike(self, profiles) -> None:
        # Гамма — колокол с максимумом у страйка. Это и есть смысл
        # графика: там, где дельта меняется быстрее всего.
        curve = profiles.gamma_vs_spot
        peak = max(range(len(curve.y)), key=lambda i: curve.y[i])

        assert curve.x[peak] == pytest.approx(STRIKE, rel=0.05)

    def test_never_negative(self, profiles) -> None:
        # Гамма — вторая производная выпуклой функции, знак не меняет.
        assert all(value >= 0 for value in profiles.gamma_vs_spot.y)

    def test_falls_off_on_both_sides(self, profiles) -> None:
        # На краях гамма гаснет: далеко от страйка дельта уже не
        # меняется.
        curve = profiles.gamma_vs_spot
        peak = max(curve.y)

        assert curve.y[0] < peak / 10
        assert curve.y[-1] < peak / 10

    def test_same_for_call_and_put(self, profiles) -> None:
        # Гамма не зависит от типа: у колла и пута она совпадает.
        # Следствие паритета — их цены различаются на линейную по
        # активу величину, а вторая производная линейного равна нулю.
        put = pricing_service.greek_profiles(request_for(option_type="put"))

        for ours, theirs in zip(
            profiles.gamma_vs_spot.y, put.gamma_vs_spot.y, strict=True
        ):
            assert ours == pytest.approx(theirs, abs=1e-9)


class TestThetaVsDays:
    def test_accelerates_towards_expiry(self, profiles) -> None:
        # У опциона на деньгах распад к концу срока резко ускоряется.
        # Ради этого график и строится: за год до экспирации тета
        # почти незаметна, в последний день — на порядок больше.
        curve = profiles.theta_vs_days

        assert curve.x[0] == pytest.approx(pricing_service.PROFILE_MIN_DAYS)
        assert abs(curve.y[0]) > abs(curve.y[-1]) * 5

    def test_is_negative_for_a_long_position(self, profiles) -> None:
        # Время работает против покупателя: тета отрицательна.
        assert all(value < 0 for value in profiles.theta_vs_days.y)

    def test_axis_reaches_the_options_own_term(self) -> None:
        # Опцион дольше года не должен оказаться за краем графика.
        long_dated = pricing_service.greek_profiles(
            request_for(expiry=date.today() + timedelta(days=500))
        )

        assert long_dated.theta_vs_days.x[-1] >= 500


class TestCurrentPoint:
    def test_marks_the_given_parameters(self, profiles) -> None:
        # Точка «ты здесь» стоит там, где заданы параметры.
        assert profiles.delta_vs_spot.current_x == SPOT
        assert profiles.gamma_vs_spot.current_x == SPOT
        assert profiles.theta_vs_days.current_x == pytest.approx(30)

    def test_matches_the_plain_calculation(self, profiles) -> None:
        # И совпадает с обычным расчётом греков — иначе она врёт.
        single = pricing_service.calculate(request_for())

        assert profiles.delta_vs_spot.current_y == pytest.approx(single.greeks.delta)
        assert profiles.gamma_vs_spot.current_y == pytest.approx(single.greeks.gamma)
        assert profiles.theta_vs_days.current_y == pytest.approx(single.greeks.theta)


class TestModelChoice:
    def test_european_uses_black_scholes(self, profiles) -> None:
        # Для европейского стиля — точное значение, а не приближение.
        assert profiles.model.value == "black_scholes"

    def test_european_ignores_a_slower_choice(self) -> None:
        # Даже если пользователь выбрал дерево: кривая та же, но в сто
        # раз дороже, и приближённая вместо точной.
        picked_tree = pricing_service.greek_profiles(request_for(model="binomial"))

        assert picked_tree.model.value == "black_scholes"

    def test_american_uses_the_tree(self) -> None:
        # Американский стиль умеет только дерево.
        american = pricing_service.greek_profiles(
            request_for(model="binomial", option_type="put", exercise="american")
        )

        assert american.model.value == "binomial"
        assert len(american.delta_vs_spot.x) == pricing_service.PROFILE_POINTS_AMERICAN


class TestSpotAxis:
    def test_covers_both_spot_and_strike(self) -> None:
        # Страйк далеко от цены — в рамку должны попасть оба. Иначе
        # точка «ты здесь» оказалась бы за краем графика.
        far = pricing_service.greek_profiles(request_for(spot=100.0, strike=180.0))
        axis = far.delta_vs_spot.x

        assert axis[0] < 100.0
        assert axis[-1] > 180.0

    def test_stays_positive(self) -> None:
        # Цена актива не бывает нулевой: сетка не должна её касаться.
        assert all(value > 0 for value in pricing_service.greek_profiles(
            request_for()
        ).delta_vs_spot.x)
