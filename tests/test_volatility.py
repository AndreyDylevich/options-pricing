"""GARCH и EGARCH: проверяем не «числа не изменились», а свойства, которые у
моделей обязаны выполняться. Ряды порождаются синтетически с известными
свойствами, чтобы тест не зависел от того, что сегодня на рынке.
"""

import numpy as np
import pytest

from app.pricing import volatility

SEED = 12345


def quiet_series(n: int = 1500, sigma: float = 0.01) -> np.ndarray:
    # Ряд с постоянной волатильностью: сгустков нет.
    rng = np.random.default_rng(SEED)
    return rng.normal(0.0, sigma, n)


def clustered_series(n: int = 1500) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    omega, alpha, beta = 1e-6, 0.1, 0.85

    variance = omega / (1 - alpha - beta)
    out = np.empty(n)

    for index in range(n):
        shock = rng.normal(0.0, np.sqrt(variance))
        out[index] = shock
        variance = omega + alpha * shock**2 + beta * variance

    return out


def leveraged_series(n: int = 1500) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    omega, alpha, beta = 1e-6, 0.05, 0.85

    variance = omega / (1 - alpha - beta)
    out = np.empty(n)

    for index in range(n):
        shock = rng.normal(0.0, np.sqrt(variance))
        out[index] = shock
        # Падение добавляет к дисперсии втрое больше, чем подъём.
        push = alpha * shock**2 * (3.0 if shock < 0 else 1.0)
        variance = omega + push + beta * variance

    return out


class TestFitting:
    @pytest.mark.parametrize(
        "fit_model", [volatility.fit_garch, volatility.fit_egarch],
        ids=["GARCH", "EGARCH"],
    )
    def test_returns_sane_annual_volatility(self, fit_model) -> None:
        # Ряд с дневным σ = 1% даёт около 16% годовых: 0.01 *
        # sqrt(252).
        fit = fit_model(quiet_series(sigma=0.01), horizon_days=30)

        expected = 0.01 * np.sqrt(volatility.TRADING_DAYS)
        assert fit.term == pytest.approx(expected, rel=0.25)
        assert fit.next_day == pytest.approx(expected, rel=0.25)

    @pytest.mark.parametrize(
        "fit_model", [volatility.fit_garch, volatility.fit_egarch],
        ids=["GARCH", "EGARCH"],
    )
    def test_is_reproducible(self, fit_model) -> None:
        series = clustered_series()

        first = fit_model(series, horizon_days=30)
        second = fit_model(series, horizon_days=30)

        assert first.term == second.term
        assert first.params == second.params

    def test_garch_finds_the_parameters_it_was_given(self) -> None:
        fit = volatility.fit_garch(clustered_series(), horizon_days=30)

        persistence = fit.params["alpha[1]"] + fit.params["beta[1]"]
        assert 0.85 < persistence < 1.0

    def test_egarch_detects_asymmetry(self) -> None:
        fit = volatility.fit_egarch(leveraged_series(), horizon_days=30)

        assert fit.params["gamma[1]"] < 0

    def test_egarch_beats_garch_on_asymmetric_data(self) -> None:
        # И описывает такой ряд лучше — по AIC, который учитывает
        # лишний параметр.
        series = leveraged_series()

        garch = volatility.fit_garch(series, horizon_days=30)
        egarch = volatility.fit_egarch(series, horizon_days=30)

        assert egarch.aic < garch.aic


class TestForecastHorizon:
    def test_term_average_differs_from_one_day(self) -> None:
        series = np.concatenate([quiet_series(1400, 0.005), quiet_series(100, 0.03)])

        fit = volatility.fit_garch(series, horizon_days=60)

        assert fit.term != fit.next_day
        # Всплеск затухает, поэтому среднее по двум месяцам ниже завтрашнего.
        assert fit.term < fit.next_day

    def test_longer_horizon_moves_towards_the_long_run(self) -> None:
        # Чем дальше горизонт, тем ближе прогноз к долгосрочному
        # уровню.
        series = np.concatenate([quiet_series(1400, 0.005), quiet_series(100, 0.03)])

        short = volatility.fit_garch(series, horizon_days=5).term
        long = volatility.fit_garch(series, horizon_days=250).term

        assert long < short

    def test_horizon_is_recorded(self) -> None:
        fit = volatility.fit_garch(quiet_series(), horizon_days=17)

        assert fit.horizon_days == 17

    def test_zero_horizon_is_refused(self) -> None:
        with pytest.raises(ValueError):
            volatility.fit_garch(quiet_series(), horizon_days=0)


class TestShortSeries:
    @pytest.mark.parametrize(
        "fit_model", [volatility.fit_garch, volatility.fit_egarch],
        ids=["GARCH", "EGARCH"],
    )
    def test_refuses_too_little_history(self, fit_model) -> None:
        # Четыре параметра на сотне точек — оценки шумнее самой
        # величины.
        with pytest.raises(volatility.NotEnoughHistory):
            fit_model(quiet_series(100), horizon_days=30)

    def test_counts_only_finite_observations(self) -> None:
        # Пропуски в ряду не должны сходить за наблюдения.
        series = quiet_series(300)
        series[:100] = np.nan

        with pytest.raises(volatility.NotEnoughHistory) as error:
            volatility.fit_garch(series, horizon_days=30)

        assert "200" in str(error.value)


class TestHistorical:
    def test_matches_the_plain_formula(self) -> None:
        # Историческая считается здесь же, чтобы вся таблица опиралась
        # на один ряд доходностей.
        series = quiet_series(500, 0.02)

        value = volatility.historical(series, 252)
        expected = series[-252:].std(ddof=1) * np.sqrt(volatility.TRADING_DAYS)

        assert value == pytest.approx(expected)

    def test_returns_none_without_enough_days(self) -> None:
        assert volatility.historical(quiet_series(100), 252) is None
