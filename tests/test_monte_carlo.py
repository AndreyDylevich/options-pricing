"""Монте-Карло: проверяем попадание в доверительный интервал вокруг цены
Блэка-Шоулза и что оценка честно сообщает о своей погрешности.
"""

from dataclasses import replace

import pytest

from app.pricing import black_scholes as bs
from app.pricing import monte_carlo as mc
from app.pricing.models import AmericanNotSupported, ExerciseStyle, OptionParams


def test_within_three_standard_errors(atm_call: OptionParams) -> None:
    result = mc.calculate(atm_call, paths=200_000)
    exact = bs.price(atm_call)
    assert abs(result.price - exact) < 3 * result.std_error


def test_confidence_interval_contains_exact_value(atm_call: OptionParams) -> None:
    result = mc.calculate(atm_call, paths=200_000)
    low, high = result.ci95
    assert low < bs.price(atm_call) < high


def test_standard_error_falls_as_square_root(atm_call: OptionParams) -> None:
    # В сто раз больше путей — в десять раз меньше ошибка.
    small = mc.calculate(atm_call, paths=10_000).std_error
    large = mc.calculate(atm_call, paths=1_000_000).std_error
    assert large == pytest.approx(small / 10, rel=0.25)


def test_same_parameters_give_same_price(atm_call: OptionParams) -> None:
    # Seed фиксирован: цена не должна дрожать между вызовами.
    assert mc.price(atm_call, paths=50_000) == mc.price(atm_call, paths=50_000)


def test_put_call_parity_on_the_sample(
    atm_call: OptionParams, atm_put: OptionParams
) -> None:
    # C - P = exp(-rT) * (среднее S_T - K) — точно, а не приближённо.
    # Полный паритет C - P = S*exp(-qT) - K*exp(-rT) на конечной выборке
    # не выполняется: среднее разыгранных S_T отличается от форварда на
    # случайную величину. А вот тождество для той же выборки обязано
    # выполняться до последнего знака — колл и пут считаются на одном
    # массиве Z, и max(x, 0) - max(-x, 0) = x поточечно.
    import numpy as np

    tau = atm_call.time_to_expiry
    normals = mc._draw_normals(200_000)
    terminal = atm_call.spot * np.exp(
        (atm_call.rate - atm_call.div_yield - atm_call.sigma**2 / 2) * tau
        + atm_call.sigma * np.sqrt(tau) * normals
    )
    expected = float(
        np.exp(-atm_call.rate * tau) * (terminal.mean() - atm_call.strike)
    )

    difference = mc.price(atm_call, paths=200_000) - mc.price(atm_put, paths=200_000)
    assert difference == pytest.approx(expected, abs=1e-9)


def test_sample_forward_is_close_to_true_forward(atm_call: OptionParams) -> None:
    import numpy as np

    tau = atm_call.time_to_expiry
    normals = mc._draw_normals(1_000_000)
    terminal = atm_call.spot * np.exp(
        (atm_call.rate - atm_call.div_yield - atm_call.sigma**2 / 2) * tau
        + atm_call.sigma * np.sqrt(tau) * normals
    )
    forward = atm_call.spot * np.exp(
        (atm_call.rate - atm_call.div_yield) * tau
    )
    assert terminal.mean() == pytest.approx(forward, rel=1e-3)


@pytest.mark.parametrize("paths", [0, mc.MIN_PATHS - 1, mc.MAX_PATHS + 1])
def test_paths_outside_limits_are_rejected(
    atm_call: OptionParams, paths: int
) -> None:
    with pytest.raises(mc.InvalidSimulationParameters):
        mc.price(atm_call, paths=paths)


def test_american_is_rejected(atm_call: OptionParams) -> None:
    with pytest.raises(AmericanNotSupported):
        mc.price(replace(atm_call, exercise=ExerciseStyle.AMERICAN))


def test_at_expiry_skips_simulation(atm_call: OptionParams) -> None:
    # T = 0: разыгрывать нечего, цена равна внутренней стоимости.
    result = mc.calculate(replace(atm_call, time_to_expiry=0.0, spot=120.0))
    assert result.price == pytest.approx(20.0)
    assert result.std_error == 0.0
    assert result.ci95 == (20.0, 20.0)


def test_quality_warning_appears_on_extreme_volatility(
    atm_call: OptionParams,
) -> None:
    # При sigma 490% оценка разъезжается — модель обязана об этом сказать.
    result = mc.calculate(replace(atm_call, sigma=4.9), paths=100_000)
    assert result.warning is not None


def test_no_warning_on_sane_parameters(atm_call: OptionParams) -> None:
    assert mc.calculate(atm_call, paths=100_000).warning is None
