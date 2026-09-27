"""Metrics behave as the textbook definitions say, on cases small enough to check by hand."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import (dislocation, double_lift, gamma_deviance, gini, grouped_bootstrap, ks_statistic,  # noqa: E402
                     lift_table, poisson_deviance, psi, woe_table)


def test_poisson_deviance_zero_for_perfect_fit_and_matches_hand_value():
    assert poisson_deviance([0, 1, 2], [1e-12, 1, 2]) == pytest.approx(0, abs=1e-9)
    # y=0, mu=0.5 -> 2*0.5 = 1.0 ; y=2, mu=1 -> 2*(2 ln 2 - 1)
    assert poisson_deviance([0, 2], [0.5, 1]) == pytest.approx((1.0 + 2 * (2 * np.log(2) - 1)) / 2)


def test_gamma_deviance_zero_for_perfect_fit():
    assert gamma_deviance([100, 250], [100, 250]) == pytest.approx(0)


def test_gini_perfect_random_and_reversed():
    rng = np.random.default_rng(0)
    exposure = np.ones(10_000)
    loss = rng.gamma(0.3, 10, 10_000)
    assert gini(loss, loss, exposure) > 0.6                         # ranks by the truth
    assert abs(gini(loss, rng.random(10_000), exposure)) < 0.03     # random ranking
    assert gini(loss, -loss, exposure) == pytest.approx(-gini(loss, loss, exposure), abs=1e-3)


def test_lift_table_actual_to_expected_is_one_for_calibrated_model():
    rng = np.random.default_rng(1)
    rate = rng.gamma(2, 50, 50_000)
    exposure = rng.uniform(0.1, 1, 50_000)
    loss = rng.poisson(rate * exposure / 50) * 50.0
    t = lift_table(loss, rate, exposure)
    assert len(t) == 10
    assert t.actual_to_expected.between(0.9, 1.1).all()
    assert t.predicted_pure_premium.is_monotonic_increasing


def test_double_lift_favours_the_true_model():
    rng = np.random.default_rng(2)
    true = rng.gamma(2, 50, 50_000)
    noisy = true * rng.lognormal(0, 0.5, 50_000)
    exposure = np.ones(50_000)
    loss = rng.poisson(true / 50) * 50.0
    t = double_lift(loss, noisy, true, exposure)
    # where the true model prices higher than the noisy one, actual losses follow the true model
    top = t.iloc[-1]
    assert abs(top.loss_per_exposure - top.b_per_exposure) < abs(top.loss_per_exposure - top.a_per_exposure)


def test_dislocation_ignores_overall_level():
    exposure = np.ones(4)
    assert dislocation([1, 2, 3, 4], [2, 4, 6, 8], exposure) == 0.0
    assert dislocation([1, 1, 1, 1], [1, 1, 1, 2], exposure) == 1.0   # rescaling moves everyone


def test_grouped_bootstrap_keeps_groups_together():
    groups = np.repeat(np.arange(100), 3)
    values = np.repeat(np.arange(100.0), 3)
    draws = grouped_bootstrap(lambda idx: len(idx), groups, reps=20)
    assert (draws == 300).all()                              # always 100 groups of 3 rows
    means = grouped_bootstrap(lambda idx: values[idx].mean(), groups, reps=300)
    assert 45 < means.mean() < 54


def test_ks_and_woe_on_a_separable_case():
    y = np.array([0] * 50 + [1] * 50)
    score = np.arange(100.0)
    assert ks_statistic(y, score) == pytest.approx(1.0)
    t = woe_table(np.where(score < 50, "low", "high"), y)
    assert t.loc["low", "woe"] > 0 > t.loc["high", "woe"]    # goods concentrate in "low"
    assert t.iv.sum() > 2


def test_psi_is_zero_for_same_sample_and_grows_with_shift():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, 20_000)
    assert psi(a, a) == pytest.approx(0, abs=1e-9)
    assert psi(a, rng.normal(0.5, 1, 20_000)) > 0.1
