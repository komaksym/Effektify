"""Toy recovery tests: univariate ``fit_curve`` and joint ``fit_joint``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.bootstrap import bootstrap_joint
from src.curve_fit import fit_curve, fit_joint


def test_univariate_recovery() -> None:
    """α=10000, β=500 + 5% noise → fit_curve recovers params with high R²."""

    rng = np.random.default_rng(0)
    true_alpha, true_beta = 10_000.0, 500.0

    spend = rng.uniform(0.0, 5_000.0, size=200)
    clean = true_alpha * np.log1p(spend / true_beta)
    noise = rng.normal(0.0, float(clean.std()) * 0.05, size=spend.size)
    revenue = clean + noise

    alpha, beta, r2 = fit_curve(spend, revenue)

    assert r2 > 0.95
    assert abs(alpha - true_alpha) / true_alpha < 0.10
    assert abs(beta - true_beta) / true_beta < 0.20


def test_joint_recovery() -> None:
    """Two-channel synthetic: joint fit recovers all truths inside 90% CIs."""

    rng = np.random.default_rng(0)
    n = 200

    base_truth = 5_000.0
    a_alpha, a_beta = 1_000.0, 200.0
    b_alpha, b_beta = 2_000.0, 500.0

    spend_a = rng.uniform(0.0, 1_500.0, size=n)
    spend_b = rng.uniform(0.0, 4_000.0, size=n)
    revenue = (
        base_truth
        + a_alpha * np.log1p(spend_a / a_beta)
        + b_alpha * np.log1p(spend_b / b_beta)
        + rng.normal(0.0, 200.0, size=n)
    )

    weeks = pd.date_range("2020-01-01", periods=n, freq="W")
    df = pd.DataFrame(
        {
            "week": list(weeks) * 2,
            "channel": ["a"] * n + ["b"] * n,
            "spend": np.concatenate([spend_a, spend_b]),
            "revenue": np.concatenate([revenue, revenue]),
        }
    )

    fit = fit_joint(df)
    boot = bootstrap_joint(df, n_resamples=120, seed=0)

    assert fit.r_squared > 0.95
    assert boot.alpha_ci["a"][0] <= a_alpha <= boot.alpha_ci["a"][1]
    assert boot.beta_ci["a"][0] <= a_beta <= boot.beta_ci["a"][1]
    assert boot.alpha_ci["b"][0] <= b_alpha <= boot.alpha_ci["b"][1]
    assert boot.beta_ci["b"][0] <= b_beta <= boot.beta_ci["b"][1]
    assert boot.base_ci[0] <= base_truth <= boot.base_ci[1]


def test_untested_channel_is_surfaced() -> None:
    """A channel with all-zero spend should land in JointFit.untested."""

    rng = np.random.default_rng(1)
    n = 100
    weeks = pd.date_range("2020-01-01", periods=n, freq="W")

    spend_a = rng.uniform(0.0, 1_000.0, size=n)
    revenue = 1_000.0 + 500.0 * np.log1p(spend_a / 100.0)
    df = pd.DataFrame(
        {
            "week": list(weeks) * 2,
            "channel": ["a"] * n + ["dead"] * n,
            "spend": np.concatenate([spend_a, np.zeros(n)]),
            "revenue": np.concatenate([revenue, revenue]),
        }
    )

    fit = fit_joint(df)

    assert fit.untested == ["dead"]
    assert "dead" not in fit.channels
    assert "a" in fit.channels
