"""Saturation curve fitting.

Two public entry points:

* ``fit_curve(spend, revenue)`` — univariate fit, used for toy recovery tests.
* ``fit_joint(df)`` — production fit. Solves one nonlinear least-squares for
  ``revenue_t = base + Σ_c α_c · log(1 + s_c,t / β_c)``. Per-channel (α, β)
  recovered jointly so each one only has to explain its own marginal effect,
  not all of revenue. Channels with zero historical spend are surfaced as
  ``untested`` and excluded from the fit (their parameters would be
  unidentifiable).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import curve_fit, least_squares

from src.simulator import saturation


@dataclass(frozen=True, slots=True)
class CurveFit:
    alpha: float
    beta: float
    n_obs: int
    spend_max: float


@dataclass(frozen=True, slots=True)
class JointFit:
    base: float
    channels: dict[str, CurveFit]
    untested: list[str]
    r_squared: float


def fit_curve(
    spend: ArrayLike, revenue: ArrayLike
) -> tuple[float, float, float]:
    """Univariate fit. Returns ``(α, β, R²)``.

    Used by the toy recovery test where there's a single channel and no
    baseline. Production fits go through ``fit_joint`` because a univariate
    per-channel fit on aggregate revenue suffers from omitted-variable bias.
    """

    spend_arr = np.asarray(spend, dtype=float)
    revenue_arr = np.asarray(revenue, dtype=float)

    pos = spend_arr[spend_arr > 0]
    beta0 = float(np.median(pos)) if pos.size else 1.0
    alpha0 = float(np.max(revenue_arr)) if revenue_arr.size else 1.0

    (alpha, beta), _ = curve_fit(
        saturation,
        spend_arr,
        revenue_arr,
        p0=(alpha0, beta0),
        bounds=([0.0, 1e-6], [np.inf, np.inf]),
        maxfev=10_000,
    )

    pred = saturation(spend_arr, alpha, beta)
    ss_res = float(np.sum((revenue_arr - pred) ** 2))
    ss_tot = float(np.sum((revenue_arr - revenue_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return float(alpha), float(beta), float(r2)


def to_matrix(
    df: pd.DataFrame,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str], list[str]]:
    """Pivot long ``(week, channel, spend, revenue)`` to matrix form.

    Returns ``(revenue, spend_mat, active_channels, untested)``. ``spend_mat``
    has shape ``(T, K_active)``. Channels whose spend is zero in every row are
    moved into ``untested`` rather than fed into the optimizer (their α/β
    would be unidentifiable since their column is the zero vector).
    """

    pivot = (
        df.pivot(index="week", columns="channel", values="spend")
        .sort_index()
        .fillna(0.0)
    )

    untested = sorted(c for c in pivot.columns if (pivot[c] > 0).sum() == 0)
    active = [c for c in pivot.columns if c not in untested]
    pivot = pivot[active]

    revenue = (
        df.drop_duplicates("week")
        .sort_values("week")["revenue"]
        .to_numpy(dtype=float)
    )
    spend_mat = pivot.to_numpy(dtype=float)

    return revenue, spend_mat, list(active), untested


def fit_joint_matrix(
    revenue: NDArray[np.float64],
    spend_mat: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64], NDArray[np.float64], float]:
    """Core joint least-squares fit. Returns ``(base, alphas, betas, R²)``."""

    t, k = spend_mat.shape
    spend_max = spend_mat.max(axis=0)

    base0 = float(revenue.min())
    alpha0 = float(revenue.max()) / max(k, 1)
    beta0 = np.array(
        [
            float(np.median(col[col > 0])) if (col > 0).any() else 1.0
            for col in spend_mat.T
        ],
        dtype=float,
    )

    # Cap β at 10× per-channel max spend: beyond that, log(1+s/β) is
    # near-linear in the data range, so (α, β) become unidentifiable —
    # only the ratio α/β is determined and the fit drifts to infinity.
    beta_upper = np.maximum(10.0 * spend_max, 1.0)

    x0 = np.concatenate([[base0], np.full(k, alpha0), beta0])
    lower = np.concatenate([[0.0], np.zeros(k), np.full(k, 1e-6)])
    upper = np.concatenate([[np.inf], np.full(k, np.inf), beta_upper])

    def residuals(p: NDArray[np.float64]) -> NDArray[np.float64]:
        base = p[0]
        alphas = p[1 : 1 + k]
        betas = p[1 + k :]

        contributions = alphas * np.log1p(spend_mat / betas)
        pred = base + contributions.sum(axis=1)

        return revenue - pred

    res = least_squares(
        residuals, x0=x0, bounds=(lower, upper), max_nfev=20_000
    )
    base = float(res.x[0])
    alphas = res.x[1 : 1 + k].astype(float)
    betas = res.x[1 + k :].astype(float)

    pred = base + (alphas * np.log1p(spend_mat / betas)).sum(axis=1)
    ss_res = float(np.sum((revenue - pred) ** 2))
    ss_tot = float(np.sum((revenue - revenue.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return base, alphas, betas, float(r2)


def fit_joint(df: pd.DataFrame) -> JointFit:
    """Joint MMM-lite fit across all active channels."""

    revenue, spend_mat, channels, untested = to_matrix(df)
    if not channels:
        raise ValueError("no active channels with positive spend to fit")

    base, alphas, betas, r2 = fit_joint_matrix(revenue, spend_mat)

    channel_fits = {
        name: CurveFit(
            alpha=float(alphas[j]),
            beta=float(betas[j]),
            n_obs=int(spend_mat.shape[0]),
            spend_max=float(spend_mat[:, j].max()),
        )
        for j, name in enumerate(channels)
    }

    return JointFit(
        base=float(base),
        channels=channel_fits,
        untested=untested,
        r_squared=float(r2),
    )


def partial_r_squared(
    revenue: NDArray[np.float64],
    spend_mat: NDArray[np.float64],
    full_r_squared: float,
) -> dict[int, float]:
    """For each column, the R² drop when that column is removed from the fit.

    A channel that meaningfully contributes shows a large drop; a channel
    that adds no signal shows ~0. Used as the per-channel quality gate
    (replaces the PRD's per-channel univariate R² < 0.3 rule).
    """

    out: dict[int, float] = {}
    k = spend_mat.shape[1]
    for j in range(k):
        if k == 1:
            out[j] = full_r_squared
            continue

        sub_mat = np.delete(spend_mat, j, axis=1)
        try:
            _, _, _, r2_without = fit_joint_matrix(revenue, sub_mat)
        except (RuntimeError, ValueError):
            r2_without = 0.0
        out[j] = float(full_r_squared - r2_without)

    return out
