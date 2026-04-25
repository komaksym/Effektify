"""Bootstrap confidence intervals for the joint saturation model.

Resamples weeks with replacement, refits jointly each time, and reports the
empirical 90% CI (5th–95th percentile) of every parameter — base, α_c, β_c.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.curve_fit import fit_joint_matrix, to_matrix


@dataclass(frozen=True, slots=True)
class JointBootstrap:
    base_samples: NDArray[np.float64]
    alpha_samples: dict[str, NDArray[np.float64]]
    beta_samples: dict[str, NDArray[np.float64]]
    base_ci: tuple[float, float]
    alpha_ci: dict[str, tuple[float, float]]
    beta_ci: dict[str, tuple[float, float]]
    n_succeeded: int


def _ci(arr: NDArray[np.float64]) -> tuple[float, float]:
    return float(np.percentile(arr, 5)), float(np.percentile(arr, 95))


def bootstrap_joint(
    df: pd.DataFrame,
    n_resamples: int = 500,
    seed: int = 42,
) -> JointBootstrap:
    """Refit ``n_resamples`` times on bootstrap-resampled weeks."""

    revenue, spend_mat, channels, _ = to_matrix(df)
    if not channels:
        raise ValueError("no active channels to bootstrap")

    rng = np.random.default_rng(seed)
    n_weeks = revenue.size

    base_list: list[float] = []
    alpha_lists: dict[str, list[float]] = {c: [] for c in channels}
    beta_lists: dict[str, list[float]] = {c: [] for c in channels}

    for _ in range(n_resamples):
        idx = rng.integers(0, n_weeks, size=n_weeks)
        try:
            base, alphas, betas, _ = fit_joint_matrix(
                revenue[idx], spend_mat[idx]
            )
        except (RuntimeError, ValueError):
            continue

        base_list.append(base)
        for j, name in enumerate(channels):
            alpha_lists[name].append(float(alphas[j]))
            beta_lists[name].append(float(betas[j]))

    if not base_list:
        raise RuntimeError("all bootstrap resamples failed to converge")

    base_arr = np.asarray(base_list, dtype=float)
    alpha_arr = {c: np.asarray(v, dtype=float) for c, v in alpha_lists.items()}
    beta_arr = {c: np.asarray(v, dtype=float) for c, v in beta_lists.items()}

    return JointBootstrap(
        base_samples=base_arr,
        alpha_samples=alpha_arr,
        beta_samples=beta_arr,
        base_ci=_ci(base_arr),
        alpha_ci={c: _ci(v) for c, v in alpha_arr.items()},
        beta_ci={c: _ci(v) for c, v in beta_arr.items()},
        n_succeeded=len(base_list),
    )
