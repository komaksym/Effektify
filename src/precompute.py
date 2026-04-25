"""Joint-fit + bootstrap + cache + per-channel plots.

CLI: ``uv run python -m src.precompute``
Outputs:
  - ``cache/channel_curves.json``     (per-channel α/β/CIs + joint base + R²)
  - ``reports/curves/{channel}.png``  (sanity plots, one per channel)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.bootstrap import JointBootstrap, bootstrap_joint
from src.curve_fit import JointFit, fit_joint, to_matrix
from src.ingest import load_robyn

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "cache" / "channel_curves.json"
PLOTS_DIR = ROOT / "reports" / "curves"

N_RESAMPLES = 500
SEED = 42


def _attributed_revenue(
    name: str,
    revenue: NDArray[np.float64],
    spend_mat: NDArray[np.float64],
    channels: list[str],
    fit: JointFit,
) -> NDArray[np.float64]:
    """Revenue minus base and minus all *other* channels' fitted contributions.

    This is what the joint model claims this channel is responsible for —
    i.e. the y-target the per-channel curve actually has to explain.
    """

    other = np.zeros_like(revenue)
    for j, c in enumerate(channels):
        if c == name:
            continue
        cf = fit.channels[c]
        other += cf.alpha * np.log1p(spend_mat[:, j] / cf.beta)

    return revenue - fit.base - other


def _plot_curve(
    name: str,
    spend: NDArray[np.float64],
    attributed: NDArray[np.float64],
    fit: JointFit,
    boot: JointBootstrap,
) -> None:
    cf = fit.channels[name]
    grid = np.linspace(0.0, max(float(spend.max()), 1.0) * 1.2, 200)
    point = cf.alpha * np.log1p(grid / cf.beta)

    samples = np.stack(
        [
            a * np.log1p(grid / b)
            for a, b in zip(
                boot.alpha_samples[name],
                boot.beta_samples[name],
                strict=True,
            )
        ]
    )
    lo = np.percentile(samples, 5, axis=0)
    hi = np.percentile(samples, 95, axis=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        spend,
        attributed,
        s=10,
        alpha=0.5,
        label="attributed (rev − base − other channels)",
    )
    ax.plot(grid, point, "C1-", label="joint fit")
    ax.fill_between(
        grid, lo, hi, color="C1", alpha=0.2, label="90% bootstrap CI"
    )
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("spend ($)")
    ax.set_ylabel("attributed revenue ($)")
    ax.set_title(f"{name}  α={cf.alpha:,.0f}  β={cf.beta:,.0f}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / f"{name}.png", dpi=120)
    plt.close(fig)


def _build_cache(fit: JointFit, boot: JointBootstrap) -> dict[str, object]:
    channel_curves: dict[str, dict[str, object]] = {}
    for name, cf in fit.channels.items():
        channel_curves[name] = {
            "alpha": cf.alpha,
            "beta": cf.beta,
            "alpha_ci": list(boot.alpha_ci[name]),
            "beta_ci": list(boot.beta_ci[name]),
            "weeks_of_data": cf.n_obs,
            "historical_spend_max": cf.spend_max,
        }

    return {
        "channel_curves": channel_curves,
        "joint": {
            "base": fit.base,
            "base_ci": list(boot.base_ci),
            "r_squared": fit.r_squared,
        },
        "untested": list(fit.untested),
        "meta": {
            "n_resamples": N_RESAMPLES,
            "n_succeeded": boot.n_succeeded,
            "seed": SEED,
        },
    }


def main() -> None:
    df = load_robyn()

    fit = fit_joint(df)
    boot = bootstrap_joint(df, n_resamples=N_RESAMPLES, seed=SEED)

    revenue, spend_mat, channels, _ = to_matrix(df)
    for j, name in enumerate(channels):
        spend_col = spend_mat[:, j]
        attributed = _attributed_revenue(
            name, revenue, spend_mat, channels, fit
        )
        _plot_curve(name, spend_col, attributed, fit, boot)

    cache = _build_cache(fit, boot)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))

    print(f"wrote {CACHE_PATH.relative_to(ROOT)}")
    print(
        f"  joint   base={fit.base:>14,.0f}  "
        f"R²={fit.r_squared:.3f}  "
        f"({boot.n_succeeded}/{N_RESAMPLES} bootstraps converged)"
    )
    for name, cf in fit.channels.items():
        a_lo, a_hi = boot.alpha_ci[name]
        print(
            f"  {name:>10}  α={cf.alpha:>14,.0f}  "
            f"β={cf.beta:>10,.0f}  "
            f"α_ci=[{a_lo:>10,.0f}, {a_hi:>10,.0f}]"
        )
    if fit.untested:
        print(f"  untested: {', '.join(fit.untested)}")


if __name__ == "__main__":
    main()
