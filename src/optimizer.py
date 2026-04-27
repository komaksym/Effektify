"""Constrained reallocation across active channels + diagnosis labels.

Reads the M1 cache, classifies channels (active / low-confidence / untested),
runs an SLSQP-constrained reallocation on the active set against the same
total budget, and reports per-channel marginal-ROAS-based diagnoses plus a
bootstrap CI on the total revenue delta.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from src.config import (
    BUDGET_FRACTION_MAX,
    BUDGET_FRACTION_MIN,
    HISTORICAL_SPEND_MULTIPLIER,
    NEAR_OPTIMUM_TOLERANCE,
    PARTIAL_R2_THRESHOLD,
)

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "cache" / "channel_curves.json"


Diagnosis = Literal[
    "saturated",
    "room to grow",
    "near optimum",
    "flat / needs more data",
    "untested",
]


@dataclass(frozen=True, slots=True)
class ChannelRecommendation:
    name: str
    current_spend: float
    recommended_spend: float
    current_revenue_contribution: float
    recommended_revenue_contribution: float
    delta: float
    marginal_roas_at_current: float
    marginal_roas_at_recommended: float
    diagnosis: Diagnosis
    in_observed_range: bool
    partial_r_squared: float


@dataclass(frozen=True, slots=True)
class OptimizerOutput:
    channels: list[ChannelRecommendation]
    untested_channels: list[str]
    low_confidence_channels: list[str]
    total_current_revenue: float
    total_recommended_revenue: float
    total_delta: float
    total_delta_ci: tuple[float, float]


def load_cache(path: Path = CACHE_PATH) -> dict:
    return json.loads(path.read_text())


def _saturation(spend: float, alpha: float, beta: float) -> float:
    return float(alpha * np.log1p(spend / beta))


def _marginal_roas(spend: float, alpha: float, beta: float) -> float:
    return float(alpha / (beta + spend))


_DOLLAR_EPS = 1.0


def _diagnose(current: float, recommended: float) -> Diagnosis:
    """Spend-ratio diagnosis (see config.py for why we don't use absolute ROAS)."""

    if current < _DOLLAR_EPS and recommended < _DOLLAR_EPS:
        return "near optimum"
    if current < _DOLLAR_EPS:
        return "room to grow"
    if recommended < _DOLLAR_EPS:
        return "saturated"

    rel_change = (recommended - current) / current
    if abs(rel_change) <= NEAR_OPTIMUM_TOLERANCE:
        return "near optimum"

    return "saturated" if rel_change < 0 else "room to grow"


def _classify(
    channel_curves: dict[str, dict],
) -> tuple[list[str], list[str]]:
    """Split tested channels into (active, low_confidence) by partial R²."""

    low_conf = sorted(
        name
        for name, c in channel_curves.items()
        if c["partial_r_squared"] < PARTIAL_R2_THRESHOLD
    )
    active = sorted(c for c in channel_curves if c not in low_conf)

    return active, low_conf


def _solve_slsqp(
    alphas: NDArray[np.float64],
    betas: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    target_sum: float,
    x0: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Maximise Σ α_c · log(1 + s_c/β_c) subject to box bounds + Σ s_c = target."""

    def neg_revenue(s: NDArray[np.float64]) -> float:
        return -float(np.sum(alphas * np.log1p(s / betas)))

    def neg_jac(s: NDArray[np.float64]) -> NDArray[np.float64]:
        return -alphas / (betas + s)

    constraint = {
        "type": "eq",
        "fun": lambda s: float(s.sum() - target_sum),
        "jac": lambda s: np.ones_like(s),
    }

    res = minimize(
        neg_revenue,
        x0=x0,
        jac=neg_jac,
        method="SLSQP",
        bounds=list(zip(lower, upper, strict=True)),
        constraints=[constraint],
        options={"maxiter": 300, "ftol": 1e-7},
    )

    return res.x if res.success else x0


def _bootstrap_delta(
    cache: dict,
    actual_allocation: dict[str, float],
    active_names: list[str],
    recommended: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-bootstrap delta = Σ_c α_c[i]·(log(1+s_rec/β_c[i]) − log(1+s_cur/β_c[i]))."""

    channel_curves = cache["channel_curves"]
    n_resamples = len(cache["joint"]["base_samples"])
    rec_lookup = dict(zip(active_names, recommended, strict=True))

    deltas = np.zeros(n_resamples)
    for name, c in channel_curves.items():
        alpha_samples = np.asarray(c["alpha_samples"], dtype=float)
        beta_samples = np.asarray(c["beta_samples"], dtype=float)
        s_current = float(actual_allocation.get(name, 0.0))
        s_rec = float(rec_lookup.get(name, s_current))

        deltas += alpha_samples * (
            np.log1p(s_rec / beta_samples) - np.log1p(s_current / beta_samples)
        )

    return deltas


def _bound_or_freeze(
    name: str,
    actual_allocation: dict[str, float],
    curve: dict,
    diagnosis: Diagnosis,
) -> ChannelRecommendation:
    """Construct a no-change recommendation for a frozen (non-optimised) channel."""

    s = float(actual_allocation.get(name, 0.0))
    alpha, beta = curve["alpha"], curve["beta"]
    rev = _saturation(s, alpha, beta)
    mr = _marginal_roas(s, alpha, beta)

    return ChannelRecommendation(
        name=name,
        current_spend=s,
        recommended_spend=s,
        current_revenue_contribution=rev,
        recommended_revenue_contribution=rev,
        delta=0.0,
        marginal_roas_at_current=mr,
        marginal_roas_at_recommended=mr,
        diagnosis=diagnosis,
        in_observed_range=s <= curve["historical_spend_max"],
        partial_r_squared=curve["partial_r_squared"],
    )


def optimize_allocation(
    cache: dict,
    actual_allocation: dict[str, float],
) -> OptimizerOutput:
    """Run constrained reallocation. Frozen channels keep their current spend."""

    channel_curves: dict[str, dict] = cache["channel_curves"]
    joint: dict = cache["joint"]
    untested: list[str] = list(cache.get("untested", []))

    active_names, low_confidence = _classify(channel_curves)
    if not active_names:
        raise ValueError("no active channels above partial R² threshold")

    total_budget = float(sum(actual_allocation.values()))
    frozen_spend = sum(
        float(actual_allocation.get(c, 0.0)) for c in low_confidence + untested
    )
    optimisable_budget = total_budget - frozen_spend

    alphas = np.array([channel_curves[c]["alpha"] for c in active_names])
    betas = np.array([channel_curves[c]["beta"] for c in active_names])
    hist_max = np.array(
        [channel_curves[c]["historical_spend_max"] for c in active_names]
    )

    # Lower bound only fires for channels currently in use; dormant channels
    # ($0 this week) stay free to remain at 0.
    lower = np.array(
        [
            BUDGET_FRACTION_MIN * total_budget
            if float(actual_allocation.get(c, 0.0)) > 0
            else 0.0
            for c in active_names
        ]
    )
    upper = np.minimum(
        BUDGET_FRACTION_MAX * total_budget,
        HISTORICAL_SPEND_MULTIPLIER * hist_max,
    )
    # Ensure feasibility: lower ≤ upper, and Σ lower ≤ target ≤ Σ upper.
    lower = np.minimum(lower, upper)
    if lower.sum() > optimisable_budget:
        lower = np.zeros_like(lower)
    if upper.sum() < optimisable_budget:
        upper = np.maximum(upper, optimisable_budget)

    x0 = np.array([float(actual_allocation.get(c, 0.0)) for c in active_names])
    x0 = np.clip(x0, lower, upper)
    if x0.sum() > 0:
        x0 = x0 * (optimisable_budget / x0.sum())
        x0 = np.clip(x0, lower, upper)

    recommended = _solve_slsqp(
        alphas, betas, lower, upper, optimisable_budget, x0
    )
    # Clean up sub-dollar SLSQP residuals so diagnosis isn't tripped by 1e-12.
    recommended = np.where(recommended < _DOLLAR_EPS, 0.0, recommended)

    channels: list[ChannelRecommendation] = []

    for j, name in enumerate(active_names):
        c = channel_curves[name]
        alpha, beta = c["alpha"], c["beta"]
        s_current = float(actual_allocation.get(name, 0.0))
        s_rec = float(recommended[j])

        rev_current = _saturation(s_current, alpha, beta)
        rev_rec = _saturation(s_rec, alpha, beta)
        mr_current = _marginal_roas(s_current, alpha, beta)
        mr_rec = _marginal_roas(s_rec, alpha, beta)

        channels.append(
            ChannelRecommendation(
                name=name,
                current_spend=s_current,
                recommended_spend=s_rec,
                current_revenue_contribution=rev_current,
                recommended_revenue_contribution=rev_rec,
                delta=rev_rec - rev_current,
                marginal_roas_at_current=mr_current,
                marginal_roas_at_recommended=mr_rec,
                diagnosis=_diagnose(s_current, s_rec),
                in_observed_range=s_rec <= c["historical_spend_max"],
                partial_r_squared=c["partial_r_squared"],
            )
        )

    for name in low_confidence:
        channels.append(
            _bound_or_freeze(
                name,
                actual_allocation,
                channel_curves[name],
                "flat / needs more data",
            )
        )

    channels.sort(key=lambda r: r.name)

    base = float(joint["base"])
    total_current = base + sum(r.current_revenue_contribution for r in channels)
    total_recommended = base + sum(
        r.recommended_revenue_contribution for r in channels
    )
    total_delta = total_recommended - total_current

    delta_samples = _bootstrap_delta(
        cache, actual_allocation, active_names, recommended
    )
    delta_ci = (
        float(np.percentile(delta_samples, 5)),
        float(np.percentile(delta_samples, 95)),
    )

    return OptimizerOutput(
        channels=channels,
        untested_channels=untested,
        low_confidence_channels=low_confidence,
        total_current_revenue=total_current,
        total_recommended_revenue=total_recommended,
        total_delta=total_delta,
        total_delta_ci=delta_ci,
    )


def to_dict(out: OptimizerOutput) -> dict:
    """Serialise for downstream JSON / agent input."""

    return {
        "channels": [asdict(c) for c in out.channels],
        "untested_channels": list(out.untested_channels),
        "low_confidence_channels": list(out.low_confidence_channels),
        "total_current_revenue": out.total_current_revenue,
        "total_recommended_revenue": out.total_recommended_revenue,
        "total_delta": out.total_delta,
        "total_delta_ci": list(out.total_delta_ci),
    }


def main() -> None:
    """CLI: optimise the demo week and pretty-print."""

    from src.config import DEMO_WEEK
    from src.ingest import load_robyn

    cache = load_cache()
    df = load_robyn()

    week_row = df[df["week"] == DEMO_WEEK]
    if week_row.empty:
        raise SystemExit(f"DEMO_WEEK {DEMO_WEEK} not in dataset")

    actual = dict(zip(week_row["channel"], week_row["spend"], strict=True))
    out = optimize_allocation(cache, actual)

    print(f"Demo week: {DEMO_WEEK}")
    print(f"  total spend  = ${sum(actual.values()):>12,.0f}  (unchanged)")
    print(
        f"  total delta  = ${out.total_delta:>+12,.0f}  "
        f"90% CI = [${out.total_delta_ci[0]:>+10,.0f}, "
        f"${out.total_delta_ci[1]:>+10,.0f}]"
    )
    print()

    print(
        f"  {'channel':>10}  {'current':>12}  {'recommended':>12}  "
        f"{'Δrev':>12}  {'mr_rec':>8}  diagnosis"
    )
    for ch in out.channels:
        print(
            f"  {ch.name:>10}  ${ch.current_spend:>11,.0f}  "
            f"${ch.recommended_spend:>11,.0f}  "
            f"${ch.delta:>+11,.0f}  {ch.marginal_roas_at_recommended:>8.2f}  "
            f"{ch.diagnosis}"
        )

    if out.untested_channels:
        print(f"  untested: {', '.join(out.untested_channels)}")


if __name__ == "__main__":
    main()
