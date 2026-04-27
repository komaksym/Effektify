"""Optimizer tests: hand-built 3-channel recovery + classification + bounds."""

from __future__ import annotations

from typing import Any

from src.config import (
    BUDGET_FRACTION_MAX,
    BUDGET_FRACTION_MIN,
    HISTORICAL_SPEND_MULTIPLIER,
)
from src.optimizer import optimize_allocation


def _fake_cache(
    channels: dict[str, dict[str, Any]],
    base: float = 1000.0,
    untested: list[str] | None = None,
) -> dict:
    """Build a minimal cache. Bootstrap samples are constant copies of the
    point estimate — gives a zero-width delta CI, fine for these tests."""

    channel_curves = {}
    for name, spec in channels.items():
        channel_curves[name] = {
            "alpha": spec["alpha"],
            "beta": spec["beta"],
            "alpha_ci": [spec["alpha"], spec["alpha"]],
            "beta_ci": [spec["beta"], spec["beta"]],
            "alpha_samples": [spec["alpha"]] * 10,
            "beta_samples": [spec["beta"]] * 10,
            "partial_r_squared": spec.get("partial_r_squared", 0.10),
            "weeks_of_data": 200,
            "historical_spend_max": spec["historical_spend_max"],
        }

    return {
        "channel_curves": channel_curves,
        "joint": {
            "base": base,
            "base_ci": [base, base],
            "base_samples": [base] * 10,
            "r_squared": 0.6,
        },
        "untested": list(untested or []),
        "meta": {"n_resamples": 10, "n_succeeded": 10, "seed": 0},
    }


def test_three_channel_hand_built_recovery() -> None:
    """A: saturated → cut. B: weakly profitable → cut. C: room → boost (capped).

    With total budget $4,000 and the 60% upper bound, C is pushed to its
    cap of $2,400. The KKT condition then equalises marginal ROAS between
    A and B over the remaining $1,600:
        1000/(200 + s_A) = 2000/(2000 + s_B),  s_A + s_B = 1600
        ⇒ s_A ≈ $1,067, s_B ≈ $533.
    """

    cache = _fake_cache(
        channels={
            "A": {
                "alpha": 1000.0,
                "beta": 200.0,
                "historical_spend_max": 5000.0,
            },
            "B": {
                "alpha": 2000.0,
                "beta": 2000.0,
                "historical_spend_max": 5000.0,
            },
            "C": {
                "alpha": 5000.0,
                "beta": 1000.0,
                "historical_spend_max": 5000.0,
            },
        },
    )
    actual = {"A": 2000.0, "B": 1000.0, "C": 1000.0}

    out = optimize_allocation(cache, actual)

    by_name = {ch.name: ch for ch in out.channels}
    assert by_name["A"].diagnosis == "saturated"
    assert by_name["B"].diagnosis == "saturated"
    assert by_name["C"].diagnosis == "room to grow"

    assert abs(by_name["A"].recommended_spend - 1067.0) < 30.0
    assert abs(by_name["B"].recommended_spend - 533.0) < 30.0
    # C capped by the 60% upper bound: 60% of $4,000 = $2,400.
    assert abs(by_name["C"].recommended_spend - 2400.0) < 5.0

    # KKT: marginal ROAS equalised between A and B at the interior optimum.
    assert (
        abs(
            by_name["A"].marginal_roas_at_recommended
            - by_name["B"].marginal_roas_at_recommended
        )
        < 0.01
    )

    assert out.total_delta > 0


def test_bounds_are_respected() -> None:
    """No active-channel recommendation breaches 5% / 60% / 1.5×-historical."""

    cache = _fake_cache(
        channels={
            "A": {
                "alpha": 1000.0,
                "beta": 200.0,
                "historical_spend_max": 5_000.0,
            },
            "B": {
                "alpha": 2000.0,
                "beta": 2000.0,
                "historical_spend_max": 5_000.0,
            },
            "C": {
                "alpha": 5000.0,
                "beta": 1000.0,
                "historical_spend_max": 200.0,
            },
        },
    )
    actual = {"A": 1500.0, "B": 1500.0, "C": 1000.0}
    total_budget = sum(actual.values())

    out = optimize_allocation(cache, actual)

    for ch in out.channels:
        if ch.current_spend > 0:
            assert (
                ch.recommended_spend >= BUDGET_FRACTION_MIN * total_budget - 0.5
            )
        assert ch.recommended_spend <= BUDGET_FRACTION_MAX * total_budget + 0.5
        # 1.5× historical-spend ceiling: C is hist_max=$200, so cap at $300.
        if ch.name == "C":
            assert (
                ch.recommended_spend
                <= HISTORICAL_SPEND_MULTIPLIER * 200.0 + 0.5
            )


def test_untested_channels_pass_through() -> None:
    """Untested channels surface in `untested_channels`, never in `channels`."""

    cache = _fake_cache(
        channels={
            "A": {
                "alpha": 1000.0,
                "beta": 200.0,
                "historical_spend_max": 5_000.0,
            },
            "B": {
                "alpha": 2000.0,
                "beta": 1000.0,
                "historical_spend_max": 5_000.0,
            },
        },
        untested=["linkedin", "reddit"],
    )
    actual = {"A": 1000.0, "B": 1000.0}

    out = optimize_allocation(cache, actual)

    names = {ch.name for ch in out.channels}
    assert "linkedin" not in names
    assert "reddit" not in names
    assert sorted(out.untested_channels) == ["linkedin", "reddit"]


def test_low_confidence_channels_are_frozen() -> None:
    """Below the partial-R² threshold → keeps current spend, marked flat."""

    cache = _fake_cache(
        channels={
            "A": {
                "alpha": 1000.0,
                "beta": 200.0,
                "historical_spend_max": 5_000.0,
                "partial_r_squared": 0.10,
            },
            "B": {
                "alpha": 2000.0,
                "beta": 1000.0,
                "historical_spend_max": 5_000.0,
                "partial_r_squared": 0.10,
            },
            "noisy": {
                "alpha": 100.0,
                "beta": 50.0,
                "historical_spend_max": 1_000.0,
                "partial_r_squared": 0.001,
            },
        },
    )
    actual = {"A": 1000.0, "B": 1000.0, "noisy": 500.0}

    out = optimize_allocation(cache, actual)

    by_name = {ch.name: ch for ch in out.channels}
    assert "noisy" in out.low_confidence_channels
    assert by_name["noisy"].recommended_spend == 500.0
    assert by_name["noisy"].delta == 0.0
    assert by_name["noisy"].diagnosis == "flat / needs more data"
