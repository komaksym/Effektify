"""Exporter tests: payload shape, partitioning, sanitization, loud failures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.agent_schemas import AgentOutput, ChannelRecommendation
from src.export_ui_payload import (
    build_brief_payload,
    load_agent_output,
    sanitize_rich_text,
)
from src.optimizer import optimize_allocation, to_dict


def _fake_cache(
    channels: dict[str, dict[str, Any]],
    base: float = 1000.0,
    untested: list[str] | None = None,
) -> dict:
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
            "r_squared": 0.61,
        },
        "untested": list(untested or []),
        "meta": {"n_resamples": 10, "n_succeeded": 10, "seed": 0},
    }


def _agent_output_for(optimizer_dict: dict) -> AgentOutput:
    recommendations: list[ChannelRecommendation] = []
    for channel in optimizer_dict["channels"]:
        name = channel["name"]
        current = channel["current_spend"]
        recommended = channel["recommended_spend"]
        delta = channel["delta"]

        if name == "facebook":
            reasoning = (
                "Cut from <b>$3,000</b> to $1,600 and keep "
                "<em>the same</em> total budget."
            )
            action = "decrease"
            confidence = "high"
        elif name == "search":
            reasoning = (
                "Raise from $5,000 to <b>$6,400</b> for a "
                f"${delta:,.0f} modeled lift."
            )
            action = "increase"
            confidence = "high"
        else:
            reasoning = (
                f"Hold at ${current:,.0f}; the optimizer keeps "
                f"{name} at ${recommended:,.0f}."
            )
            action = "hold"
            confidence = "low"

        recommendations.append(
            ChannelRecommendation(
                channel=name,
                action=action,
                reasoning=reasoning,
                confidence=confidence,
            )
        )

    return AgentOutput(
        headline="Shift the same <b>$12,000</b> budget toward search.",
        recommendations=recommendations,
        untested_channels=["tv"],
        warnings=["Preserve <b>bold</b> but drop <span>other</span> tags."],
    )


def test_build_brief_payload_shape_and_partitioning() -> None:
    cache = _fake_cache(
        channels={
            "facebook": {
                "alpha": 900.0,
                "beta": 300.0,
                "historical_spend_max": 5_000.0,
                "partial_r_squared": 0.11,
            },
            "search": {
                "alpha": 3_000.0,
                "beta": 1_000.0,
                "historical_spend_max": 5_000.0,
                "partial_r_squared": 0.12,
            },
            "print": {
                "alpha": 100.0,
                "beta": 100.0,
                "historical_spend_max": 5_000.0,
                "partial_r_squared": 0.001,
            },
        },
        untested=["tv"],
    )
    actual = {
        "facebook": 3_000.0,
        "search": 5_000.0,
        "print": 4_000.0,
        "tv": 0.0,
    }
    optimizer_dict = to_dict(optimize_allocation(cache, actual))
    agent_output = _agent_output_for(optimizer_dict)

    brief = build_brief_payload(
        cache=cache,
        optimizer_dict=optimizer_dict,
        agent_output=agent_output,
        actual_allocation=actual,
        actual_revenue=50_000.0,
    )

    assert set(brief) == {"meta", "hero", "channels", "asides"}
    assert brief["meta"]["week_label"] == "Week of Jun 25, 2018"
    assert brief["meta"]["bootstrap_count"] == 10
    assert brief["hero"]["budget"] == pytest.approx(12_000.0)
    assert brief["hero"]["delta_point"] == pytest.approx(
        optimizer_dict["total_delta"]
    )
    assert brief["hero"]["delta_low"] == pytest.approx(
        optimizer_dict["total_delta_ci"][0]
    )
    assert brief["hero"]["delta_high"] == pytest.approx(
        optimizer_dict["total_delta_ci"][1]
    )
    assert brief["hero"]["total_current_revenue"] == pytest.approx(50_000.0)
    assert brief["hero"]["total_recommended_revenue"] == pytest.approx(
        50_000.0 + optimizer_dict["total_delta"]
    )

    assert [row["channel"] for row in brief["channels"]] == [
        "search",
        "facebook",
    ]
    assert brief["asides"]["untested_channels"] == ["tv"]
    assert [row["channel"] for row in brief["asides"]["low_confidence_channels"]] == [
        "print"
    ]


def test_sanitization_preserves_bold_and_strips_other_html() -> None:
    assert (
        sanitize_rich_text("Keep <b>this</b> and drop <em>that</em>.")
        == "Keep <b>this</b> and drop that."
    )

    cache = _fake_cache(
        channels={
            "facebook": {
                "alpha": 900.0,
                "beta": 300.0,
                "historical_spend_max": 5_000.0,
                "partial_r_squared": 0.11,
            },
            "search": {
                "alpha": 3_000.0,
                "beta": 1_000.0,
                "historical_spend_max": 5_000.0,
                "partial_r_squared": 0.12,
            },
            "print": {
                "alpha": 100.0,
                "beta": 100.0,
                "historical_spend_max": 5_000.0,
                "partial_r_squared": 0.001,
            },
        },
    )
    actual = {"facebook": 3_000.0, "search": 5_000.0, "print": 4_000.0}
    optimizer_dict = to_dict(optimize_allocation(cache, actual))
    agent_output = _agent_output_for(optimizer_dict)

    brief = build_brief_payload(
        cache=cache,
        optimizer_dict=optimizer_dict,
        agent_output=agent_output,
        actual_allocation=actual,
        actual_revenue=50_000.0,
    )

    facebook = next(
        row for row in brief["channels"] if row["channel"] == "facebook"
    )
    assert facebook["grounded_reasoning"].startswith(
        "Cut from <b>$3,000</b> to $1,600 and keep the same"
    )
    assert brief["meta"]["warnings"] == [
        "Preserve <b>bold</b> but drop other tags."
    ]


def test_load_agent_output_errors_when_missing(tmp_path: Path) -> None:
    missing = tmp_path / "agent_output.json"
    with pytest.raises(FileNotFoundError, match="missing agent output cache"):
        load_agent_output(missing)


def test_build_brief_payload_errors_when_channel_names_do_not_line_up() -> None:
    cache = _fake_cache(
        channels={
            "facebook": {
                "alpha": 900.0,
                "beta": 300.0,
                "historical_spend_max": 5_000.0,
                "partial_r_squared": 0.11,
            },
            "search": {
                "alpha": 3_000.0,
                "beta": 1_000.0,
                "historical_spend_max": 5_000.0,
                "partial_r_squared": 0.12,
            },
        }
    )
    actual = {"facebook": 3_000.0, "search": 5_000.0}
    optimizer_dict = to_dict(optimize_allocation(cache, actual))
    broken = AgentOutput(
        headline="Broken",
        recommendations=[
            ChannelRecommendation(
                channel="search",
                action="increase",
                reasoning="Raise from $5,000 to $6,400.",
                confidence="high",
            )
        ],
        untested_channels=[],
        warnings=[],
    )

    with pytest.raises(ValueError, match="missing recommendations for: facebook"):
        build_brief_payload(cache, optimizer_dict, broken, actual, 50_000.0)


def test_build_brief_payload_errors_on_duplicate_channel_names() -> None:
    cache = _fake_cache(
        channels={
            "facebook": {
                "alpha": 900.0,
                "beta": 300.0,
                "historical_spend_max": 5_000.0,
                "partial_r_squared": 0.11,
            }
        }
    )
    actual = {"facebook": 3_000.0}
    optimizer_dict = to_dict(optimize_allocation(cache, actual))
    broken = AgentOutput(
        headline="Broken",
        recommendations=[
            ChannelRecommendation(
                channel="facebook",
                action="decrease",
                reasoning="Cut from $3,000 to $2,500.",
                confidence="high",
            ),
            ChannelRecommendation(
                channel="facebook",
                action="decrease",
                reasoning="Cut from $3,000 to $2,000.",
                confidence="high",
            ),
        ],
        untested_channels=[],
        warnings=[],
    )

    with pytest.raises(ValueError, match="duplicate channel names in agent output"):
        build_brief_payload(cache, optimizer_dict, broken, actual, 50_000.0)
