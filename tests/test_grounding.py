"""Grounding regex + tolerance check tests."""

from __future__ import annotations

from src.grounding import (
    Ungrounded,
    check_grounded,
    collect_grounded_values,
    extract_numbers,
)


def test_extract_numbers_handles_dollars_commas_decimals() -> None:
    text = "Cut facebook from $3,757 to $2,271 (saving $1,486 or 39.5%)."

    nums = extract_numbers(text)

    assert 3757.0 in nums
    assert 2271.0 in nums
    assert 1486.0 in nums
    assert 39.5 in nums


def test_extract_numbers_ignores_pure_punctuation() -> None:
    assert extract_numbers("nothing here -- $") == []


def test_collect_grounded_values_walks_nested() -> None:
    payload = {
        "channels": [
            {"name": "fb", "current_spend": 3757.0, "delta": -71441.0},
            {"name": "search", "current_spend": 5160.0},
        ],
        "totals": {"delta_ci": [-46191, 70807]},
        "is_demo": True,  # bool: must be ignored
    }

    found = collect_grounded_values(payload)

    assert 3757.0 in found
    assert -71441.0 in found
    assert 70807.0 in found
    assert 1.0 not in found  # bool True must NOT leak in as 1.0


def test_check_grounded_passes_within_tolerance() -> None:
    text = "Recommended spend is $2,272 (was $3,757)."
    payload = {"recommended": 2271.0, "current": 3757.0}

    ungrounded = check_grounded(text, payload, tolerance=0.01)

    # 2272 is within 1% of 2271 — should pass.
    assert ungrounded == []


def test_check_grounded_flags_hallucinated_value() -> None:
    text = "Facebook will yield an extra $99,999 next week."
    payload = {"current": 3757.0, "delta": 20468.0}

    ungrounded = check_grounded(text, payload, tolerance=0.01)

    assert len(ungrounded) == 1
    assert isinstance(ungrounded[0], Ungrounded)
    assert ungrounded[0].value == 99999.0
    assert ungrounded[0].relative_diff > 0.01
