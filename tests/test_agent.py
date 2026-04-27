"""Agent tests with deterministic fake diagnose/recommend functions.

Tests the LangGraph wiring + grounding + retry behaviour without hitting
the real DeepSeek API. Production wires `_make_diagnose` / `_make_recommend`
from `agent_graph.py`; tests inject hand-crafted Python functions.
"""

from __future__ import annotations

from src.agent_graph import build_graph
from src.agent_schemas import ChannelRecommendation
from src.grounding import Ungrounded


def _input_dict() -> dict:
    """Realistic mini optimizer dict for tests."""

    return {
        "channels": [ {
                "name": "facebook",
                "current_spend": 3757.0,
                "recommended_spend": 2271.0,
                "delta": -71441.0,
                "marginal_roas_at_recommended": 61.6,
                "partial_r_squared": 0.110,
            },
            {
                "name": "search",
                "current_spend": 5160.0,
                "recommended_spend": 6646.0,
                "delta": 91909.0,
                "marginal_roas_at_recommended": 61.6,
                "partial_r_squared": 0.154,
            },
        ],
        "untested_channels": ["linkedin"],
        "low_confidence_channels": [],
        "total_delta": 20468.0,
    }


def _invoke(graph, input_dict: dict):
    return graph.invoke(
        {
            "input_dict": input_dict,
            "diagnosis_summary": "",
            "draft": [],
            "last_offenses": [],
            "retries": 0,
        }
    )


def test_happy_path_returns_grounded_output() -> None:
    """Diagnose + recommend produce grounded prose → empty warnings."""

    def fake_diagnose(_: dict) -> str:
        return "+$20,468 weekly delta; 1 saturated, 1 room-to-grow."

    def fake_recommend(
        _input: dict, _summary: str, _offenses: list[Ungrounded]
    ) -> list[ChannelRecommendation]:
        return [
            ChannelRecommendation(
                channel="facebook",
                action="decrease",
                reasoning="Cut from $3,757 to $2,271 (delta -$71,441).",
                confidence="high",
            ),
            ChannelRecommendation(
                channel="search",
                action="increase",
                reasoning="Raise from $5,160 to $6,646 (delta +$91,909).",
                confidence="high",
            ),
        ]

    graph = build_graph(fake_diagnose, fake_recommend)
    final = _invoke(graph, _input_dict())

    output = final["output"]
    assert output.warnings == []
    assert len(output.recommendations) == 2
    assert output.untested_channels == ["linkedin"]
    assert "20,468" in output.headline


def test_hallucination_triggers_retry_and_fail_loud() -> None:
    """Two consecutive hallucinated drafts → final output has warnings."""

    calls: list[int] = []

    def fake_diagnose(_: dict) -> str:
        return "summary"

    def fake_recommend(
        _input: dict, _summary: str, offenses: list[Ungrounded]
    ) -> list[ChannelRecommendation]:
        # Every call returns the same hallucination; retry won't save us.
        calls.append(len(offenses))
        return [
            ChannelRecommendation(
                channel="facebook",
                action="decrease",
                reasoning="Will yield an extra $99,999 next week.",
                confidence="medium",
            )
        ]

    graph = build_graph(fake_diagnose, fake_recommend)
    final = _invoke(graph, _input_dict())

    output = final["output"]
    assert output.warnings, "expected warnings from fail-loud path"
    assert any(
        "99999" in w or "99,999" in w or "99999" in w for w in output.warnings
    )
    # First call had 0 prior offenses; second call (retry) had >0.
    assert calls == [0, 1] or (len(calls) == 2 and calls[1] > 0)


def test_retry_recovers_when_second_draft_is_clean() -> None:
    """First draft hallucinates, retry produces grounded prose → no warnings."""

    attempt = {"n": 0}

    def fake_diagnose(_: dict) -> str:
        return "summary"

    def fake_recommend(
        _input: dict, _summary: str, _offenses: list[Ungrounded]
    ) -> list[ChannelRecommendation]:
        attempt["n"] += 1
        if attempt["n"] == 1:
            return [
                ChannelRecommendation(
                    channel="facebook",
                    action="decrease",
                    reasoning="Will yield an extra $99,999 next week.",
                    confidence="medium",
                )
            ]
        return [
            ChannelRecommendation(
                channel="facebook",
                action="decrease",
                reasoning="Cut from $3,757 to $2,271.",
                confidence="high",
            )
        ]

    graph = build_graph(fake_diagnose, fake_recommend)
    final = _invoke(graph, _input_dict())

    output = final["output"]
    assert output.warnings == []
    assert attempt["n"] == 2
