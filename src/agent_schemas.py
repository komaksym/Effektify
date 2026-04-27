"""Pydantic schemas for the agent layer's structured output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ActionLabel = Literal["increase", "decrease", "hold", "untested"]
ConfidenceLabel = Literal["high", "medium", "low"]


class ChannelRecommendation(BaseModel):
    """Per-channel English recommendation translated from the optimizer dict.

    `reasoning` must reference specific numbers from the input — the
    grounding check enforces this at validation time.
    """

    channel: str
    action: ActionLabel
    reasoning: str = Field(
        ...,
        description=(
            "At most two sentences. Cite specific numbers from the optimizer "
            "input (current spend, recommended spend, marginal ROAS, delta)."
        ),
    )
    confidence: ConfidenceLabel


class AgentOutput(BaseModel):
    """Final agent output written to cache/agent_output.json."""

    headline: str
    recommendations: list[ChannelRecommendation]
    untested_channels: list[str]
    warnings: list[str]
