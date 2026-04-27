"""LangGraph agent: diagnose → recommend → validate → format.

The LLM has only two jobs: (1) summarise the week, (2) translate the
optimizer dict into per-channel English recommendations. All numerical
work stays upstream in M1/M2. The grounding check enforces that the
prose cannot quote numbers absent from the structured input.

DeepSeek is reached via langchain-openai's ChatOpenAI with a base_url
override — DeepSeek exposes an OpenAI-compatible API.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from src.agent_schemas import AgentOutput, ChannelRecommendation
from src.grounding import Ungrounded, check_grounded

DEFAULT_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

DiagnoseFn = Callable[[dict], str]
RecommendFn = Callable[
    [dict, str, list[Ungrounded]], list[ChannelRecommendation]
]


class _RecommendationList(BaseModel):
    """Wrapper for structured-output: LangChain wants a single Pydantic root."""

    recommendations: list[ChannelRecommendation]


class AgentState(TypedDict, total=False):
    input_dict: dict
    diagnosis_summary: str
    draft: list[ChannelRecommendation]
    last_offenses: list[Ungrounded]
    retries: int
    output: AgentOutput


def _validate_node(state: AgentState) -> dict:
    """Run the grounding check across every draft recommendation's reasoning.

    Concatenates all ungrounded numbers into ``last_offenses``; an empty list
    means the draft is clean and the graph can route to ``format``.
    """

    all_ungrounded: list[Ungrounded] = []
    for rec in state["draft"]:
        all_ungrounded.extend(
            check_grounded(rec.reasoning, state["input_dict"])
        )

    return {"last_offenses": all_ungrounded}


def _route_after_validate(state: AgentState) -> str:
    """Routing decision after validation: format, retry, or fail-loud.

    Clean draft → ``format``. Dirty draft with no retry yet → ``retry``.
    Dirty draft after one retry → ``format`` (warnings surfaced in output).
    """

    if not state.get("last_offenses"):
        return "format"
    if state.get("retries", 0) < 1:
        return "retry"

    return "format"  # give up after one retry; warnings surface in format


def _retry_node(state: AgentState) -> dict:
    """Increment the retry counter; the next ``recommend`` call gets a critique prompt."""

    return {"retries": state.get("retries", 0) + 1}


def _format_node(state: AgentState) -> dict:
    """Assemble the final ``AgentOutput`` from accumulated state.

    Translates any leftover ``last_offenses`` into human-readable warnings so
    a reviewer sees exactly which numbers the agent invented when grounding
    failed twice.
    """

    offenses = state.get("last_offenses", [])
    warnings = [
        f"Ungrounded number cited: {u.value:g} "
        f"(nearest input value: {u.nearest:g}, off by {u.relative_diff:.1%})"
        for u in offenses
    ]

    output = AgentOutput(
        headline=state.get("diagnosis_summary", ""),
        recommendations=list(state.get("draft", [])),
        untested_channels=list(
            state["input_dict"].get("untested_channels", [])
        ),
        warnings=warnings,
    )

    return {"output": output}


def build_graph(diagnose_fn: DiagnoseFn, recommend_fn: RecommendFn):
    """Wire the 4-node graph with injectable diagnose/recommend functions.

    Production passes LLM-backed implementations; tests pass deterministic
    Python functions so the graph topology is verified without API calls.
    """

    def diagnose(state: AgentState) -> dict:
        """LangGraph adapter: pull `input_dict` out of state, hand to `diagnose_fn`."""

        return {"diagnosis_summary": diagnose_fn(state["input_dict"])}

    def recommend(state: AgentState) -> dict:
        """LangGraph adapter: invoke `recommend_fn` with state + prior offenses for retry."""

        recs = recommend_fn(
            state["input_dict"],
            state.get("diagnosis_summary", ""),
            list(state.get("last_offenses", [])),
        )
        return {"draft": recs}

    builder = StateGraph(AgentState)
    builder.add_node("diagnose", diagnose)
    builder.add_node("recommend", recommend)
    builder.add_node("validate", _validate_node)
    builder.add_node("retry", _retry_node)
    builder.add_node("format", _format_node)

    builder.set_entry_point("diagnose")
    builder.add_edge("diagnose", "recommend")
    builder.add_edge("recommend", "validate")
    builder.add_conditional_edges(
        "validate",
        _route_after_validate,
        {"format": "format", "retry": "retry"},
    )
    builder.add_edge("retry", "recommend")
    builder.add_edge("format", END)

    return builder.compile()


def _make_diagnose(llm: BaseChatModel) -> DiagnoseFn:
    """Real-LLM diagnose: short prose summary of this week's optimization."""

    def diagnose(input_dict: dict) -> str:
        """Send the optimizer dict to the LLM, return the prose summary string."""

        prompt = (
            "Summarise this week's marketing budget optimization in 1-2 "
            "sentences. State the total dollar delta and how many channels "
            "are saturated, room-to-grow, or frozen. Cite only numbers "
            "from the JSON.\n\nOptimizer output:\n"
            + json.dumps(input_dict, indent=2)
        )
        msg = llm.invoke(prompt)
        return msg.content if hasattr(msg, "content") else str(msg)

    return diagnose


def _make_recommend(llm: BaseChatModel) -> RecommendFn:
    """Real-LLM recommend with structured output (Pydantic-validated)."""

    structured = llm.with_structured_output(_RecommendationList)

    def recommend(
        input_dict: dict,
        diagnosis_summary: str,
        prior_offenses: list[Ungrounded],
    ) -> list[ChannelRecommendation]:
        """Ask the LLM for a structured-output list of per-channel recommendations.

        On retry (``prior_offenses`` non-empty) the prompt includes a critique
        clause naming the offending numbers so the model can correct itself.
        """

        retry_clause = ""
        if prior_offenses:
            offenses_str = ", ".join(f"{u.value:g}" for u in prior_offenses)
            retry_clause = (
                "\n\nYour previous draft cited these numbers, none of which "
                f"appear in the input: {offenses_str}. Revise the reasoning "
                "to only cite numbers literally present in the JSON above."
            )

        prompt = (
            "Translate the optimizer output into a list of per-channel "
            "recommendations.\n\n"
            "RULES:\n"
            "- One ChannelRecommendation per channel in `channels` "
            "(do NOT recommend on `untested_channels`).\n"
            "- `reasoning` is AT MOST 2 sentences. Cite specific numbers "
            "from the input JSON (current_spend, recommended_spend, delta, "
            "marginal_roas_at_*). Do not introduce new numbers or compute "
            "new ratios.\n"
            "- `action`: 'increase' if recommended_spend > current_spend, "
            "'decrease' if recommended_spend < current_spend, 'hold' if "
            "they are within ~10% of each other.\n"
            "- `confidence`: 'high' if partial_r_squared > 0.10, 'medium' "
            "if 0.02-0.10, 'low' otherwise.\n\n"
            f"Diagnosis summary: {diagnosis_summary}\n\n"
            "Optimizer output:\n"
            + json.dumps(input_dict, indent=2)
            + retry_clause
        )
        result = structured.invoke(prompt)
        return list(result.recommendations)

    return recommend


def run_agent(input_dict: dict, model: str = DEFAULT_MODEL) -> AgentOutput:
    """Build the LLM client, run the graph end-to-end, return AgentOutput."""

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY env var not set. Get one at "
            "platform.deepseek.com and `export DEEPSEEK_API_KEY=...` "
            "before running the agent."
        )

    llm = ChatOpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        model=model,
        temperature=0.0,
    )

    graph = build_graph(_make_diagnose(llm), _make_recommend(llm))
    final = graph.invoke(
        {
            "input_dict": input_dict,
            "diagnosis_summary": "",
            "draft": [],
            "last_offenses": [],
            "retries": 0,
        }
    )

    return final["output"]
