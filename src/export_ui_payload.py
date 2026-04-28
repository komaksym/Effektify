"""Build the static frontend payload for the demo week.

Reads the deterministic optimizer inputs from cache plus the grounded
agent prose, then exports:

- ``cache/optimizer_output.json`` for inspection/debugging.
- ``frontend/data/brief.json`` for the static UI.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime, timezone
from html import escape
from html.parser import HTMLParser
from pathlib import Path

from src.agent_schemas import AgentOutput
from src.config import DEMO_WEEK
from src.ingest import load_robyn
from src.optimizer import (
    CACHE_PATH,
    load_cache,
    optimize_allocation,
    to_dict,
)

ROOT = Path(__file__).resolve().parent.parent
AGENT_OUTPUT_PATH = ROOT / "cache" / "agent_output.json"
OPTIMIZER_OUTPUT_PATH = ROOT / "cache" / "optimizer_output.json"
FRONTEND_BRIEF_PATH = ROOT / "frontend" / "data" / "brief.json"


class _BoldOnlyHTMLParser(HTMLParser):
    """Allow plain text plus bare ``<b>`` tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        if tag == "b":
            self.parts.append("<b>")

    def handle_endtag(self, tag: str):
        if tag == "b":
            self.parts.append("</b>")

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag == "b":
            self.parts.append("<b></b>")

    def handle_data(self, data: str) -> None:
        self.parts.append(escape(data))


def sanitize_rich_text(text: str) -> str:
    """Strip all HTML except ``<b>`` while keeping the text content."""

    parser = _BoldOnlyHTMLParser()
    parser.feed(text)
    parser.close()
    return "".join(parser.parts)


def _ensure_unique(names: list[str], label: str) -> None:
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        joined = ", ".join(duplicates)
        raise ValueError(f"duplicate channel names in {label}: {joined}")


def _signal_strength(confidence: str) -> str:
    return {
        "high": "Strong signal",
        "medium": "Moderate signal",
        "low": "Weak signal",
    }[confidence]


def _format_week_label(week: str) -> str:
    parsed = date.fromisoformat(week)
    return f"Week of {parsed.strftime('%b %d, %Y').replace(' 0', ' ')}"


def _generated_at_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _build_demo_meta() -> dict[str, int | str]:
    df = load_robyn()
    weeks = sorted({str(week.date()) for week in df["week"].drop_duplicates()})
    if DEMO_WEEK not in weeks:
        raise ValueError(f"DEMO_WEEK {DEMO_WEEK} not in dataset")

    return {
        "week_iso": DEMO_WEEK,
        "brief_number": weeks.index(DEMO_WEEK) + 1,
        "history_weeks": len(weeks),
    }


def load_agent_output(path: Path = AGENT_OUTPUT_PATH) -> AgentOutput:
    if not path.exists():
        raise FileNotFoundError(
            f"missing agent output cache: {path}. "
            "Run `uv run python -m src.agent_run` or provide "
            "`cache/agent_output.json` before exporting the UI payload."
        )

    return AgentOutput.model_validate_json(path.read_text())


def _build_actual_allocation() -> dict[str, float]:
    df = load_robyn()
    week_row = df[df["week"] == DEMO_WEEK]
    if week_row.empty:
        raise ValueError(f"DEMO_WEEK {DEMO_WEEK} not in dataset")

    return {
        str(channel): float(spend)
        for channel, spend in zip(
            week_row["channel"], week_row["spend"], strict=True
        )
    }


def _join_channels(
    optimizer_dict: dict,
    cache: dict,
    agent_output: AgentOutput,
) -> tuple[list[dict], list[dict]]:
    optimizer_channels = list(optimizer_dict["channels"])
    optimizer_names = [str(channel["name"]) for channel in optimizer_channels]
    agent_names = [rec.channel for rec in agent_output.recommendations]

    _ensure_unique(optimizer_names, "optimizer output")
    _ensure_unique(agent_names, "agent output")

    missing = sorted(set(optimizer_names) - set(agent_names))
    unexpected = sorted(set(agent_names) - set(optimizer_names))
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing recommendations for: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected recommendations for: {', '.join(unexpected)}")
        raise ValueError("channel name mismatch between optimizer and agent: " + "; ".join(details))

    agent_by_name = {
        rec.channel: rec for rec in agent_output.recommendations
    }
    low_confidence = set(optimizer_dict["low_confidence_channels"])

    enriched: list[dict] = []
    for channel in optimizer_channels:
        name = str(channel["name"])
        rec = agent_by_name[name]
        curve = cache["channel_curves"][name]

        enriched.append(
            {
                "channel": name,
                "action": rec.action,
                "current_spend": float(channel["current_spend"]),
                "recommended_spend": float(channel["recommended_spend"]),
                "current_revenue": float(
                    channel["current_revenue_contribution"]
                ),
                "recommended_revenue": float(
                    channel["recommended_revenue_contribution"]
                ),
                "delta_revenue": float(channel["delta"]),
                "diagnosis_label": str(channel["diagnosis"]),
                "confidence": rec.confidence,
                "signal_strength": _signal_strength(rec.confidence),
                "grounded_reasoning": sanitize_rich_text(rec.reasoning),
                "alpha": float(curve["alpha"]),
                "beta": float(curve["beta"]),
                "alpha_samples": [float(x) for x in curve["alpha_samples"]],
                "beta_samples": [float(x) for x in curve["beta_samples"]],
                "historical_max": float(curve["historical_spend_max"]),
                "marginal_roas_at_current": float(
                    channel["marginal_roas_at_current"]
                ),
                "marginal_roas_at_recommended": float(
                    channel["marginal_roas_at_recommended"]
                ),
                "partial_r2": float(channel["partial_r_squared"]),
                "in_observed_range": bool(channel["in_observed_range"]),
            }
        )

    ledger_channels = sorted(
        (row for row in enriched if row["channel"] not in low_confidence),
        key=lambda row: (-abs(row["delta_revenue"]), row["channel"]),
    )
    low_confidence_channels = sorted(
        (row for row in enriched if row["channel"] in low_confidence),
        key=lambda row: row["channel"],
    )

    return ledger_channels, low_confidence_channels


def build_brief_payload(
    cache: dict,
    optimizer_dict: dict,
    agent_output: AgentOutput,
    actual_allocation: dict[str, float],
) -> dict:
    demo_meta = _build_demo_meta()
    ledger_channels, low_confidence_channels = _join_channels(
        optimizer_dict, cache, agent_output
    )

    budget = float(sum(actual_allocation.values()))
    delta_low, delta_high = optimizer_dict["total_delta_ci"]

    return {
        "meta": {
            "week_label": _format_week_label(DEMO_WEEK),
            "week_iso": str(demo_meta["week_iso"]),
            "brief_number": int(demo_meta["brief_number"]),
            "history_weeks": int(demo_meta["history_weeks"]),
            "generated_at": _generated_at_iso(),
            "bootstrap_count": int(cache["meta"]["n_resamples"]),
            "joint_r_squared": float(cache["joint"]["r_squared"]),
            "warnings": [
                sanitize_rich_text(warning)
                for warning in agent_output.warnings
            ],
        },
        "hero": {
            "budget": budget,
            "delta_point": float(optimizer_dict["total_delta"]),
            "delta_low": float(delta_low),
            "delta_high": float(delta_high),
            "total_current_revenue": float(
                optimizer_dict["total_current_revenue"]
            ),
            "total_recommended_revenue": float(
                optimizer_dict["total_recommended_revenue"]
            ),
            "base": float(cache["joint"]["base"]),
            "headline": sanitize_rich_text(agent_output.headline),
        },
        "channels": ledger_channels,
        "asides": {
            "untested_channels": sorted(optimizer_dict["untested_channels"]),
            "low_confidence_channels": low_confidence_channels,
        },
    }


def export_payloads(
    cache_path: Path = CACHE_PATH,
    agent_output_path: Path = AGENT_OUTPUT_PATH,
    optimizer_output_path: Path = OPTIMIZER_OUTPUT_PATH,
    frontend_brief_path: Path = FRONTEND_BRIEF_PATH,
) -> dict:
    cache = load_cache(cache_path)
    actual_allocation = _build_actual_allocation()
    optimizer_output = optimize_allocation(cache, actual_allocation)
    optimizer_dict = to_dict(optimizer_output)
    agent_output = load_agent_output(agent_output_path)
    brief = build_brief_payload(
        cache=cache,
        optimizer_dict=optimizer_dict,
        agent_output=agent_output,
        actual_allocation=actual_allocation,
    )

    optimizer_output_path.parent.mkdir(parents=True, exist_ok=True)
    optimizer_output_path.write_text(json.dumps(optimizer_dict, indent=2) + "\n")

    frontend_brief_path.parent.mkdir(parents=True, exist_ok=True)
    frontend_brief_path.write_text(json.dumps(brief, indent=2) + "\n")

    return brief


def main() -> None:
    brief = export_payloads()
    print(f"wrote {OPTIMIZER_OUTPUT_PATH.relative_to(ROOT)}")
    print(f"wrote {FRONTEND_BRIEF_PATH.relative_to(ROOT)}")
    print(
        f"{brief['meta']['week_label']}: "
        f"delta ${brief['hero']['delta_point']:,.0f} "
        f"on ${brief['hero']['budget']:,.0f} budget"
    )


if __name__ == "__main__":
    main()
