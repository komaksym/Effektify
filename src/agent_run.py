"""CLI: run the optimizer + agent end-to-end on the demo week, write JSON.

Usage::

    export DEEPSEEK_API_KEY=...
    uv run python -m src.agent_run

Outputs ``cache/agent_output.json``. Kept separate from ``src.precompute`` so
the deterministic math (free, fast, reproducible) is decoupled from the LLM
step (metered, slower, non-deterministic).
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from src.agent_graph import run_agent
from src.config import DEMO_WEEK
from src.ingest import load_robyn
from src.optimizer import load_cache, optimize_allocation, to_dict

ROOT = Path(__file__).resolve().parent.parent
AGENT_OUTPUT_PATH = ROOT / "cache" / "agent_output.json"


def main() -> None:
    load_dotenv()

    cache = load_cache()
    df = load_robyn()

    week_row = df[df["week"] == DEMO_WEEK]
    if week_row.empty:
        raise SystemExit(f"DEMO_WEEK {DEMO_WEEK} not in dataset")

    actual = dict(zip(week_row["channel"], week_row["spend"], strict=True))
    optimizer_out = optimize_allocation(cache, actual)
    optimizer_dict = to_dict(optimizer_out)

    agent_out = run_agent(optimizer_dict)

    AGENT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AGENT_OUTPUT_PATH.write_text(json.dumps(agent_out.model_dump(), indent=2))

    print(f"wrote {AGENT_OUTPUT_PATH.relative_to(ROOT)}")
    print(f"\nheadline: {agent_out.headline}\n")
    for rec in agent_out.recommendations:
        print(
            f"  {rec.channel:>10}  "
            f"action={rec.action:<8}  "
            f"confidence={rec.confidence}\n"
            f"             {rec.reasoning}"
        )
    if agent_out.untested_channels:
        print(f"\nuntested: {', '.join(agent_out.untested_channels)}")
    if agent_out.warnings:
        print("\nwarnings:")
        for w in agent_out.warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
