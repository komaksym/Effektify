"""Regex-based grounding check.

Every numeric value the agent quotes in its prose must be traceable to a
literal value in the upstream optimizer dict, within 1% relative tolerance.
This is the demo's technical centerpiece per PRD §13: it makes the
``"the LLM never invents numbers"`` claim mechanically enforceable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Matches integers and decimals with optional leading sign and $ in either
# order — covers `-$71,441`, `$-71,441`, `$71,441`, `-71441`, `71441.5`.
NUMBER_RE = re.compile(
    r"-?\$?-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\$?-?\d+(?:\.\d+)?"
)

DEFAULT_TOLERANCE = 0.01


@dataclass(frozen=True, slots=True)
class Ungrounded:
    """A number cited in agent prose that doesn't match any input value."""

    value: float
    nearest: float
    relative_diff: float


class GroundingFailure(Exception):
    """Raised when the agent prose contains numbers not present in input."""

    def __init__(self, ungrounded: list[Ungrounded]) -> None:
        self.ungrounded = ungrounded
        details = ", ".join(
            f"{u.value:g} (nearest={u.nearest:g}, off={u.relative_diff:.2%})"
            for u in ungrounded
        )
        super().__init__(f"ungrounded numbers in agent prose: {details}")


def extract_numbers(text: str) -> list[float]:
    """Pull every number out of agent prose. Strips $ and , then parses float."""

    out: list[float] = []
    for match in NUMBER_RE.finditer(text):
        token = match.group(0).replace("$", "").replace(",", "")
        if not token or token in {"-", "."}:
            continue
        try:
            out.append(float(token))
        except ValueError:
            continue

    return out


def collect_grounded_values(structured: object) -> set[float]:
    """Walk the optimizer dict recursively, collect every numeric leaf."""

    found: set[float] = set()

    def walk(node: object) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, int | float):
            found.add(float(node))
            return
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
            return
        if isinstance(node, list | tuple):
            for v in node:
                walk(v)

    walk(structured)

    return found


def _closest(value: float, candidates: set[float]) -> tuple[float, float]:
    """Find the candidate with the smallest relative distance to `value`."""

    best = min(candidates, key=lambda c: abs(c - value))
    denom = max(abs(value), 1.0)
    rel_diff = abs(best - value) / denom

    return best, rel_diff


def check_grounded(
    text: str,
    structured: object,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[Ungrounded]:
    """Return the list of numbers in `text` that don't appear in `structured`.

    Empty list = clean. Non-empty = the agent invented those numbers (or used
    transformations the input doesn't justify).
    """

    candidates = collect_grounded_values(structured)
    if not candidates:
        return [
            Ungrounded(value=v, nearest=0.0, relative_diff=float("inf"))
            for v in extract_numbers(text)
        ]

    out: list[Ungrounded] = []
    for value in extract_numbers(text):
        nearest, rel_diff = _closest(value, candidates)
        if rel_diff > tolerance:
            out.append(
                Ungrounded(value=value, nearest=nearest, relative_diff=rel_diff)
            )

    return out
