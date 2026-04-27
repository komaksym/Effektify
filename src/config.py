"""Constants for optimizer + demo selection.

These are the realism guards from PRD §8 plus thresholds for the diagnosis
labels and the per-channel quality gate. The demo week is hand-picked in M5;
the placeholder here is just a known reasonable week from the Robyn dataset.
"""

from __future__ import annotations

# Realism guards on per-channel allocation (PRD §8).
# The 5% lower bound applies only to channels the customer is *currently*
# spending on — we don't force a $0 channel up to 5% of budget, since any
# spend on a dormant channel is itself a recommendation we can't justify
# without a pilot test.
BUDGET_FRACTION_MIN = 0.05
BUDGET_FRACTION_MAX = 0.60
HISTORICAL_SPEND_MULTIPLIER = 1.5

# Quality gate. PRD §8 specified per-channel univariate R² < 0.3 → exclude;
# we replaced univariate fits with a joint model in M1, so the gate moved
# to per-channel partial R² (incremental joint-R² this channel contributes).
PARTIAL_R2_THRESHOLD = 0.02

# Diagnosis labels. PRD §8 used absolute marginal-ROAS thresholds (<0.5 /
# >2.0); on our data those thresholds are vacuous because joint-fit ROAS
# magnitudes sit in the 10–60 range (β at upper bound for some channels,
# typical of MMM without adstock or external controls). We diagnose off
# the spend-ratio instead — direct and scale-invariant. Marginal ROAS at
# current and at recommended are still surfaced for the agent to cite.
NEAR_OPTIMUM_TOLERANCE = 0.10  # |Δs / s_current| under this → "near optimum"

# Demo week. Picked in M5 — placeholder for now.
DEMO_WEEK = "2018-06-25"
