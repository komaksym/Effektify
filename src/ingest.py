"""Load the Robyn MMM benchmark CSV into a long-format DataFrame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "data" / "robyn.csv"

# Robyn's wide format includes impressions (_I), clicks (_P), and competitor
# sales (_B); we keep only the dollar-denominated spend columns (_S) since
# the saturation curve is defined on spend.
SPEND_COLS: dict[str, str] = {
    "tv_S": "tv",
    "ooh_S": "ooh",
    "print_S": "print",
    "search_S": "search",
    "facebook_S": "facebook",
}


def load_robyn(csv_path: Path = DEFAULT_CSV) -> pd.DataFrame:
    """Return a long-format DataFrame with columns: week, channel, spend, revenue.

    Total weekly revenue is replicated across each channel's row for that week
    — the modeling layer fits a saturation curve per channel against this
    shared y. This is the simplification the PRD calls out (no adstock, no
    cross-channel interactions).
    """

    raw = pd.read_csv(csv_path, parse_dates=["DATE"])

    long = raw.melt(
        id_vars=["DATE", "revenue"],
        value_vars=list(SPEND_COLS.keys()),
        var_name="channel",
        value_name="spend",
    )
    long["channel"] = long["channel"].map(SPEND_COLS)
    long = long.rename(columns={"DATE": "week"})

    return (
        long[["week", "channel", "spend", "revenue"]]
        .sort_values(["week", "channel"])
        .reset_index(drop=True)
    )


def main() -> None:
    df = load_robyn()

    n_weeks = df["week"].nunique()
    n_channels = df["channel"].nunique()
    total_spend = float(df["spend"].sum())
    total_revenue = float(df.drop_duplicates("week")["revenue"].sum())

    print(
        f"weeks={n_weeks}  channels={n_channels}  "
        f"total_spend=${total_spend:,.0f}  "
        f"total_revenue=${total_revenue:,.0f}"
    )


if __name__ == "__main__":
    main()
