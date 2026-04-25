"""Saturation curve formula and evaluation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def saturation(
    spend: ArrayLike,
    alpha: float,
    beta: float,
) -> NDArray[np.float64]:
    """Diminishing-returns curve: revenue = α · log(1 + spend / β).

    Vectorised over `spend`. Used both for fitting (curve_fit.py) and for
    evaluating the fitted curve at arbitrary spend (plots, optimizer).
    """

    return alpha * np.log1p(np.asarray(spend, dtype=float) / beta)
