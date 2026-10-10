"""Summary statistics of P&L samples, with standard errors."""

from __future__ import annotations

import numpy as np


def summarize(x: np.ndarray) -> dict[str, float]:
    """Mean, std (with standard errors), higher moments, quantiles and expected shortfall.

    ``se_std`` uses the large-sample formula ``std * sqrt((kurtosis - 1) / (4 n))``, which
    stays valid for the fat-tailed P&L of coarse or jump-exposed hedges.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if n > 1 else 0.0
    z = (x - mean) / std if std > 0 else np.zeros_like(x)
    kurt = float(np.mean(z**4)) if std > 0 else 3.0
    q = np.quantile(x, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    srt = np.sort(x)
    es = {f"es{int(a * 100):02d}": float(srt[: max(1, int(a * n))].mean()) for a in (0.01, 0.05)}
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "se_mean": std / np.sqrt(n),
        "se_std": std * np.sqrt(max(kurt - 1.0, 0.0) / (4 * n)),
        "skew": float(np.mean(z**3)) if std > 0 else 0.0,
        "excess_kurtosis": kurt - 3.0,
        "q01": float(q[0]), "q05": float(q[1]), "q25": float(q[2]), "median": float(q[3]),
        "q75": float(q[4]), "q95": float(q[5]), "q99": float(q[6]),
        **es,
        "min": float(srt[0]),
        "max": float(srt[-1]),
    }
