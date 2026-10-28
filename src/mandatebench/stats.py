"""Small, dependency-free statistics used in the paper."""

from __future__ import annotations

from math import comb, sqrt

#: 97.5th percentile of the standard normal, for two-sided 95% intervals.
Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (Wilson 1927).

    Used instead of the normal approximation because several cells have
    proportions at or near 0 or 1, where the normal interval leaves [0, 1].
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # At k = 0 and k = n the bounds are exactly 0 and 1; set them directly
    # rather than leave floating-point residue such as 1.4e-17.
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return (lo, hi)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test (McNemar 1947) on the discordant pairs.

    b: pairs positive only under condition A; c: positive only under B.
    Under H0 each discordant pair is equally likely to fall either way, so the
    smaller count is Binomial(b + c, 1/2). Returns 1.0 when there are no
    discordant pairs.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm (1979) step-down adjustment; returns adjusted p-values by key."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
