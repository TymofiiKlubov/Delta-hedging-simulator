"""Closed-form and leading-order results the simulations are checked against.

Notation: ``sigma_h`` is the volatility used for the hedge's delta/gamma, ``sigma_r`` the
volatility of the (GBM) paths and ``mu`` their drift; ``S^2 Gamma`` is the "dollar gamma".

Two Gaussian integrals do most of the work. For ``S_t`` log-normal with drift ``mu`` and vol
``sigma_r`` and ``Gamma_h`` the Black-Scholes gamma at vol ``sigma_h`` and time to expiry
``tau = T - t``, write ``s^2 = sigma_r^2 t`` and ``b^2 = sigma_h^2 tau``. Then

    E[S_t^2 Gamma_h]     = e^{-q tau} E[S_t] phi(D1) / sqrt(s^2 + b^2)
    E[(S_t^2 Gamma_h)^2] = e^{-2 q tau} E[S_t^2] phi(D2) / (2 sqrt(pi) b sqrt(s^2 + b^2 / 2))

with ``D1 = (ln(S0/K) + mu t + s^2/2 + c) / sqrt(s^2 + b^2)``,
``D2 = (ln(S0/K) + mu t + 3 s^2/2 + c) / sqrt(s^2 + b^2/2)`` and ``c = (r - q + sigma_h^2/2) tau``.
They give, to leading order in the hedging interval ``dt = T / N``:

* discrete-hedging error  Var ~ (1/2) sigma_r^4 dt int_0^T E[(S^2 Gamma)^2] dt'
  (the per-step error is (1/2) Gamma S^2 (sigma^2 dt - R^2), R the step return); for an ATM
  option this is the Derman-Kamal rule std ~ sqrt(pi/4) * vega * sigma / sqrt(N);
* vol-gap P&L of an implied-vol hedge  E[PnL] = (1/2)(sigma_i^2 - sigma_r^2) int E[S^2 Gamma_i] dt;
* Leland-style hedging costs  E|d Delta| S ~ sqrt(2/pi) sigma_r sqrt(dt) E[S^2 Gamma] per rebalance.

All P&L figures are for the hedged position ``position`` (``-1`` = short one option), valued at T.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from . import bs

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def time_quadrature(T: float, n: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes/weights for ``int_0^T f(t) dt``.

    Substituting ``t = T - u^2`` absorbs the ``1/sqrt(T - t)`` growth of gamma near expiry,
    so the rule converges fast for every integrand used here.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    a = math.sqrt(T)
    u = 0.5 * a * (x + 1.0)
    return T - u**2, 0.5 * a * w * 2.0 * u


def expected_dollar_gamma(t, S0, K, T, sigma_h, sigma_r, r=0.0, q=0.0, mu=0.0) -> np.ndarray:
    """E[S_t^2 Gamma(T - t, S_t; sigma_h)] when S follows GBM(mu, sigma_r)."""
    t = np.asarray(t, dtype=float)
    tau = T - t
    var = sigma_r**2 * t + sigma_h**2 * tau
    d = (math.log(S0 / K) + mu * t + 0.5 * sigma_r**2 * t + (r - q + 0.5 * sigma_h**2) * tau)
    d = d / np.sqrt(var)
    return np.exp(-q * tau) * S0 * np.exp(mu * t) * bs.norm_pdf(d) / np.sqrt(var)


def expected_dollar_gamma_sq(t, S0, K, T, sigma_h, sigma_r, r=0.0, q=0.0, mu=0.0) -> np.ndarray:
    """E[(S_t^2 Gamma(T - t, S_t; sigma_h))^2] when S follows GBM(mu, sigma_r)."""
    t = np.asarray(t, dtype=float)
    tau = T - t
    s2 = sigma_r**2 * t
    b = sigma_h * np.sqrt(tau)
    var = s2 + 0.5 * b**2
    d = (math.log(S0 / K) + mu * t + 1.5 * s2 + (r - q + 0.5 * sigma_h**2) * tau) / np.sqrt(var)
    second_moment = S0**2 * np.exp((2.0 * mu + sigma_r**2) * t)
    with np.errstate(divide="ignore"):
        return (np.exp(-2.0 * q * tau) * second_moment * bs.norm_pdf(d)
                / (2.0 * math.sqrt(math.pi) * b * np.sqrt(var)))


def derman_kamal_std(n_rebalances, S0, K, T, sigma, r=0.0, q=0.0, position=-1.0) -> np.ndarray:
    """Derman & Kamal: std of the discrete-hedging P&L ~ sqrt(pi/4) * vega * sigma / sqrt(N).

    Derived for an at-the-money option hedged N times at its true volatility.
    """
    vega = float(bs.vega(S0, K, T, sigma, r, q))
    return abs(position) * math.sqrt(math.pi / 4.0) * vega * sigma / np.sqrt(n_rebalances)


def discretisation_std(n_rebalances, S0, K, T, sigma_r, sigma_h=None, r=0.0, q=0.0, mu=0.0,
                       position=-1.0) -> np.ndarray:
    """Leading-order std of the hedging error from rebalancing N times (any strike):
    sqrt((1/2) sigma_r^4 (T/N) int_0^T e^{2r(T-t)} E[(S_t^2 Gamma_t)^2] dt)."""
    sigma_h = sigma_r if sigma_h is None else sigma_h
    t, w = time_quadrature(T)
    integral = np.sum(w * np.exp(2 * r * (T - t))
                      * expected_dollar_gamma_sq(t, S0, K, T, sigma_h, sigma_r, r, q, mu))
    n = np.asarray(n_rebalances, dtype=float)
    return abs(position) * np.sqrt(0.5 * sigma_r**4 * (T / n) * integral)


def vol_gap_mean(S0, K, T, sigma_i, sigma_r, r=0.0, q=0.0, mu=0.0, kind="call", position=-1.0,
                 hedge="implied") -> float:
    """Expected P&L (continuous hedging) of an option sold at sigma_i on GBM(mu, sigma_r) paths.

    hedge="realised": delta at sigma_r replicates the option, locking in
        -position * (BS(sigma_i) - BS(sigma_r)) e^{rT} on every path, whatever mu is.
    hedge="implied": delta at sigma_i earns -position * (1/2) int e^{r(T-t)} Gamma_i S^2
        (sigma_i^2 - sigma_r^2) dt, path by path; its mean equals the realised-vol value
        when mu = r - q.
    """
    if hedge == "realised":
        gap = bs.price(S0, K, T, sigma_i, r, q, kind) - bs.price(S0, K, T, sigma_r, r, q, kind)
        return float(-position * gap * math.exp(r * T))
    if hedge != "implied":
        raise ValueError("hedge must be 'implied' or 'realised'")
    t, w = time_quadrature(T)
    integral = np.sum(w * np.exp(r * (T - t))
                      * expected_dollar_gamma(t, S0, K, T, sigma_i, sigma_r, r, q, mu))
    return float(-position * 0.5 * (sigma_i**2 - sigma_r**2) * integral)


def expected_hedge_cost(n_rebalances, S0, K, T, sigma_r, sigma_h=None, bps=0.0, fixed=0.0,
                        r=0.0, q=0.0, mu=0.0, kind="call", position=-1.0) -> np.ndarray:
    """Leading-order expected transaction costs of hedging N times (Leland 1985 style).

    initial hedge  k |Delta_0| S_0
    rebalances     k sqrt(2/pi) sigma_r sqrt(dt) sum_{j=0}^{N-2} E[S^2 Gamma](t_j)
    unwind         k E[S_T; option in the money]    (the final hedge is ~0 or ~1 unit)
    fixed fees     fixed * (N + 1) trades
    with k = bps / 10^4, all per unit |position|.
    """
    sigma_h = sigma_r if sigma_h is None else sigma_h
    k = bps * 1e-4
    ns = np.atleast_1d(np.asarray(n_rebalances, dtype=int))
    d0 = abs(float(bs.delta(S0, K, T, sigma_h, r, q, kind)))
    d_itm = (math.log(S0 / K) + (mu + 0.5 * sigma_r**2) * T) / (sigma_r * math.sqrt(T))
    p_itm = float(ndtr(d_itm if bs.is_call(kind) else -d_itm))
    unwind = S0 * math.exp(mu * T) * p_itm
    out = np.empty(ns.shape, dtype=float)
    for i, n in enumerate(ns):
        dt = T / n
        tj = dt * np.arange(n - 1)
        dgam = expected_dollar_gamma(tj, S0, K, T, sigma_h, sigma_r, r, q, mu)
        rebalance = _SQRT_2_OVER_PI * sigma_r * math.sqrt(dt) * float(np.sum(dgam))
        out[i] = abs(position) * (k * (S0 * d0 + rebalance + unwind)) + fixed * (n + 1)
    return out if np.ndim(n_rebalances) else out[0]


@dataclass(frozen=True)
class CostTradeoff:
    n_rebalances: np.ndarray
    mean: np.ndarray  # expected P&L (vol-gap mean minus expected costs)
    std: np.ndarray
    cost: np.ndarray
    objective: np.ndarray  # J = -mean + kappa * var / premium

    @property
    def best(self) -> int:
        return int(self.n_rebalances[np.argmin(self.objective)])


def cost_tradeoff(n_rebalances, S0, K, T, sigma_i, sigma_r, bps, fixed=0.0, risk_aversion=1.0,
                  r=0.0, q=0.0, mu=0.0, kind="call", position=-1.0) -> CostTradeoff:
    """Theoretical mean-variance objective J(N) = -E[PnL] + kappa Var[PnL] / premium."""
    ns = np.asarray(n_rebalances, dtype=int)
    premium = abs(position) * float(bs.price(S0, K, T, sigma_i, r, q, kind))
    cost = expected_hedge_cost(ns, S0, K, T, sigma_r, sigma_i, bps, fixed, r, q, mu, kind, position)
    gap = vol_gap_mean(S0, K, T, sigma_i, sigma_r, r, q, mu, kind, position, "implied")
    std = discretisation_std(ns, S0, K, T, sigma_r, sigma_i, r, q, mu, position)
    mean = gap - cost
    return CostTradeoff(ns, mean, std, cost, -mean + risk_aversion * std**2 / premium)


def optimal_rebalances(S0, K, T, sigma_i, sigma_r, bps, fixed=0.0, risk_aversion=1.0, r=0.0,
                       q=0.0, mu=0.0, kind="call", position=-1.0, n_max=100_000) -> CostTradeoff:
    """Minimise the theoretical objective over N: coarse log grid, then every integer nearby."""
    grid = np.unique(np.geomspace(1, n_max, 300).astype(int))
    args = (S0, K, T, sigma_i, sigma_r, bps, fixed, risk_aversion, r, q, mu, kind, position)
    best = cost_tradeoff(grid, *args).best
    lo, hi = max(1, int(best / 1.1) - 1), min(n_max, int(best * 1.1) + 1)
    return cost_tradeoff(np.arange(lo, hi + 1), *args)
