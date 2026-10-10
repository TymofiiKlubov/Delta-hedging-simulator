"""The hedging engine: sell an option, delta-hedge it along simulated paths, attribute the P&L.

Timeline on the grid ``0 = t_0 < t_1 < ... < t_n = T`` (P&L is valued at T):

* ``t_0``: receive the premium ``-position * BS(S_0; sigma_i)`` and buy the initial hedge.
* rebalance dates: trade to the new hedge, paying transaction costs.
* ``(t_k, t_{k+1}]``: cash accrues at ``r``; the hedge earns the yield ``q`` (reinvested
  continuously, credited to cash at ``t_{k+1}``).
* ``t_n = T``: settle the option payoff in cash and unwind the hedge at ``S_T``.

The hedge is ``h_k = -position * delta(t_k, S_k; sigma_hedge)`` units of the underlying, so a
short call (``position = -1``) holds ``+delta``. Summation by parts gives the exact attribution

    PnL = premium + payoff + sum_k h_k (S_{k+1} - S_k) + financing + costs

where *financing* is interest on the cash account plus the yield on the hedge and *costs*
is minus the transaction costs paid. Marking the option to model at ``sigma_i``, each
step's P&L splits further into

    theta_k    = position * Theta_k * dt_k               time decay (> 0 for a short)
    gamma_k    = position * Gamma_k * dS_k^2 / 2         convexity  (< 0 for a short)
    mismatch_k = (h_k + position * Delta_k) * dS_k       hedge differs from the sigma_i delta
    financing_k, costs_k and a residual: the third- and higher-order Taylor terms.

With ``r = q = 0`` the BS PDE gives ``theta_k + gamma_k = position * Gamma_k S_k^2
(R_k^2 - sigma_i^2 dt_k) / 2`` with ``R_k = dS_k / S_k``: a short collects
``Gamma S^2 sigma_i^2 dt / 2`` and pays ``Gamma S^2 R_k^2 / 2`` on every step.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from . import bs


@dataclass(frozen=True)
class Option:
    strike: float
    expiry: float  # years
    kind: str = "call"
    position: float = -1.0  # options held; -1 = short one option

    def payoff(self, S) -> np.ndarray:
        return bs.payoff(S, self.strike, self.kind)


@dataclass(frozen=True)
class Market:
    spot: float
    r: float = 0.0  # interest on the cash account
    q: float = 0.0  # dividend / funding yield earned by holders of the underlying


@dataclass(frozen=True)
class Costs:
    bps: float = 0.0  # proportional cost in basis points of traded notional |trade| * S
    fixed: float = 0.0  # fixed cost per (non-zero) trade

    @property
    def is_zero(self) -> bool:
        return self.bps == 0.0 and self.fixed == 0.0

    def __call__(self, trades: np.ndarray, S: np.ndarray) -> np.ndarray:
        fee = (self.bps * 1e-4) * np.abs(trades) * S
        if self.fixed:
            fee += self.fixed * (trades != 0.0)
        return fee


@dataclass(frozen=True)
class Strategy:
    """How the hedge is run.

    hedge_vol: volatility used for the hedge delta (``None`` -> the implied vol ``sigma_i``).
    every:     rebalance every ``every`` grid steps (with a band: check the band that often).
    band:      if set, trade only when ``|target - holding| > band`` (delta units per option).
    band_mode: ``"target"`` trades back to the target delta, ``"edge"`` only to the band edge.
    """

    hedge_vol: float | None = None
    every: int = 1
    band: float | None = None
    band_mode: str = "target"


@dataclass
class HedgeResult:
    """Per-path P&L and its attribution. All amounts are valued at expiry T."""

    pnl: np.ndarray
    premium: np.ndarray
    payoff: np.ndarray
    hedge: np.ndarray  # sum_k h_k (S_{k+1} - S_k)
    interest: np.ndarray  # interest earned on the cash account
    carry: np.ndarray  # yield q earned on the hedge
    costs: np.ndarray  # minus transaction costs paid
    n_trades: np.ndarray
    turnover: np.ndarray  # traded notional, sum |trade| * S
    theta: np.ndarray | None = None
    gamma: np.ndarray | None = None
    mismatch: np.ndarray | None = None
    residual: np.ndarray | None = None
    steps: dict[str, np.ndarray] | None = None  # per-step detail (record=True)

    @property
    def financing(self) -> np.ndarray:
        return self.interest + self.carry

    def attribution(self) -> dict[str, np.ndarray]:
        return {"premium": self.premium, "payoff": self.payoff, "hedge": self.hedge,
                "financing": self.financing, "costs": self.costs}

    def decomposition(self) -> dict[str, np.ndarray]:
        if self.theta is None:
            raise ValueError("run_hedge(..., decompose=True) is needed for the decomposition")
        return {"theta": self.theta, "gamma": self.gamma, "mismatch": self.mismatch,
                "financing": self.financing, "costs": self.costs, "residual": self.residual}

    @classmethod
    def concat(cls, results: list[HedgeResult]) -> HedgeResult:
        """Stack results computed on separate chunks of paths."""
        out = {}
        for f in fields(cls):
            vals = [getattr(r, f.name) for r in results]
            if f.name == "steps":
                out[f.name] = None
            elif any(v is None for v in vals):
                out[f.name] = None
            else:
                out[f.name] = np.concatenate(vals)
        return cls(**out)


def run_hedge(S, t, option: Option, market: Market, sigma_i: float,
              strategy: Strategy = Strategy(), costs: Costs = Costs(), *,
              target_delta: np.ndarray | None = None, decompose: bool = False,
              record: bool = False) -> HedgeResult:
    """Sell (or buy) ``option`` at ``BS(sigma_i)`` and delta-hedge it along each path.

    Parameters
    ----------
    S : array (n_paths, n + 1)
        Spot paths on the grid ``t`` (a single path may be passed as a 1-D array).
    t : array (n + 1,)
        Time grid from 0 to ``option.expiry``.
    sigma_i : float
        Implied vol: sets the sale price and the mark-to-model in the decomposition.
    target_delta : array (n_paths, n), optional
        Precomputed per-option delta at the hedge vol on the grid points ``t_0 .. t_{n-1}``
        (lets several strategies share one delta computation).
    decompose : bool
        Also return the per-path theta / gamma / mismatch / residual split.
    record : bool
        Also return per-step arrays in ``result.steps`` (use with few paths).
    """
    S = np.asarray(S, dtype=float)
    if S.ndim == 1:
        S = S[None, :]
    t = np.asarray(t, dtype=float)
    _validate(S, t, option, sigma_i, strategy, target_delta)
    n_paths, n = S.shape[0], S.shape[1] - 1
    K, T, pos, kind = option.strike, option.expiry, option.position, option.kind
    r, q = market.r, market.q
    hedge_vol = sigma_i if strategy.hedge_vol is None else strategy.hedge_vol

    tau = T - t[:-1]
    dt = np.diff(t)
    h = _holdings(S, tau, option, market, hedge_vol, strategy, target_delta)

    trades = np.empty((n_paths, n + 1))
    trades[:, 0] = h[:, 0]
    np.subtract(h[:, 1:], h[:, :-1], out=trades[:, 1:n])
    trades[:, n] = -h[:, -1]
    fees = np.zeros_like(trades) if costs.is_zero else costs(trades, S)

    dS = np.diff(S, axis=1)
    carry = h * S[:, 1:] * np.expm1(q * dt) if q else np.zeros_like(h)
    premium = -pos * bs.price(S[:, 0], K, T, sigma_i, r, q, kind)
    payoff = pos * option.payoff(S[:, -1])

    # Cash flows at each grid date (excluding interest); the P&L is the cash account at T.
    flows = -trades * S
    flows -= fees
    flows[:, 1:] += carry
    flows[:, 0] += premium
    flows[:, -1] += payoff
    pnl = flows @ np.exp(r * (T - t))

    abs_trades = np.abs(trades)
    result = HedgeResult(
        pnl=pnl,
        premium=premium,
        payoff=payoff,
        hedge=np.einsum("ij,ij->i", h, dS),
        interest=flows @ np.expm1(r * (T - t)),
        carry=carry.sum(axis=1),
        costs=-fees.sum(axis=1),
        n_trades=np.count_nonzero(trades, axis=1),
        turnover=np.einsum("ij,ij->i", abs_trades, S),
    )
    if decompose or record:
        _decompose(result, S, t, tau, dt, dS, h, trades, fees, carry, flows, option, market,
                   sigma_i, record)
    return result


def _validate(S, t, option, sigma_i, strategy, target_delta) -> None:
    n = S.shape[1] - 1
    if n < 1:
        raise ValueError("need at least one time step")
    if option.expiry <= 0:
        raise ValueError("option expiry must be positive")
    if t.shape != (n + 1,):
        raise ValueError(f"time grid has shape {t.shape}, expected ({n + 1},)")
    if abs(t[0]) > 1e-12 or abs(t[-1] - option.expiry) > 1e-9 * max(option.expiry, 1.0):
        raise ValueError("time grid must run from 0 to the option expiry")
    if np.any(np.diff(t) <= 0):
        raise ValueError("time grid must be strictly increasing")
    if sigma_i < 0 or (strategy.hedge_vol is not None and strategy.hedge_vol < 0):
        raise ValueError("volatilities must be non-negative")
    if int(strategy.every) != strategy.every or strategy.every < 1:
        raise ValueError("strategy.every must be a positive integer")
    if strategy.band is not None and strategy.band < 0:
        raise ValueError("band must be non-negative")
    if strategy.band_mode not in ("target", "edge"):
        raise ValueError("band_mode must be 'target' or 'edge'")
    if target_delta is not None and target_delta.shape != (S.shape[0], n):
        raise ValueError(f"target_delta must have shape ({S.shape[0]}, {n})")


def _holdings(S, tau, option, market, hedge_vol, strategy, target_delta) -> np.ndarray:
    """Units of the underlying held over each interval (t_k, t_{k+1}], shape (n_paths, n)."""
    n = S.shape[1] - 1
    every = int(strategy.every)
    cols = np.arange(0, n, every)  # rebalance (or band-check) dates
    if target_delta is not None:
        delta = target_delta[:, cols]
    else:
        delta = bs.delta(S[:, cols], option.strike, tau[cols], hedge_vol, market.r, market.q,
                         option.kind)
    target = -option.position * delta
    if strategy.band is None:
        held = target
    else:
        held = _band_path(target, strategy.band * abs(option.position), strategy.band_mode)
    return held if every == 1 else np.repeat(held, every, axis=1)[:, :n]


def _band_path(target: np.ndarray, band: float, mode: str) -> np.ndarray:
    """Hold the hedge until the target moves more than ``band`` away from it."""
    tgt = np.ascontiguousarray(target.T)  # time-major for contiguous per-date access
    held = np.empty_like(tgt)
    cur = tgt[0].copy()
    held[0] = cur
    for k in range(1, tgt.shape[0]):
        gap = tgt[k] - cur
        hit = np.abs(gap) > band
        if mode == "target":
            cur = np.where(hit, tgt[k], cur)
        else:
            cur = np.where(hit, tgt[k] - np.copysign(band, gap), cur)
        held[k] = cur
    return held.T


def _decompose(res, S, t, tau, dt, dS, h, trades, fees, carry, flows, option, market,
               sigma_i, record) -> None:
    """Theta / gamma / mismatch / residual split of the mark-to-model P&L at sigma_i."""
    pos, r = option.position, market.r
    g = bs.greeks(S[:, :-1], option.strike, tau, sigma_i, r, market.q, option.kind)
    growth = np.exp(r * t)
    cash = np.cumsum(flows / growth, axis=1) * growth  # cash after the flows at each date
    interest = cash[:, :-1] * np.expm1(r * dt)
    value = np.empty_like(S)  # option value per unit position, payoff at T
    value[:, :-1] = g.price
    value[:, -1] = option.payoff(S[:, -1])
    mtm = np.empty_like(S)  # mark-to-model P&L of the hedged position
    mtm[:, :-1] = cash[:, :-1] + h * S[:, :-1] + pos * g.price
    mtm[:, -1] = cash[:, -1]

    theta = pos * g.theta * dt
    gamma = 0.5 * pos * g.gamma * dS**2
    mismatch = (h + pos * g.delta) * dS
    financing = interest + carry
    step_costs = -fees[:, 1:]
    residual = np.diff(mtm, axis=1) - theta - gamma - mismatch - financing - step_costs

    res.theta, res.gamma = theta.sum(axis=1), gamma.sum(axis=1)
    res.mismatch, res.residual = mismatch.sum(axis=1), residual.sum(axis=1)
    if record:
        res.steps = {
            "t": t, "S": S, "holding": h, "trades": trades, "fees": fees,
            "delta": g.delta, "gamma_greek": g.gamma, "theta_greek": g.theta,
            "value": value, "cash": cash, "mtm_pnl": mtm,
            "theta": theta, "gamma": gamma, "mismatch": mismatch,
            "financing": financing, "costs": step_costs, "residual": residual,
        }
