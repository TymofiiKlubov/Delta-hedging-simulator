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

A third, two-date integral E[S_s^2 Gamma_s S_t^2 Gamma_t] gives Var(int S^2 Gamma dt): the
path-to-path spread of the vol-gap P&L of an implied-vol hedge and of the hedging costs.

All P&L figures are for the hedged position ``position`` (``-1`` = short one option), valued at T.
"""

from __future__ import annotations

import functools
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


def expected_dollar_gamma_cross(s, t, S0, K, T, sigma_h, sigma_r, r=0.0, q=0.0,
                                mu=0.0) -> np.ndarray:
    """E[S_s^2 Gamma_s * S_t^2 Gamma_t] for two dates on the same GBM(mu, sigma_r) path.

    With Y_u = ln(S_u/K) + c_u (jointly Gaussian, Cov = sigma_r^2 min(s, t)) the product of the
    two Gaussian kernels integrates to
        e^{-q(tau_s + tau_t)} E[S_s S_t] exp(-n' C^{-1} n' / 2) / (2 pi sqrt(det C)),
    where C = Cov(Y) + diag(b_s^2, b_t^2) and n' = E[Y] + Cov(Y) (1, 1)'.
    """
    s, t = np.broadcast_arrays(np.asarray(s, dtype=float), np.asarray(t, dtype=float))
    lo, hi = np.minimum(s, t), np.maximum(s, t)
    tau1, tau2 = T - lo, T - hi
    c_rate = r - q + 0.5 * sigma_h**2
    x0 = math.log(S0 / K)
    v1, v2 = sigma_r**2 * lo, sigma_r**2 * hi
    cov = v1
    n1 = x0 + (mu - 0.5 * sigma_r**2) * lo + c_rate * tau1 + v1 + cov
    n2 = x0 + (mu - 0.5 * sigma_r**2) * hi + c_rate * tau2 + cov + v2
    c11, c22 = v1 + sigma_h**2 * tau1, v2 + sigma_h**2 * tau2
    det = c11 * c22 - cov**2
    quad = (c22 * n1**2 - 2.0 * cov * n1 * n2 + c11 * n2**2) / det
    cross_moment = S0**2 * np.exp(mu * (lo + hi) + sigma_r**2 * lo)  # E[S_s S_t]
    return (np.exp(-q * (tau1 + tau2)) * cross_moment * np.exp(-0.5 * quad)
            / (2.0 * math.pi * np.sqrt(det)))


@functools.lru_cache(maxsize=512)
def gamma_moments(S0: float, K: float, T: float, sigma_h: float, sigma_r: float, r: float = 0.0,
                  q: float = 0.0, mu: float = 0.0, n_quad: int = 64) -> tuple[float, float, float]:
    """The three dollar-gamma integrals behind every variance formula here.

    With G_t = S_t^2 Gamma_h(T - t, S_t) on GBM(mu, sigma_r) paths and w_t = e^{r(T-t)}:
    ``I1 = int w E[G] dt``, ``I2 = int w^2 E[G^2] dt`` and ``V = Var(int w G dt)``.
    """
    t, wq = time_quadrature(T, n_quad)
    w = np.exp(r * (T - t))
    args = (S0, K, T, sigma_h, sigma_r, r, q, mu)
    i1 = float(np.sum(wq * w * expected_dollar_gamma(t, *args)))
    i2 = float(np.sum(wq * w**2 * expected_dollar_gamma_sq(t, *args)))
    ww = wq * w
    second = float(ww @ expected_dollar_gamma_cross(t[:, None], t[None, :], *args) @ ww)
    return i1, i2, max(second - i1**2, 0.0)


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
    _, i2, _ = gamma_moments(S0, K, T, sigma_h, sigma_r, r, q, mu)
    n = np.asarray(n_rebalances, dtype=float)
    return abs(position) * np.sqrt(0.5 * sigma_r**4 * (T / n) * i2)


def vol_gap_std(S0, K, T, sigma_i, sigma_r, r=0.0, q=0.0, mu=0.0, position=-1.0,
                n_rebalances=None) -> float:
    """Std of the P&L of an implied-vol delta hedge on GBM(mu, sigma_r) paths.

    Without ``n_rebalances``: the continuous-hedging part, the std of
    (1/2)(sigma_i^2 - sigma_r^2) int e^{r(T-t)} S^2 Gamma_i dt across paths. With it, the
    discrete-hedging error is included (see :func:`pnl_std`).
    """
    if n_rebalances is not None:
        return float(pnl_std(n_rebalances, S0, K, T, sigma_i, sigma_r, r=r, q=q, mu=mu,
                             position=position))
    _, _, v = gamma_moments(S0, K, T, sigma_i, sigma_r, r, q, mu)
    return abs(0.5 * position * (sigma_i**2 - sigma_r**2)) * math.sqrt(v)


def pnl_std(n_rebalances, S0, K, T, sigma_i, sigma_r, hedge_vol=None, bps=0.0, r=0.0, q=0.0,
            mu=0.0, position=-1.0) -> np.ndarray:
    """Leading-order std of the hedged P&L: discrete hedging + vol gap + proportional costs.

    With G = S^2 Gamma at the hedge vol sigma_h, dt = T/N, k = bps/10^4 and the integrals of
    :func:`gamma_moments`, the variance per unit position is

        (1/2) sigma_r^4 dt I2                        discrete-hedging error
      + (1/4) (sigma_h^2 - sigma_r^2)^2 V            path-dependent vol-gap P&L
      + k^2 (2/pi) sigma_r^2 V / dt                  path-dependent costs: paths that stay near
                                                     the strike carry more gamma and trade more
      + k^2 (1 - 2/pi) sigma_r^2 I2                  noise in each trade's size
      - k sqrt(2/pi) sigma_r (sigma_h^2 - sigma_r^2) V / sqrt(dt)   vol-gap P&L vs costs (short)

    Costs grow like sqrt(N) on average and their variance like N, so with costs the risk stops
    falling as N increases. The covariance between the hedging error and the costs is left
    out: a big move costs twice on the day (hedge error and fees) but also pushes the spot away
    from the strike, cutting future gamma and future fees; the two effects largely cancel
    (in simulation the net term is < 3% of the variance at every frequency tested).
    """
    sigma_h = sigma_i if hedge_vol is None else hedge_vol
    _, i2, v = gamma_moments(S0, K, T, sigma_h, sigma_r, r, q, mu)
    dt = T / np.asarray(n_rebalances, dtype=float)
    k = bps * 1e-4
    gap = sigma_h**2 - sigma_r**2
    sym = (0.5 * sigma_r**4 * dt * i2 + 0.25 * gap**2 * v
           + k**2 * ((2 / math.pi) * sigma_r**2 * v / dt + (1 - 2 / math.pi) * sigma_r**2 * i2))
    cross = -k * _SQRT_2_OVER_PI * sigma_r * gap * v / np.sqrt(dt)
    var = position**2 * sym + (-position) * abs(position) * cross
    return np.sqrt(np.maximum(var, 0.0))


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


def merton_price(S0, K, T, sigma, lam, jump_mean, jump_std, r=0.0, q=0.0, kind="call",
                 n_terms: int = 100) -> float:
    """Merton (1976) jump-diffusion price: a Poisson mixture of Black-Scholes prices.

    Given n jumps the log-price is Gaussian with variance sigma^2 T + n jump_std^2 and the
    compensated drift r - q - lam k + n ln(1 + k) / T, where k = E[J] - 1; the weights use the
    jump-size-adjusted intensity lam (1 + k).
    """
    k = math.expm1(jump_mean + 0.5 * jump_std**2)
    lam_t = lam * (1.0 + k) * T
    total = 0.0
    for n in range(n_terms):
        weight = math.exp(-lam_t + (n * math.log(lam_t) if lam_t > 0 else 0.0) - math.lgamma(n + 1))
        if lam_t == 0 and n > 0:
            break
        sig_n = math.sqrt(sigma**2 + n * jump_std**2 / T)
        r_n = r - lam * k + n * math.log1p(k) / T
        total += weight * float(bs.price(S0, K, T, sig_n, r_n, q, kind))
    return total


def heston_price(S0, K, T, v0, kappa, theta, xi, rho, r=0.0, q=0.0, kind="call") -> float:
    """Heston (1993) price by Fourier inversion (Gil-Pelaez), using the numerically stable
    form of the characteristic function from Albrecher et al. (2007)."""
    from scipy.integrate import quad

    x, log_k = math.log(S0), math.log(K)

    def cf(u: complex) -> complex:
        iu = 1j * u
        beta = kappa - rho * xi * iu
        d = np.sqrt(beta**2 + xi**2 * (iu + u**2))
        g = (beta - d) / (beta + d)
        e = np.exp(-d * T)
        c = (r - q) * iu * T + kappa * theta / xi**2 * ((beta - d) * T - 2.0 * np.log((1 - g * e) / (1 - g)))
        dv = (beta - d) / xi**2 * (1 - e) / (1 - g * e)
        return np.exp(c + dv * v0 + iu * x)

    forward_cf = cf(-1j)  # E[S_T] = S0 e^{(r - q) T}
    p1 = 0.5 + quad(lambda u: (np.exp(-1j * u * log_k) * cf(u - 1j) / (1j * u * forward_cf)).real,
                    0, np.inf, limit=400)[0] / math.pi
    p2 = 0.5 + quad(lambda u: (np.exp(-1j * u * log_k) * cf(u) / (1j * u)).real,
                    0, np.inf, limit=400)[0] / math.pi
    call = S0 * math.exp(-q * T) * p1 - K * math.exp(-r * T) * p2
    if bs.is_call(kind):
        return call
    return call - S0 * math.exp(-q * T) + K * math.exp(-r * T)


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
    std = pnl_std(ns, S0, K, T, sigma_i, sigma_r, None, bps, r, q, mu, position)
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
