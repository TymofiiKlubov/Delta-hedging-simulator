"""Black-Scholes-Merton prices and Greeks for European calls and puts.

All functions broadcast over their array arguments. Conventions:

* ``T``      time to expiry in years (>= 0)
* ``sigma``  annualised volatility (>= 0)
* ``r``      continuously-compounded interest rate
* ``q``      continuous dividend / funding yield paid to holders of the underlying
* ``vega``   per unit of volatility (multiply by 0.01 for "per vol point")
* ``theta``  dV/dt per year of calendar time (divide by 365 for "per day")

Degenerate inputs are handled explicitly. When the total volatility ``sigma * sqrt(T)``
is zero (``T == 0`` or ``sigma == 0``) the option is worth its discounted intrinsic value
on the forward and delta is the discounted exercise indicator (1/2 exactly at the kink).
Gamma, a Dirac mass at the kink in that limit, is reported as 0.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.special import ndtr

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)
_TINY_VOL = 1e-12  # total vol below which the closed-form limit is used


class Greeks(NamedTuple):
    price: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    theta: np.ndarray


class _Core(NamedTuple):
    S: np.ndarray
    K: np.ndarray
    sigma: np.ndarray
    r: np.ndarray
    q: np.ndarray
    sqrt_t: np.ndarray
    vol_t: np.ndarray  # sigma * sqrt(T), with degenerate entries replaced by 1
    deg: np.ndarray  # True where the total volatility is (numerically) zero
    df_r: np.ndarray
    df_q: np.ndarray
    d1: np.ndarray
    d2: np.ndarray


def norm_pdf(x: ArrayLike) -> np.ndarray:
    return _INV_SQRT_2PI * np.exp(-0.5 * np.square(x))


def is_call(kind: str) -> bool:
    k = kind.lower()
    if k in ("call", "c"):
        return True
    if k in ("put", "p"):
        return False
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def payoff(S: ArrayLike, K: ArrayLike, kind: str = "call") -> np.ndarray:
    S = np.asarray(S, dtype=float)
    return np.maximum(S - K, 0.0) if is_call(kind) else np.maximum(K - S, 0.0)


def _core(S, K, T, sigma, r, q) -> _Core:
    S, K, T, sigma, r, q = (np.asarray(x, dtype=float) for x in (S, K, T, sigma, r, q))
    if np.any(T < 0):
        raise ValueError("time to expiry T must be >= 0")
    if np.any(sigma < 0):
        raise ValueError("volatility must be >= 0")
    sqrt_t = np.sqrt(T)
    vol_t = sigma * sqrt_t
    with np.errstate(divide="ignore"):
        log_fk = np.log(S / K) + (r - q) * T  # log(forward / strike)
    deg = vol_t <= _TINY_VOL
    if np.any(deg):
        vol_t = np.where(deg, 1.0, vol_t)
        limit = np.where(log_fk > 0, np.inf, np.where(log_fk < 0, -np.inf, 0.0))
        d1 = np.where(deg, limit, log_fk / vol_t + 0.5 * vol_t)
        d2 = np.where(deg, limit, d1 - vol_t)
    else:
        d1 = log_fk / vol_t + 0.5 * vol_t
        d2 = d1 - vol_t
    return _Core(S, K, sigma, r, q, sqrt_t, vol_t, deg, np.exp(-r * T), np.exp(-q * T), d1, d2)


def price(S, K, T, sigma, r=0.0, q=0.0, kind: str = "call") -> np.ndarray:
    """Black-Scholes-Merton price."""
    c = _core(S, K, T, sigma, r, q)
    if is_call(kind):
        return c.S * c.df_q * ndtr(c.d1) - c.K * c.df_r * ndtr(c.d2)
    return c.K * c.df_r * ndtr(-c.d2) - c.S * c.df_q * ndtr(-c.d1)


def delta(S, K, T, sigma, r=0.0, q=0.0, kind: str = "call") -> np.ndarray:
    """dV/dS."""
    c = _core(S, K, T, sigma, r, q)
    return c.df_q * ndtr(c.d1) if is_call(kind) else -c.df_q * ndtr(-c.d1)


def gamma(S, K, T, sigma, r=0.0, q=0.0) -> np.ndarray:
    """d2V/dS2 (identical for calls and puts)."""
    return _gamma(_core(S, K, T, sigma, r, q))


def vega(S, K, T, sigma, r=0.0, q=0.0) -> np.ndarray:
    """dV/dsigma per unit of volatility (identical for calls and puts)."""
    c = _core(S, K, T, sigma, r, q)
    return c.S * c.df_q * norm_pdf(c.d1) * c.sqrt_t


def theta(S, K, T, sigma, r=0.0, q=0.0, kind: str = "call") -> np.ndarray:
    """dV/dt per year of calendar time (= -dV/dT)."""
    return greeks(S, K, T, sigma, r, q, kind).theta


def greeks(S, K, T, sigma, r=0.0, q=0.0, kind: str = "call") -> Greeks:
    """Price, delta, gamma, vega and theta in one pass (shares d1, d2 and N(.))."""
    c = _core(S, K, T, sigma, r, q)
    pdf = norm_pdf(c.d1)
    s_fwd = c.S * c.df_q  # S e^{-qT}
    k_disc = c.K * c.df_r  # K e^{-rT}
    with np.errstate(divide="ignore", invalid="ignore"):
        decay = np.where(c.sqrt_t > 0, -s_fwd * pdf * c.sigma / (2.0 * c.sqrt_t), 0.0)
    if is_call(kind):
        n1, n2 = ndtr(c.d1), ndtr(c.d2)
        prc = s_fwd * n1 - k_disc * n2
        dlt = c.df_q * n1
        tht = decay - c.r * k_disc * n2 + c.q * s_fwd * n1
    else:
        m1, m2 = ndtr(-c.d1), ndtr(-c.d2)
        prc = k_disc * m2 - s_fwd * m1
        dlt = -c.df_q * m1
        tht = decay + c.r * k_disc * m2 - c.q * s_fwd * m1
    return Greeks(prc, dlt, _gamma(c, pdf), s_fwd * pdf * c.sqrt_t, tht)


def _gamma(c: _Core, pdf: np.ndarray | None = None) -> np.ndarray:
    pdf = norm_pdf(c.d1) if pdf is None else pdf
    with np.errstate(divide="ignore", invalid="ignore"):
        g = c.df_q * pdf / (c.S * c.vol_t)
    return np.where(c.deg, 0.0, g) if np.any(c.deg) else g


def implied_vol(target_price: float, S: float, K: float, T: float, r: float = 0.0,
                q: float = 0.0, kind: str = "call", lo: float = 1e-6, hi: float = 10.0) -> float:
    """Black-Scholes implied volatility of a single price (Brent's method)."""
    if T <= 0:
        raise ValueError("implied vol is undefined at expiry")
    f = lambda s: float(price(S, K, T, s, r, q, kind)) - target_price  # noqa: E731
    if f(lo) > 0 or f(hi) < 0:
        raise ValueError("price outside the Black-Scholes no-arbitrage range")
    return brentq(f, lo, hi, xtol=1e-12)
