"""Price-path simulators on a uniform time grid.

Each model has ``simulate(S0, T, n_steps, n_paths, rng, mu=0.0) -> PathSet`` where ``mu``
is the real-world drift of the spot price (excluding any yield ``q``): ``E[S_T] = S0 e^{mu T}``
for every model. Volatilities are annualised and time is in years.

* :class:`GBM`     exact log-normal simulation with volatility ``sigma``
* :class:`Merton`  GBM plus compensated log-normal jumps (Merton 1976)
* :class:`Heston`  stochastic variance, full-truncation Euler (Lord, Koekkoek & van Dijk 2010)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class PathSet:
    """Simulated paths: ``S`` has shape (n_paths, n_steps + 1) on the time grid ``t``."""

    t: np.ndarray
    S: np.ndarray
    v: np.ndarray | None = None  # instantaneous variance (Heston only)

    @property
    def n_paths(self) -> int:
        return self.S.shape[0]

    @property
    def n_steps(self) -> int:
        return self.S.shape[1] - 1

    def subsample(self, every: int) -> PathSet:
        """Keep every ``every``-th grid point (the same paths observed less often)."""
        if every < 1 or self.n_steps % every:
            raise ValueError(f"cannot subsample {self.n_steps} steps every {every}")
        v = None if self.v is None else self.v[:, ::every]
        return PathSet(self.t[::every], self.S[:, ::every], v)

    def realised_vol(self) -> np.ndarray:
        """Annualised realised volatility of each path, sqrt(sum (d log S)^2 / T)."""
        log_ret = np.diff(np.log(self.S), axis=1)
        return np.sqrt(np.sum(log_ret**2, axis=1) / (self.t[-1] - self.t[0]))


class Model(Protocol):
    def simulate(self, S0: float, T: float, n_steps: int, n_paths: int,
                 rng: np.random.Generator, mu: float = 0.0) -> PathSet: ...


def time_grid(T: float, n_steps: int) -> np.ndarray:
    if n_steps < 1:
        raise ValueError("need at least one time step")
    return np.linspace(0.0, T, n_steps + 1)


def chunk_sizes(n_paths: int, n_steps: int, max_elements: int) -> list[int]:
    """Split ``n_paths`` so that each chunk holds at most ~``max_elements`` path points."""
    per = max(1, int(max_elements) // max(int(n_steps) + 1, 1))
    return [min(per, n_paths - start) for start in range(0, n_paths, per)]


def _from_log_increments(S0: float, T: float, inc: np.ndarray) -> PathSet:
    n_paths, n_steps = inc.shape
    S = np.empty((n_paths, n_steps + 1))
    S[:, 0] = 0.0
    np.cumsum(inc, axis=1, out=S[:, 1:])
    np.exp(S, out=S)
    S *= S0
    return PathSet(time_grid(T, n_steps), S)


@dataclass(frozen=True)
class GBM:
    sigma: float

    @property
    def total_vol(self) -> float:
        return self.sigma

    def simulate(self, S0, T, n_steps, n_paths, rng, mu=0.0) -> PathSet:
        dt = T / n_steps
        inc = rng.standard_normal((n_paths, n_steps))
        inc *= self.sigma * math.sqrt(dt)
        inc += (mu - 0.5 * self.sigma**2) * dt
        return _from_log_increments(S0, T, inc)


@dataclass(frozen=True)
class Merton:
    """dS/S = (mu - lam*k) dt + sigma dW + (J - 1) dN with log J ~ N(jump_mean, jump_std^2)."""

    sigma: float  # diffusive volatility
    lam: float  # jump intensity per year
    jump_mean: float
    jump_std: float

    @classmethod
    def matched(cls, total_vol: float, lam: float, jump_mean: float, jump_std: float) -> Merton:
        """Choose the diffusive vol so that the expected quadratic variation of log S
        equals ``total_vol**2`` per year: sigma^2 + lam (m^2 + s^2) = total_vol^2."""
        jump_var = lam * (jump_mean**2 + jump_std**2)
        if jump_var >= total_vol**2:
            raise ValueError("jump variance exceeds the target total variance")
        return cls(math.sqrt(total_vol**2 - jump_var), lam, jump_mean, jump_std)

    @property
    def total_vol(self) -> float:
        return math.sqrt(self.sigma**2 + self.lam * (self.jump_mean**2 + self.jump_std**2))

    def simulate(self, S0, T, n_steps, n_paths, rng, mu=0.0) -> PathSet:
        dt = T / n_steps
        kbar = math.expm1(self.jump_mean + 0.5 * self.jump_std**2)  # E[J - 1]
        inc = rng.standard_normal((n_paths, n_steps))
        inc *= self.sigma * math.sqrt(dt)
        inc += (mu - self.lam * kbar - 0.5 * self.sigma**2) * dt
        n_jumps = rng.poisson(self.lam * dt, size=(n_paths, n_steps))
        i, j = np.nonzero(n_jumps)
        k = n_jumps[i, j]
        inc[i, j] += k * self.jump_mean + np.sqrt(k) * self.jump_std * rng.standard_normal(k.size)
        return _from_log_increments(S0, T, inc)


@dataclass(frozen=True)
class Heston:
    """dS/S = mu dt + sqrt(v) dW1,  dv = kappa (theta - v) dt + xi sqrt(v) dW2,  d<W1,W2> = rho dt."""

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def expected_variance(self, T: float) -> float:
        """E[(1/T) int_0^T v_t dt], the expected realised variance per year."""
        if self.kappa * T < 1e-12:
            return self.v0
        return self.theta + (self.v0 - self.theta) * (-math.expm1(-self.kappa * T)) / (self.kappa * T)

    def simulate(self, S0, T, n_steps, n_paths, rng, mu=0.0) -> PathSet:
        dt = T / n_steps
        sdt = math.sqrt(dt)
        rho_c = math.sqrt(1.0 - self.rho**2)
        log_s = np.empty((n_paths, n_steps + 1))
        var = np.empty((n_paths, n_steps + 1))
        log_s[:, 0] = math.log(S0)
        var[:, 0] = self.v0
        x = log_s[:, 0].copy()
        v = np.full(n_paths, float(self.v0))
        for k in range(n_steps):
            z1, z2 = rng.standard_normal((2, n_paths))
            vp = np.maximum(v, 0.0)  # full truncation: negative variance is treated as zero
            sv = np.sqrt(vp) * sdt
            x += (mu - 0.5 * vp) * dt + sv * z1
            v += self.kappa * (self.theta - vp) * dt + self.xi * sv * (self.rho * z1 + rho_c * z2)
            log_s[:, k + 1] = x
            var[:, k + 1] = np.maximum(v, 0.0)
        return PathSet(time_grid(T, n_steps), np.exp(log_s), var)
