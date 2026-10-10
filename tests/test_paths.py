import math

import numpy as np
import pytest

from dhsim.paths import GBM, Heston, Merton, PathSet, chunk_sizes, time_grid

S0, T = 60_000.0, 30 / 365


def _mean_within(x, target, n_se=4.0):
    se = x.std(ddof=1) / math.sqrt(x.size)
    assert abs(x.mean() - target) < n_se * se, (x.mean(), target, se)


def test_gbm_shapes_grid_and_reproducibility():
    p = GBM(0.6).simulate(S0, T, 30, 5, np.random.default_rng(7))
    assert p.S.shape == (5, 31) and p.t.shape == (31,)
    assert p.t[0] == 0.0 and p.t[-1] == T
    np.testing.assert_array_equal(p.S[:, 0], S0)
    q = GBM(0.6).simulate(S0, T, 30, 5, np.random.default_rng(7))
    np.testing.assert_array_equal(p.S, q.S)


@pytest.mark.parametrize("mu", [0.0, 0.3])
def test_gbm_moments(mu):
    sigma = 0.6
    p = GBM(sigma).simulate(S0, T, 24, 200_000, np.random.default_rng(1), mu=mu)
    _mean_within(p.S[:, -1], S0 * math.exp(mu * T))
    log_ret = np.diff(np.log(p.S), axis=1)
    assert log_ret.std() == pytest.approx(sigma * math.sqrt(T / 24), rel=5e-3)
    assert p.realised_vol().mean() == pytest.approx(sigma, rel=2e-2)


def test_subsample_keeps_the_same_path():
    p = GBM(0.6).simulate(S0, T, 12, 3, np.random.default_rng(2))
    s = p.subsample(4)
    assert s.n_steps == 3
    np.testing.assert_array_equal(s.S, p.S[:, [0, 4, 8, 12]])
    np.testing.assert_array_equal(s.t, p.t[[0, 4, 8, 12]])
    with pytest.raises(ValueError):
        p.subsample(5)


def test_merton_matched_variance_and_martingale():
    m = Merton.matched(0.6, lam=5.0, jump_mean=-0.05, jump_std=0.15)
    assert m.total_vol == pytest.approx(0.6)
    assert m.sigma == pytest.approx(math.sqrt(0.36 - 5 * (0.05**2 + 0.15**2)))
    p = m.simulate(S0, 1.0, 50, 100_000, np.random.default_rng(3), mu=0.1)
    _mean_within(p.S[:, -1], S0 * math.exp(0.1))
    log_ret = np.diff(np.log(p.S), axis=1)
    qv = np.sum(log_ret**2, axis=1)  # realised quadratic variation over one year
    assert qv.mean() == pytest.approx(0.36, rel=1e-2)
    with pytest.raises(ValueError):
        Merton.matched(0.2, lam=5.0, jump_mean=0.0, jump_std=0.15)


def test_heston_variance_mean_reversion_and_martingale():
    h = Heston(v0=0.20, kappa=3.0, theta=0.36, xi=1.0, rho=-0.5)
    p = h.simulate(S0, 0.5, 500, 40_000, np.random.default_rng(4), mu=0.05)
    expected_v = h.theta + (h.v0 - h.theta) * math.exp(-h.kappa * 0.5)
    assert p.v[:, -1].mean() == pytest.approx(expected_v, rel=2e-2)
    _mean_within(p.S[:, -1], S0 * math.exp(0.05 * 0.5))
    assert np.mean(p.realised_vol() ** 2) == pytest.approx(h.expected_variance(0.5), rel=2e-2)
    assert np.all(p.v >= 0.0)


def test_heston_without_vol_of_vol_is_gbm():
    h = Heston(v0=0.36, kappa=2.0, theta=0.36, xi=0.0, rho=0.0)
    p = h.simulate(S0, T, 40, 50_000, np.random.default_rng(5))
    np.testing.assert_allclose(p.v, 0.36)
    log_ret = np.diff(np.log(p.S), axis=1)
    assert log_ret.std() == pytest.approx(0.6 * math.sqrt(T / 40), rel=5e-3)


def test_chunking_and_grid_helpers():
    assert sum(chunk_sizes(10_001, 99, 1_000)) == 10_001
    assert max(chunk_sizes(10_001, 99, 1_000)) == 10
    assert chunk_sizes(5, 10_000, 100) == [1, 1, 1, 1, 1]
    with pytest.raises(ValueError):
        time_grid(1.0, 0)
    assert isinstance(GBM(0.1).simulate(1.0, 1.0, 1, 1, np.random.default_rng(0)), PathSet)
