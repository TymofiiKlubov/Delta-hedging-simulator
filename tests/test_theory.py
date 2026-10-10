import math

import numpy as np
import pytest

from dhsim import bs
from dhsim import theory as th
from dhsim.hedging import Costs, Market, Option, run_hedge
from dhsim.paths import GBM, Heston, Merton
from dhsim.stats import summarize

S0, K, T = 60_000.0, 60_000.0, 30 / 365


@pytest.mark.parametrize("sigma_h, sigma_r, strike, r, q, mu", [
    (0.6, 0.6, 60_000.0, 0.0, 0.0, 0.0),
    (0.6, 0.45, 66_000.0, 0.03, 0.01, 0.2),
    (0.5, 0.8, 54_000.0, 0.0, 0.02, -0.3),
])
def test_dollar_gamma_moments_match_monte_carlo(sigma_h, sigma_r, strike, r, q, mu):
    rng = np.random.default_rng(0)
    for t in (0.2 * T, 0.8 * T):
        z = rng.standard_normal(400_000)
        s_t = S0 * np.exp((mu - 0.5 * sigma_r**2) * t + sigma_r * math.sqrt(t) * z)
        dg = s_t**2 * bs.gamma(s_t, strike, T - t, sigma_h, r, q)
        for sample, closed in [
            (dg, th.expected_dollar_gamma(t, S0, strike, T, sigma_h, sigma_r, r, q, mu)),
            (dg**2, th.expected_dollar_gamma_sq(t, S0, strike, T, sigma_h, sigma_r, r, q, mu)),
        ]:
            se = sample.std() / math.sqrt(sample.size)
            assert abs(sample.mean() - closed) < 4 * se


@pytest.mark.parametrize("s, t, sigma_r, strike, r, q, mu", [
    (0.2 * T, 0.7 * T, 0.45, 60_000.0, 0.0, 0.0, 0.0),
    (0.5 * T, 0.95 * T, 0.8, 63_000.0, 0.03, 0.01, 0.1),
    (0.3 * T, 0.3 * T, 0.6, 60_000.0, 0.0, 0.0, 0.0),  # s = t: reduces to the second moment
])
def test_dollar_gamma_cross_moment_matches_monte_carlo(s, t, sigma_r, strike, r, q, mu):
    rng = np.random.default_rng(3)
    z1, z2 = rng.standard_normal((2, 400_000))
    s_s = S0 * np.exp((mu - 0.5 * sigma_r**2) * s + sigma_r * math.sqrt(s) * z1)
    s_t = s_s * np.exp((mu - 0.5 * sigma_r**2) * (t - s) + sigma_r * math.sqrt(t - s) * z2)
    prod = (s_s**2 * bs.gamma(s_s, strike, T - s, 0.6, r, q)) * (s_t**2 * bs.gamma(s_t, strike, T - t, 0.6, r, q))
    closed = th.expected_dollar_gamma_cross(s, t, S0, strike, T, 0.6, sigma_r, r, q, mu)
    assert abs(prod.mean() - closed) < 4 * prod.std() / math.sqrt(prod.size)
    if s == t:
        assert closed == pytest.approx(th.expected_dollar_gamma_sq(t, S0, strike, T, 0.6, sigma_r, r, q, mu), rel=1e-12)


def test_vol_gap_std_matches_the_simulated_pathwise_integral():
    paths = GBM(0.45).simulate(S0, T, 1000, 8000, np.random.default_rng(5))
    S, tau = paths.S[:, :-1], T - paths.t[:-1]
    pathwise = 0.5 * (0.6**2 - 0.45**2) * ((S**2 * bs.gamma(S, K, tau, 0.6)) @ np.diff(paths.t))
    assert pathwise.std() == pytest.approx(th.vol_gap_std(S0, K, T, 0.6, 0.45), rel=0.03)
    assert th.vol_gap_std(S0, K, T, 0.6, 0.6) == 0.0


def test_general_formula_reduces_to_derman_kamal_at_the_money():
    n = np.array([1, 30, 720])
    np.testing.assert_allclose(th.discretisation_std(n, S0, K, T, 0.6),
                               th.derman_kamal_std(n, S0, K, T, 0.6), rtol=5e-3)
    # away from the money the ATM rule understates the hedging error
    for strike in (54_000.0, 66_000.0):
        assert th.discretisation_std(30, S0, strike, T, 0.6) > 1.04 * th.derman_kamal_std(30, S0, strike, T, 0.6)


@pytest.mark.parametrize("sigma_r, r, q", [(0.45, 0.0, 0.0), (0.8, 0.05, 0.02), (0.3, -0.01, 0.03)])
def test_implied_vol_hedge_mean_equals_price_gap_under_risk_neutral_drift(sigma_r, r, q):
    implied = th.vol_gap_mean(S0, K, T, 0.6, sigma_r, r, q, mu=r - q, hedge="implied")
    realised = th.vol_gap_mean(S0, K, T, 0.6, sigma_r, r, q, mu=r - q, hedge="realised")
    assert implied == pytest.approx(realised, rel=1e-10)
    assert np.sign(realised) == np.sign(0.6 - sigma_r)  # the short wins when sigma_r < sigma_i


def test_simulated_hedging_error_and_costs_match_theory():
    # leading order in dt: at N ~ 30 the formula still overstates the std by ~2-3%, so the
    # check uses frequencies where the expansion is accurate
    paths = GBM(0.6).simulate(S0, T, 720, 4000, np.random.default_rng(1))
    option, market = Option(K, T), Market(S0)
    for n in (120, 240, 720):
        sub = paths.subsample(720 // n)
        res = run_hedge(sub.S, sub.t, option, market, 0.6, costs=Costs(bps=5.0))
        frictionless = summarize(res.pnl - res.costs)
        assert frictionless["std"] == pytest.approx(th.discretisation_std(n, S0, K, T, 0.6), rel=0.05)
        assert -res.costs.mean() == pytest.approx(th.expected_hedge_cost(n, S0, K, T, 0.6, bps=5.0), rel=0.03)


def test_costs_make_risk_rise_again_at_high_frequency():
    # path-dependent costs: their variance grows like N while the hedging error shrinks like 1/N
    paths = GBM(0.6).simulate(S0, T, 1440, 4000, np.random.default_rng(12))
    option, market = Option(K, T), Market(S0)
    for n in (240, 1440):
        sub = paths.subsample(1440 // n)
        res = run_hedge(sub.S, sub.t, option, market, 0.6, costs=Costs(bps=5.0))
        theory = th.pnl_std(n, S0, K, T, 0.6, 0.6, bps=5.0)
        assert res.pnl.std() == pytest.approx(theory, rel=0.05)
        assert (-res.costs).std() == pytest.approx(
            math.sqrt(theory**2 - th.discretisation_std(n, S0, K, T, 0.6) ** 2), rel=0.1)
    # with 5 bp the risk bottoms out around N ~ 1000 and rises again beyond
    assert th.pnl_std(2880, S0, K, T, 0.6, 0.6, bps=5.0) > th.pnl_std(960, S0, K, T, 0.6, 0.6, bps=5.0)


def test_optimal_frequency_scales_like_cost_to_the_minus_two_thirds():
    # J ~ a sqrt(N) k + b / N  =>  N* ~ k^(-2/3)
    n1 = th.optimal_rebalances(S0, K, T, 0.6, 0.6, bps=1.0).best
    n8 = th.optimal_rebalances(S0, K, T, 0.6, 0.6, bps=8.0).best
    assert n1 / n8 == pytest.approx(8 ** (2 / 3), rel=0.15)
    tradeoff = th.cost_tradeoff([1, n8, 100_000], S0, K, T, 0.6, 0.6, bps=8.0)
    assert tradeoff.objective[1] < tradeoff.objective[0] and tradeoff.objective[1] < tradeoff.objective[2]


def test_merton_price_limits_and_monte_carlo():
    assert th.merton_price(S0, K, T, 0.6, 0.0, -0.05, 0.15) == pytest.approx(float(bs.price(S0, K, T, 0.6)), rel=1e-12)
    m = Merton.matched(0.6, lam=5.0, jump_mean=-0.05, jump_std=0.15)
    for strike, r, q, kind in [(60_000.0, 0.0, 0.0, "call"), (54_000.0, 0.03, 0.01, "put")]:
        p = m.simulate(S0, T, 20, 300_000, np.random.default_rng(6), mu=r - q)
        disc = math.exp(-r * T) * bs.payoff(p.S[:, -1], strike, kind)
        closed = th.merton_price(S0, strike, T, m.sigma, m.lam, m.jump_mean, m.jump_std, r, q, kind)
        assert abs(disc.mean() - closed) < 4 * disc.std() / math.sqrt(disc.size)


def test_heston_price_limits_and_monte_carlo():
    # no vol of vol: Black-Scholes at sqrt(theta)
    assert th.heston_price(S0, K, T, 0.36, 3.0, 0.36, 1e-4, 0.0) == pytest.approx(float(bs.price(S0, K, T, 0.6)), rel=1e-6)
    h = Heston(v0=0.30, kappa=2.0, theta=0.40, xi=1.0, rho=-0.5)
    p = h.simulate(S0, T, 400, 60_000, np.random.default_rng(7), mu=0.02)
    for strike in (54_000.0, 60_000.0, 66_000.0):
        disc = math.exp(-0.02 * T) * np.maximum(p.S[:, -1] - strike, 0.0)
        closed = th.heston_price(S0, strike, T, h.v0, h.kappa, h.theta, h.xi, h.rho, r=0.02)
        assert abs(disc.mean() - closed) < 4 * disc.std() / math.sqrt(disc.size)


def test_cost_formula_counts_fixed_fees_per_trade():
    base = th.expected_hedge_cost(30, S0, K, T, 0.6, bps=5.0)
    with_fixed = th.expected_hedge_cost(30, S0, K, T, 0.6, bps=5.0, fixed=2.0)
    assert with_fixed - base == pytest.approx(2.0 * 31)
