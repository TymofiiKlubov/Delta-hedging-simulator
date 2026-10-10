import math

import numpy as np
import pytest

from dhsim import bs
from dhsim import theory as th
from dhsim.hedging import Costs, Market, Option, run_hedge
from dhsim.paths import GBM
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


def test_optimal_frequency_scales_like_cost_to_the_minus_two_thirds():
    # J ~ a sqrt(N) k + b / N  =>  N* ~ k^(-2/3)
    n1 = th.optimal_rebalances(S0, K, T, 0.6, 0.6, bps=1.0).best
    n8 = th.optimal_rebalances(S0, K, T, 0.6, 0.6, bps=8.0).best
    assert n1 / n8 == pytest.approx(8 ** (2 / 3), rel=0.15)
    tradeoff = th.cost_tradeoff([1, n8, 100_000], S0, K, T, 0.6, 0.6, bps=8.0)
    assert tradeoff.objective[1] < tradeoff.objective[0] and tradeoff.objective[1] < tradeoff.objective[2]


def test_cost_formula_counts_fixed_fees_per_trade():
    base = th.expected_hedge_cost(30, S0, K, T, 0.6, bps=5.0)
    with_fixed = th.expected_hedge_cost(30, S0, K, T, 0.6, bps=5.0, fixed=2.0)
    assert with_fixed - base == pytest.approx(2.0 * 31)
