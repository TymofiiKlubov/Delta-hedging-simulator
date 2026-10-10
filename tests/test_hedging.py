import math

import numpy as np
import pytest

from dhsim import bs
from dhsim.hedging import Costs, HedgeResult, Market, Option, Strategy, run_hedge
from dhsim.paths import GBM, time_grid
from dhsim.stats import summarize

S0, K, T = 60_000.0, 60_000.0, 30 / 365
CALL = Option(K, T, "call", -1.0)
FLAT = Market(S0)


def simulate(sigma, n_steps, n_paths, seed=0, mu=0.0, T=T):
    return GBM(sigma).simulate(S0, T, n_steps, n_paths, np.random.default_rng(seed), mu=mu)


def reference_pnl(S, t, option, market, sigma_i, strategy, costs):
    """Independent scalar implementation: an explicit cash account, path by path."""
    K, T, pos, kind = option.strike, option.expiry, option.position, option.kind
    r, q = market.r, market.q
    hv = sigma_i if strategy.hedge_vol is None else strategy.hedge_vol
    band = None if strategy.band is None else strategy.band * abs(pos)
    out = []
    for path in S:
        n = len(path) - 1
        cash, h = -pos * float(bs.price(path[0], K, T, sigma_i, r, q, kind)), 0.0
        for k in range(n + 1):
            if k > 0:  # accrue interest and the hedge's yield over (t_{k-1}, t_k]
                dt = t[k] - t[k - 1]
                cash = cash * math.exp(r * dt) + h * path[k] * (math.exp(q * dt) - 1.0)
            if k == n:
                new_h = 0.0  # unwind at expiry
            else:
                target = -pos * float(bs.delta(path[k], K, T - t[k], hv, r, q, kind))
                if k == 0 or (k % strategy.every == 0 and band is None):
                    new_h = target
                elif k % strategy.every == 0 and abs(target - h) > band:
                    gap = target - h
                    new_h = target if strategy.band_mode == "target" else target - math.copysign(band, gap)
                else:
                    new_h = h
            trade = new_h - h
            cash -= trade * path[k] + costs.bps * 1e-4 * abs(trade) * path[k]
            cash -= costs.fixed if trade != 0 else 0.0
            h = new_h
        out.append(cash + pos * float(option.payoff(path[-1])))
    return np.array(out)


@pytest.mark.parametrize("strategy", [
    Strategy(),
    Strategy(every=4),
    Strategy(hedge_vol=0.45, every=3),
    Strategy(band=0.03, every=2),
    Strategy(band=0.05, band_mode="edge"),
])
def test_accounting_identity_matches_an_independent_cash_account(strategy):
    option = Option(63_000.0, 45 / 365, "put", -2.0)
    market = Market(S0, r=0.04, q=0.015)
    costs = Costs(bps=7.0, fixed=2.5)
    p = GBM(0.7).simulate(S0, option.expiry, 24, 12, np.random.default_rng(11), mu=0.1)
    res = run_hedge(p.S, p.t, option, market, 0.6, strategy, costs, decompose=True)
    ref = reference_pnl(p.S, p.t, option, market, 0.6, strategy, costs)
    np.testing.assert_allclose(res.pnl, ref, rtol=0, atol=1e-7 * S0)
    total = sum(res.attribution().values())
    np.testing.assert_allclose(total, res.pnl, rtol=0, atol=1e-7 * S0)
    split = sum(res.decomposition().values())
    np.testing.assert_allclose(split, res.pnl, rtol=0, atol=1e-7 * S0)
    np.testing.assert_allclose(res.costs, -(costs.bps * 1e-4 * res.turnover + costs.fixed * res.n_trades))


def test_matching_vols_and_fine_hedging_give_zero_mean_and_small_std():
    premium = float(bs.price(S0, K, T, 0.6))
    p = simulate(0.6, 2000, 4000, seed=1)
    fine = summarize(run_hedge(p.S, p.t, CALL, FLAT, 0.6).pnl)
    assert abs(fine["mean"]) < 3 * fine["se_mean"]
    assert fine["std"] < 0.03 * premium
    coarse = summarize(run_hedge(p.subsample(16).S, p.subsample(16).t, CALL, FLAT, 0.6).pnl)
    assert abs(coarse["mean"]) < 3 * coarse["se_mean"]
    # 16x more rebalancing -> ~4x smaller hedging error
    assert coarse["std"] / fine["std"] == pytest.approx(4.0, rel=0.1)


def test_hedging_with_realised_vol_locks_in_the_price_difference():
    sigma_i, sigma_r = 0.6, 0.45
    locked = float(bs.price(S0, K, T, sigma_i) - bs.price(S0, K, T, sigma_r))
    p = simulate(sigma_r, 2000, 4000, seed=2)
    realised = summarize(run_hedge(p.S, p.t, CALL, FLAT, sigma_i, Strategy(hedge_vol=sigma_r)).pnl)
    implied = summarize(run_hedge(p.S, p.t, CALL, FLAT, sigma_i).pnl)
    assert abs(realised["mean"] - locked) < 3 * realised["se_mean"]
    assert realised["std"] < 0.1 * locked
    # hedging at the implied vol earns the same on average but path-dependently
    assert abs(implied["mean"] - locked) < 3 * implied["se_mean"]
    assert implied["std"] > 3 * realised["std"]


@pytest.mark.parametrize("n_steps, strategy", [(4, Strategy()), (30, Strategy(hedge_vol=0.8)),
                                               (60, Strategy(band=0.05))])
def test_expected_pnl_is_the_price_difference_for_any_hedge(n_steps, strategy):
    # With mu = r - q the discounted hedge gains are martingale increments, so the mean P&L
    # is exactly BS(sigma_i) - BS(sigma_r) however coarsely or oddly the option is hedged.
    p = simulate(0.75, n_steps, 40_000, seed=3)
    s = summarize(run_hedge(p.S, p.t, CALL, FLAT, 0.6, strategy).pnl)
    expected = float(bs.price(S0, K, T, 0.6) - bs.price(S0, K, T, 0.75))
    assert abs(s["mean"] - expected) < 3 * s["se_mean"]


def test_decomposition_gamma_versus_theta():
    p = simulate(0.6, 720, 200, seed=4)
    res = run_hedge(p.S, p.t, CALL, FLAT, 0.6, decompose=True, record=True)
    np.testing.assert_allclose(res.mismatch, 0.0, atol=1e-9)  # holding == sigma_i delta
    np.testing.assert_allclose(sum(res.decomposition().values()), res.pnl, atol=1e-7 * S0)
    assert np.all(res.theta > 0) and np.all(res.gamma < 0)  # short: earn theta, pay gamma
    assert np.mean(np.abs(res.residual)) < 0.1 * np.mean(np.abs(res.gamma))
    # with r = q = 0 theta is exactly the implied-variance carry 0.5 Gamma S^2 sigma_i^2 dt
    st = res.steps
    carry_i = 0.5 * st["gamma_greek"] * st["S"][:, :-1] ** 2 * 0.6**2 * np.diff(st["t"])
    np.testing.assert_allclose(st["theta"], carry_i, rtol=1e-10, atol=1e-12)
    assert st["mtm_pnl"].shape == p.S.shape
    np.testing.assert_allclose(st["mtm_pnl"][:, -1], res.pnl, atol=1e-8 * S0)


@pytest.mark.parametrize("strike", [50_000.0, 70_000.0, 60_000.0 * math.exp((0.03 - 0.01) * T)])
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("n_steps", [1, 7, 100])
def test_zero_vol_paths_replicate_exactly(strike, kind, n_steps):
    market = Market(S0, r=0.03, q=0.01)
    option = Option(strike, T, kind, -1.0)
    t = time_grid(T, n_steps)
    S = S0 * np.exp((market.r - market.q) * t)[None, :]  # the deterministic forward path
    # sold at zero implied vol: exact replication, P&L = 0
    res = run_hedge(S, t, option, market, 0.0)
    assert res.pnl[0] == pytest.approx(0.0, abs=1e-8 * S0)
    # sold at 60% vol and hedged with the (zero) realised vol: locks in the whole premium gap
    res = run_hedge(S, t, option, market, 0.6, Strategy(hedge_vol=0.0))
    gap = float(bs.price(S0, strike, T, 0.6, 0.03, 0.01, kind) - bs.price(S0, strike, T, 0.0, 0.03, 0.01, kind))
    assert res.pnl[0] == pytest.approx(gap * math.exp(0.03 * T), rel=1e-12, abs=1e-8 * S0)


def test_band_limits_reduce_to_fixed_and_static_hedges():
    p = simulate(0.6, 48, 50, seed=5)
    fixed = run_hedge(p.S, p.t, CALL, FLAT, 0.6)
    zero_band = run_hedge(p.S, p.t, CALL, FLAT, 0.6, Strategy(band=0.0))
    np.testing.assert_allclose(zero_band.pnl, fixed.pnl, atol=1e-9)
    static = run_hedge(p.S, p.t, CALL, FLAT, 0.6, Strategy(band=10.0))
    delta0 = float(bs.delta(S0, K, T, 0.6))
    expected = bs.price(S0, K, T, 0.6) - CALL.payoff(p.S[:, -1]) + delta0 * (p.S[:, -1] - S0)
    np.testing.assert_allclose(static.pnl, expected, atol=1e-8 * S0)
    np.testing.assert_array_equal(static.n_trades, 2)  # initial hedge and unwind only


def test_position_scaling_and_sign():
    p = simulate(0.6, 30, 100, seed=6)
    one = run_hedge(p.S, p.t, CALL, FLAT, 0.6, costs=Costs(bps=5.0))
    three = run_hedge(p.S, p.t, Option(K, T, "call", -3.0), FLAT, 0.6, costs=Costs(bps=5.0))
    np.testing.assert_allclose(three.pnl, 3 * one.pnl, rtol=0, atol=1e-9 * S0)
    long = run_hedge(p.S, p.t, Option(K, T, "call", +1.0), FLAT, 0.6)
    short = run_hedge(p.S, p.t, CALL, FLAT, 0.6)
    np.testing.assert_allclose(long.pnl, -short.pnl, atol=1e-9)


def test_costs_reduce_pnl_by_rate_times_turnover():
    p = simulate(0.6, 30, 200, seed=7)
    free = run_hedge(p.S, p.t, CALL, FLAT, 0.6)
    paid = run_hedge(p.S, p.t, CALL, FLAT, 0.6, costs=Costs(bps=10.0, fixed=1.0))
    np.testing.assert_allclose(free.pnl - paid.pnl, 10e-4 * paid.turnover + paid.n_trades, rtol=1e-12)
    # initial trade + 29 rebalances + unwind; deep in the money N(d1) rounds to exactly 1.0
    # on consecutive dates, and then there is nothing to trade
    assert paid.n_trades.max() == 31 and np.mean(paid.n_trades == 31) > 0.9


def test_put_is_hedged_like_a_call():
    p = simulate(0.6, 720, 3000, seed=8)
    s = summarize(run_hedge(p.S, p.t, Option(K, T, "put"), FLAT, 0.6).pnl)
    assert abs(s["mean"]) < 3 * s["se_mean"]
    assert s["std"] < 0.05 * float(bs.price(S0, K, T, 0.6, kind="put"))


def test_precomputed_delta_and_chunk_concat_match_direct_run():
    p = simulate(0.6, 60, 40, seed=9)
    direct = run_hedge(p.S, p.t, CALL, FLAT, 0.6, Strategy(band=0.02))
    delta = bs.delta(p.S[:, :-1], K, T - p.t[:-1], 0.6)
    shared = run_hedge(p.S, p.t, CALL, FLAT, 0.6, Strategy(band=0.02), target_delta=delta)
    np.testing.assert_array_equal(shared.pnl, direct.pnl)
    halves = [run_hedge(p.S[s], p.t, CALL, FLAT, 0.6, Strategy(band=0.02)) for s in (slice(0, 20), slice(20, 40))]
    np.testing.assert_allclose(HedgeResult.concat(halves).pnl, direct.pnl)


def test_input_validation():
    p = simulate(0.6, 10, 2)
    with pytest.raises(ValueError):
        run_hedge(p.S, p.t * 2, CALL, FLAT, 0.6)  # grid does not end at expiry
    with pytest.raises(ValueError):
        run_hedge(p.S[:, :1], p.t[:1], CALL, FLAT, 0.6)  # no time step
    with pytest.raises(ValueError):
        run_hedge(p.S, p.t, CALL, FLAT, 0.6, Strategy(band=0.1, band_mode="sideways"))
    with pytest.raises(ValueError):
        run_hedge(p.S, p.t, CALL, FLAT, 0.6, Strategy(every=0))
    with pytest.raises(ValueError):
        run_hedge(p.S, p.t, Option(K, 0.0), FLAT, 0.6)
