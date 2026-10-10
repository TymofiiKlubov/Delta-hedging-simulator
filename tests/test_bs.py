import numpy as np
import pytest

from dhsim import bs

CASES = [
    # S, K, T, sigma, r, q, kind
    (60_000.0, 60_000.0, 30 / 365, 0.60, 0.00, 0.00, "call"),
    (60_000.0, 66_000.0, 30 / 365, 0.60, 0.03, 0.01, "call"),
    (60_000.0, 54_000.0, 90 / 365, 0.45, 0.05, 0.02, "put"),
    (100.0, 110.0, 1.0, 0.20, 0.04, 0.06, "put"),
    (100.0, 90.0, 0.25, 0.30, -0.01, 0.00, "call"),
]


@pytest.mark.parametrize("S, K, T, sigma, r, q, kind", CASES)
def test_price_matches_monte_carlo_within_3_se(S, K, T, sigma, r, q, kind):
    rng = np.random.default_rng(1234)
    z = rng.standard_normal(1_000_000)
    s_t = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    disc_payoff = np.exp(-r * T) * bs.payoff(s_t, K, kind)
    mc, se = disc_payoff.mean(), disc_payoff.std(ddof=1) / np.sqrt(z.size)
    assert abs(mc - bs.price(S, K, T, sigma, r, q, kind)) < 3 * se


def test_textbook_value():
    # Hull, "Options, Futures, and Other Derivatives": S=42, K=40, r=10%, sigma=20%, T=0.5
    assert bs.price(42, 40, 0.5, 0.2, 0.1, 0.0, "call") == pytest.approx(4.76, abs=5e-3)
    assert bs.price(42, 40, 0.5, 0.2, 0.1, 0.0, "put") == pytest.approx(0.81, abs=5e-3)


def test_put_call_parity_on_a_grid():
    S = np.linspace(20_000, 120_000, 41)[:, None, None]
    K = np.array([40_000, 60_000, 80_000])[None, :, None]
    T = np.array([1 / 365, 30 / 365, 1.0])[None, None, :]
    for r, q, sigma in [(0.0, 0.0, 0.6), (0.05, 0.02, 0.3), (-0.01, 0.04, 1.2)]:
        lhs = bs.price(S, K, T, sigma, r, q, "call") - bs.price(S, K, T, sigma, r, q, "put")
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        np.testing.assert_allclose(lhs, rhs, rtol=0, atol=1e-8 * 60_000)


@pytest.mark.parametrize("S, K, T, sigma, r, q, kind", CASES)
def test_greeks_match_finite_differences(S, K, T, sigma, r, q, kind):
    g = bs.greeks(S, K, T, sigma, r, q, kind)
    p = lambda **kw: bs.price(**{**dict(S=S, K=K, T=T, sigma=sigma, r=r, q=q, kind=kind), **kw})  # noqa: E731
    hs, hv, ht = 1e-4 * S, 1e-5, 1e-6
    fd_delta = (p(S=S + hs) - p(S=S - hs)) / (2 * hs)
    hg = 1e-3 * S
    fd_gamma = (p(S=S + hg) - 2 * p() + p(S=S - hg)) / hg**2
    fd_vega = (p(sigma=sigma + hv) - p(sigma=sigma - hv)) / (2 * hv)
    fd_theta = -(p(T=T + ht) - p(T=T - ht)) / (2 * ht)
    assert g.price == pytest.approx(p(), rel=1e-12)
    assert g.delta == pytest.approx(fd_delta, rel=1e-6, abs=1e-9)
    assert g.gamma == pytest.approx(fd_gamma, rel=1e-4)
    assert g.vega == pytest.approx(fd_vega, rel=1e-6)
    assert g.theta == pytest.approx(fd_theta, rel=1e-5)
    # the single-Greek functions agree with greeks()
    assert bs.delta(S, K, T, sigma, r, q, kind) == pytest.approx(g.delta, rel=1e-14)
    assert bs.gamma(S, K, T, sigma, r, q) == pytest.approx(g.gamma, rel=1e-14)
    assert bs.vega(S, K, T, sigma, r, q) == pytest.approx(g.vega, rel=1e-14)
    assert bs.theta(S, K, T, sigma, r, q, kind) == pytest.approx(g.theta, rel=1e-14)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_satisfy_the_black_scholes_pde(kind):
    S = np.linspace(30_000, 100_000, 57)[:, None]
    T = np.array([0.5 / 365, 7 / 365, 30 / 365, 0.5])[None, :]
    sigma, r, q, K = 0.6, 0.04, 0.015, 60_000
    g = bs.greeks(S, K, T, sigma, r, q, kind)
    pde = g.theta + (r - q) * S * g.delta + 0.5 * sigma**2 * S**2 * g.gamma - r * g.price
    np.testing.assert_allclose(pde, 0.0, atol=1e-9 * K)


@pytest.mark.filterwarnings("error")
def test_zero_time_to_expiry_is_intrinsic():
    S = np.array([50_000.0, 60_000.0, 70_000.0])
    K, r, q = 60_000.0, 0.05, 0.02
    call = bs.greeks(S, K, 0.0, 0.6, r, q, "call")
    put = bs.greeks(S, K, 0.0, 0.6, r, q, "put")
    np.testing.assert_allclose(call.price, [0.0, 0.0, 10_000.0])
    np.testing.assert_allclose(put.price, [10_000.0, 0.0, 0.0])
    np.testing.assert_allclose(call.delta, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(put.delta, [-1.0, -0.5, 0.0])
    for g in (call, put):
        np.testing.assert_array_equal(g.gamma, 0.0)
        np.testing.assert_array_equal(g.vega, 0.0)
        assert np.all(np.isfinite(g.theta))
    # theta at expiry is the carry of the intrinsic value: q S - r K for an ITM call
    assert call.theta[2] == pytest.approx(q * 70_000 - r * K)


@pytest.mark.filterwarnings("error")
def test_zero_volatility_is_discounted_forward_intrinsic():
    S = np.array([50_000.0, 70_000.0])
    K, T, r, q = 60_000.0, 0.25, 0.05, 0.02
    call = bs.greeks(S, K, T, 0.0, r, q, "call")
    put = bs.greeks(S, K, T, 0.0, r, q, "put")
    fwd_intr = S * np.exp(-q * T) - K * np.exp(-r * T)
    np.testing.assert_allclose(call.price, np.maximum(fwd_intr, 0.0))
    np.testing.assert_allclose(put.price, np.maximum(-fwd_intr, 0.0))
    np.testing.assert_allclose(call.delta, [0.0, np.exp(-q * T)])
    np.testing.assert_allclose(put.delta, [-np.exp(-q * T), 0.0])
    np.testing.assert_array_equal(call.gamma, 0.0)
    np.testing.assert_allclose(call.theta, [0.0, q * 70_000 * np.exp(-q * T) - r * K * np.exp(-r * T)])
    # the sigma -> 0 limit is continuous
    np.testing.assert_allclose(bs.price(S, K, T, 1e-9, r, q), call.price, atol=1e-6)


@pytest.mark.filterwarnings("error")
def test_mixed_degenerate_and_regular_inputs_vectorise():
    T = np.array([0.0, 1 / 365, 30 / 365])
    sigma = np.array([0.6, 0.0, 0.6])
    g = bs.greeks(60_000.0, 59_000.0, T, sigma)
    assert all(np.all(np.isfinite(x)) for x in g)
    assert g.price[0] == pytest.approx(1_000.0)
    assert g.price[2] == pytest.approx(bs.price(60_000.0, 59_000.0, 30 / 365, 0.6))


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        bs.price(100, 100, -0.1, 0.2)
    with pytest.raises(ValueError):
        bs.price(100, 100, 0.1, -0.2)
    with pytest.raises(ValueError):
        bs.price(100, 100, 0.1, 0.2, kind="straddle")


@pytest.mark.parametrize("kind", ["call", "put"])
def test_implied_vol_round_trip(kind):
    target = float(bs.price(60_000, 63_000, 30 / 365, 0.73, 0.02, 0.01, kind))
    assert bs.implied_vol(target, 60_000, 63_000, 30 / 365, 0.02, 0.01, kind) == pytest.approx(0.73, abs=1e-9)
