"""Experiment 4 - one path, step by step: where a delta-hedged option's P&L comes from.

One simulated path, with the option marked to model at the implied vol. Each step the short
collects theta, Gamma S^2 sigma_i^2 dt / 2, and pays gamma, Gamma dS^2 / 2. Whether the
total ends up positive depends on whether the moves the path realised were larger or smaller
than the implied vol priced in, weighted by the gamma held at the time.
"""

from __future__ import annotations

import math

import numpy as np

from .. import plotting as plt_
from ..config import Config
from ..hedging import Strategy, run_hedge
from ..paths import GBM
from .common import Checks, ExperimentResult, Setup, Timer, write_outputs


def run(cfg: Config, out_dir=None, verbose: bool = True) -> ExperimentResult:
    setup = Setup.from_config(cfg)
    n, every, seed = int(cfg.path.steps), int(cfg.path.every), int(cfg.path.seed)
    with Timer() as timer:
        paths = GBM(setup.sigma_r).simulate(setup.S0, setup.T, n, 1, np.random.default_rng(seed),
                                            mu=setup.mu)
        res = run_hedge(paths.S, paths.t, setup.option, setup.market, setup.sigma_i,
                        Strategy(setup.hedge_vol, every=every), setup.costs,
                        decompose=True, record=True)
    st = {k: (v[0] if v.ndim == 2 else v) for k, v in res.steps.items()}
    days = st["t"] * setup.cfg.simulation.days_per_year

    attribution = {k: float(v[0]) for k, v in res.attribution().items()}
    decomposition = {k: float(v[0]) for k, v in res.decomposition().items()}
    pnl = float(res.pnl[0])
    realised = float(paths.realised_vol()[0])
    summary = {
        "seed": seed, "steps": n, "rebalance_every": setup.interval(n // every if n % every == 0 else n),
        "realised_vol": realised, "implied_vol": setup.sigma_i, "final_spot": float(paths.S[0, -1]),
        "pnl": pnl, "attribution": attribution, "decomposition": decomposition,
        "trades": int(res.n_trades[0]),
    }
    checks = Checks()
    err = abs(sum(attribution.values()) - pnl)
    checks.add("attribution sums to total P&L", err < 1e-6 * setup.S0,
               "premium + payoff + hedge + financing + costs = total "
               f"({pnl:+,.2f}; error {err:.1e})")
    err = abs(sum(decomposition.values()) - pnl)
    checks.add("theta/gamma decomposition sums to total P&L", err < 1e-6 * setup.S0,
               f"error {err:.1e}")
    scale = abs(decomposition["theta"]) + abs(decomposition["gamma"])
    share = abs(decomposition["residual"]) / scale if scale else 0.0
    checks.add("theta + gamma explain the P&L (third-order residual < 10% of their size)",
               share < 0.1, f"residual {decomposition['residual']:+,.1f} = {share:.1%} of |theta| + |gamma|")

    rows = _rows(st, days)
    result = ExperimentResult("path", rows, summary, checks,
                              {"path": _plot(setup, st, days, summary, every)},
                              seconds=timer.seconds)
    write_outputs(result, setup, out_dir)
    if verbose:
        _print(setup, result)
    return result


def _rows(st: dict, days: np.ndarray) -> list[dict]:
    n = len(days) - 1
    cum = {k: np.concatenate([[0.0], np.cumsum(st[k])]) for k in
           ("theta", "gamma", "mismatch", "financing", "costs", "residual")}
    rows = []
    for k in range(n + 1):
        step = k - 1  # per-step quantities are attributed to the end of their step
        rows.append({
            "day": days[k], "spot": st["S"][k],
            "delta_implied": st["delta"][k] if k < n else float("nan"),
            "holding": st["holding"][k] if k < n else 0.0,
            "dollar_gamma": st["gamma_greek"][k] * st["S"][k] ** 2 if k < n else float("nan"),
            "option_value": st["value"][k], "cash": st["cash"][k], "mtm_pnl": st["mtm_pnl"][k],
            "theta_step": st["theta"][step] if k else 0.0,
            "gamma_step": st["gamma"][step] if k else 0.0,
            **{f"cum_{key}": float(v[k]) for key, v in cum.items()},
        })
    return rows


def _rolling_vol(S: np.ndarray, t: np.ndarray, window: int) -> np.ndarray:
    r2 = np.diff(np.log(S)) ** 2
    c = np.concatenate([[0.0], np.cumsum(r2)])
    k = np.arange(1, len(r2) + 1)
    lo = np.maximum(k - window, 0)
    return np.sqrt((c[k] - c[lo]) / (t[k] - t[lo]))


def _plot(setup: Setup, st: dict, days: np.ndarray, summary: dict, every: int):
    blue, orange = plt_.SERIES[0], plt_.SERIES[1]
    pos = setup.option.position
    S = st["S"]
    n = len(S) - 1
    per_day = n / setup.expiry_days
    window = max(2, int(round(per_day)))

    with plt_.style():
        fig = plt_.Figure(figsize=(12, 11.5), layout="constrained")
        axes = fig.subplots(4, 1, sharex=True, gridspec_kw={"height_ratios": [1.0, 0.9, 0.8, 1.4]})
        ax1, ax2, ax3, ax4 = axes

        ax1.axhline(setup.K, color=plt_.MUTED, lw=1.0)
        plt_.end_label(ax1, days[-1], setup.K, f"strike {plt_.compact(setup.K)}", dx=4)
        ax1.plot(days, S, color=plt_.INK, lw=1.4)
        plt_.money_axis(ax1)
        ax1.set_ylabel("spot")
        ax1.set_title(f"Spot path (seed {summary['seed']}): realised vol {summary['realised_vol']:.1%} "
                      f"vs {setup.sigma_i:.0%} implied")

        ax2.plot(days[:-1], st["delta"], color=blue, lw=1.4, label="Δ at implied vol")
        hedge = st["holding"] / abs(pos) if pos else st["holding"]
        ax2.step(days[:-1], hedge, where="post", color=orange, lw=1.2,
                 label=f"hedge held per option (rebalanced every {_interval(setup, n, every)})")
        ax2.set_ylabel("units of spot")
        ax2.set_title("The hedge: buy as the spot rises, sell as it falls")
        ax2.legend(loc="best")

        vol = _rolling_vol(S, st["t"], window)[window - 1:]  # from the first full window
        vdays = days[window:]
        ax3.axhline(setup.sigma_i, color=plt_.INK, lw=1.2)
        plt_.end_label(ax3, days[-1], setup.sigma_i, "implied", dx=4)
        ax3.plot(vdays, vol, color=blue, lw=1.2)
        ax3.fill_between(vdays, setup.sigma_i, vol, where=vol > setup.sigma_i, color=orange,
                         alpha=0.18, lw=0, label="realised above implied: the short bleeds")
        ax3.set_ylabel("vol")
        ax3.yaxis.set_major_formatter(plt_.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax3.set_title(f"Realised volatility, trailing {window * setup.expiry_days / n:.3g}-day window")
        ax3.legend(loc="upper left")

        cum = lambda key: np.concatenate([[0.0], np.cumsum(st[key])])  # noqa: E731
        other = cum("mismatch") + cum("financing") + cum("costs") + cum("residual")
        ax4.axhline(0.0, color=plt_.AXIS, lw=0.8)
        ax4.plot(days, cum("theta"), color=blue, label="theta collected  −Θ·dt")
        ax4.plot(days, cum("gamma"), color=orange, label="gamma paid  ½Γ·dS²")
        ax4.plot(days, other, color=plt_.MUTED, lw=1.2,
                 label="other: residual, hedge ≠ Δ, financing, costs")
        ax4.plot(days, st["mtm_pnl"], color=plt_.INK, lw=1.6, label="hedged P&L, marked at implied vol")
        for key, y in (("theta", cum("theta")[-1]), ("gamma", cum("gamma")[-1]),
                       ("total", st["mtm_pnl"][-1])):
            plt_.end_label(ax4, days[-1], y, f"{key} {plt_.compact(y)}", dx=5)
        plt_.money_axis(ax4)
        ax4.set_ylabel("cumulative P&L")
        ax4.set_xlabel("days")
        side = "short" if pos < 0 else "long"
        ax4.set_title(f"Cumulative P&L of the {side} option: theta earned vs gamma paid")
        ax4.legend(loc="upper left")
        ax4.set_xlim(days[0], days[-1] * 1.08)
        plt_.headline(fig, "One path: a short option collects theta and pays for every move "
                           "through gamma")
    return fig


def _interval(setup: Setup, n: int, every: int) -> str:
    return setup.interval(n / every) if every > 1 else setup.interval(n)


def _print(setup: Setup, result: ExperimentResult) -> None:
    s = result.summary
    print(f"\n=== Experiment 4: single-path walkthrough (seed {s['seed']}, {s['steps']} steps) ===")
    print(setup.describe())
    print(f"realised vol {s['realised_vol']:.1%} vs implied {s['implied_vol']:.0%}; "
          f"final spot {s['final_spot']:,.0f}; {s['trades']} trades")
    print("attribution:   " + ", ".join(f"{k} {v:+,.1f}" for k, v in s["attribution"].items())
          + f"  = {s['pnl']:+,.1f}")
    print("decomposition: " + ", ".join(f"{k} {v:+,.1f}" for k, v in s["decomposition"].items()))
    print(result.checks.report())
    for f in result.files:
        print(f"  wrote {f}")
