"""Experiment 1 - hedge frequency: the risk falls like 1/sqrt(N), the expected P&L does not move.

The same GBM paths are hedged at every frequency (common random numbers): each schedule
observes the paths on its own sub-grid of one fine grid.
"""

from __future__ import annotations

import math

import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from .. import plotting as plt_
from .. import theory as th
from ..config import Config
from ..hedging import Strategy, run_hedge
from ..paths import GBM
from ..stats import summarize
from .common import (Checks, ExperimentResult, Setup, Timer, common_grid, print_table,
                     simulate_chunks, write_outputs)

KEY_INTERVALS = {1: "monthly", 4: "~weekly", 30: "daily", 720: "hourly"}


def run(cfg: Config, out_dir=None, verbose: bool = True) -> ExperimentResult:
    setup = Setup.from_config(cfg)
    steps = sorted({int(s) for s in cfg.frequency.steps})
    n_paths = int(cfg.frequency.paths or cfg.simulation.paths)
    fine = common_grid(steps)
    strategy = Strategy(hedge_vol=setup.hedge_vol)

    with Timer() as timer:
        chunks: dict[int, list[np.ndarray]] = {n: [] for n in steps}
        for paths in simulate_chunks(GBM(setup.sigma_r), setup, fine, n_paths, cfg.simulation.seed):
            for n in steps:
                sub = paths.subsample(fine // n)
                res = run_hedge(sub.S, sub.t, setup.option, setup.market, setup.sigma_i,
                                strategy, setup.costs)
                chunks[n].append(res.pnl)
        pnl = {n: np.concatenate(v) for n, v in chunks.items()}

    rows = _table(setup, steps, pnl)
    checks, summary = _check(setup, rows)
    result = ExperimentResult("frequency", rows, summary, checks,
                              {"frequency": _plot(setup, rows, pnl)}, seconds=timer.seconds)
    write_outputs(result, setup, out_dir)
    if verbose:
        _print(setup, result, n_paths)
    return result


def _table(setup: Setup, steps: list[int], pnl: dict[int, np.ndarray]) -> list[dict]:
    S0, K, T, pos = setup.S0, setup.K, setup.T, setup.option.position
    kw = setup.theory_kwargs
    hedge = "implied" if setup.hedge_vol is None else "realised"
    hedge_vol = setup.sigma_i if setup.hedge_vol is None else setup.hedge_vol
    gap = th.vol_gap_mean(S0, K, T, setup.sigma_i, setup.sigma_r, kind=setup.option.kind,
                          position=pos, hedge=hedge, **kw)
    rows = []
    for n in steps:
        s = summarize(pnl[n])
        cost = 0.0
        if not setup.costs.is_zero:
            cost = float(th.expected_hedge_cost(n, S0, K, T, setup.sigma_r, hedge_vol,
                                                setup.costs.bps, setup.costs.fixed,
                                                kind=setup.option.kind, position=pos, **kw))
        dk = float(th.derman_kamal_std(n, S0, K, T, setup.sigma_r, setup.market.r,
                                       setup.market.q, pos))
        lo = float(th.pnl_std(n, S0, K, T, setup.sigma_i, setup.sigma_r, hedge_vol,
                              setup.costs.bps, position=pos, **kw))
        rows.append({
            "rebalances": n, "interval": setup.interval(n),
            "mean": s["mean"], "se_mean": s["se_mean"], "theory_mean": gap - cost,
            "std": s["std"], "se_std": s["se_std"], "std_derman_kamal": dk, "std_leading_order": lo,
            "std_ratio_dk": s["std"] / dk, "skew": s["skew"], "excess_kurtosis": s["excess_kurtosis"],
            "q01": s["q01"], "q99": s["q99"],
        })
    return rows


def _check(setup: Setup, rows: list[dict]) -> tuple[Checks, dict]:
    checks = Checks()
    atm = abs(math.log(setup.K / setup.S0)) < 0.02
    ref_key, ref_name = ("std_derman_kamal", "Derman-Kamal") if atm else ("std_leading_order", "leading-order")

    # Expected P&L: exact in expectation (frictionless, mu = r - q) for any N.
    slack = lambda r: 0.05 * abs(r["theory_mean"]) if not setup.costs.is_zero else 0.0  # noqa: E731
    bad = [r for r in rows if abs(r["mean"] - r["theory_mean"]) > 3 * r["se_mean"] + slack(r)]
    checks.add("mean P&L vs theory",
               not bad,
               "within 3 SE at every N" if not bad else
               "outside 3 SE at N = " + ", ".join(f"{r['rebalances']} ({r['mean']:+.1f} vs "
                                                   f"{r['theory_mean']:+.1f})" for r in bad))

    matched = math.isclose(setup.sigma_r, setup.sigma_i) and setup.hedge_vol is None
    summary: dict = {"premium": setup.premium, "reference_formula": ref_name}
    if matched:
        large = [r for r in rows if r["rebalances"] >= 30]
        small = [r for r in rows if r["rebalances"] < 30]
        ratio = lambda r: r["std"] / r[ref_key]  # noqa: E731
        worst = max(large, key=lambda r: abs(ratio(r) - 1), default=None)
        if worst is not None:
            checks.add(f"std vs {ref_name} (N >= 30)", abs(ratio(worst) - 1) < 0.05,
                       f"sim/theory within {abs(ratio(worst) - 1):.1%} (worst at N={worst['rebalances']})")
        if small:
            off = [r for r in small if abs(ratio(r) - 1) > 0.05]
            checks.add(f"std vs {ref_name} (N < 30)", not off,
                       "within 5%" if not off else
                       "sim/theory = " + ", ".join(f"{ratio(r):.2f} at N={r['rebalances']}" for r in off)
                       + " - the formula is leading order in 1/N and overstates the std "
                         "when the option is hedged only a few times")
        fit = [r for r in rows if r["rebalances"] >= 30]
        if len(fit) >= 2:
            x = np.log([r["rebalances"] for r in fit])
            y = np.log([r["std"] for r in fit])
            slope = float(np.polyfit(x, y, 1)[0])
            summary["slope_log_std_vs_log_n"] = slope
            checks.add("std ~ N^-1/2", abs(slope + 0.5) < 0.03, f"fitted slope {slope:.3f} for N >= 30")
    summary["std_ratio_dk"] = {r["rebalances"]: r["std_ratio_dk"] for r in rows}
    return checks, summary


def _plot(setup: Setup, rows: list[dict], pnl: dict[int, np.ndarray]):
    n = np.array([r["rebalances"] for r in rows], dtype=float)
    std = np.array([r["std"] for r in rows])
    se_std = np.array([r["se_std"] for r in rows])
    mean = np.array([r["mean"] for r in rows])
    se_mean = np.array([r["se_mean"] for r in rows])
    grid = np.geomspace(n.min(), n.max(), 200)
    S0, K, T, pos = setup.S0, setup.K, setup.T, setup.option.position
    blue, orange = plt_.SERIES[0], plt_.SERIES[1]

    with plt_.style():
        fig, axes = plt_.figure(1, 3, width=15, height=4.6)
        ax1, ax2, ax3 = axes.flat

        # (a) std vs N, log-log, with the Derman-Kamal line
        dk = th.derman_kamal_std(grid, S0, K, T, setup.sigma_r, setup.market.r, setup.market.q, pos)
        ax1.plot(grid, dk, color=plt_.INK, lw=1.5, label="Derman-Kamal  √(π/4)·vega·σ/√N")
        if abs(math.log(K / S0)) >= 0.02:
            lo = th.discretisation_std(grid, S0, K, T, setup.sigma_r, position=pos, **setup.theory_kwargs)
            ax1.plot(grid, lo, color=plt_.MUTED, lw=1.5, label="leading order (any strike)")
        ax1.errorbar(n, std, yerr=2 * se_std, fmt="o", color=blue, ecolor=blue, elinewidth=1,
                     label="simulated std (±2 SE)", zorder=3)
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel("rebalances over the option's life, N")
        ax1.set_ylabel("std of hedged P&L")
        plt_.money_axis(ax1)
        ax1.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax1.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
        ax1.set_title("Risk falls like 1/√N")
        ax1.legend(loc="lower left")
        _interval_axis(ax1, setup, n)

        # (b) mean P&L vs N
        theory_mean = np.array([r["theory_mean"] for r in rows])
        ax2.axhline(0.0, color=plt_.AXIS, lw=0.8)
        ax2.plot(n, theory_mean, color=plt_.INK, lw=1.5, label="theory")
        ax2.errorbar(n, mean, yerr=2 * se_mean, fmt="o", color=blue, ecolor=blue, elinewidth=1,
                     label="simulated mean (±2 SE)", zorder=3)
        ax2.set_xscale("log")
        lim = max(np.max(np.abs(mean) + 2 * se_mean), np.max(np.abs(theory_mean))) * 1.6
        ax2.set_ylim(-max(lim, 0.02 * setup.premium), max(lim, 0.02 * setup.premium))
        ax2.set_xlabel("rebalances over the option's life, N")
        ax2.set_ylabel("mean hedged P&L")
        plt_.money_axis(ax2)
        ax2.set_title("Expected P&L does not depend on N")
        ax2.legend(loc="upper right")
        _interval_axis(ax2, setup, n)

        # (c) distributions at two frequencies
        n_lo, n_hi = _pick_pair(list(pnl))
        x_lo, x_hi = pnl[n_lo], pnl[n_hi]
        lim = np.quantile(np.abs(x_lo), 0.995)
        bins = np.linspace(-lim, lim, 81)
        for x, nn, color in ((x_lo, n_lo, orange), (x_hi, n_hi, blue)):
            label = f"every {setup.interval(nn)} (N={nn}): std {plt_.compact(x.std())}"
            ax3.hist(x, bins=bins, density=True, histtype="stepfilled", color=color, alpha=0.12)
            ax3.hist(x, bins=bins, density=True, histtype="step", color=color, lw=1.6, label=label)
        ax3.axvline(0.0, color=plt_.AXIS, lw=0.8)
        ax3.set_yticks([])
        ax3.set_xlabel("hedged P&L at expiry")
        plt_.money_axis(ax3, "x")
        ax3.set_title("Same mean, much tighter spread")
        ax3.legend(loc="upper left")

        plt_.headline(fig, "Hedge frequency changes the risk of a delta-hedged option, not its expected P&L")
    return fig


def _pick_pair(ns: list[int]) -> tuple[int, int]:
    ns = sorted(ns)
    lo = 30 if 30 in ns else ns[min(1, len(ns) - 1)]
    hi = 720 if 720 in ns else ns[-1]
    return lo, hi


def _interval_axis(ax, setup: Setup, n: np.ndarray) -> None:
    """Label key rebalancing counts with their interval (monthly ... every 5 minutes)."""
    marks = [m for m in (1, 4, 30, 720, 2880, 8640, 43200) if n.min() <= m <= n.max()]
    names = {**KEY_INTERVALS, 2880: "15 min", 8640: "5 min", 43200: "1 min"}
    ax.set_xticks(marks)
    ax.set_xticklabels([f"{m:,}\n{names.get(m, setup.interval(m))}" for m in marks])
    ax.xaxis.set_minor_formatter(NullFormatter())


def _print(setup: Setup, result: ExperimentResult, n_paths: int) -> None:
    print(f"\n=== Experiment 1: hedge frequency ({n_paths:,} paths, {result.seconds:.1f}s) ===")
    print(setup.describe())
    print(f"sigma_r={setup.sigma_r:.0%}, hedge with {setup.cfg.vol.hedge_with} vol, premium {setup.premium:,.2f}")
    print(print_table(result.rows, [
        ("rebalances", "N", "d"), ("interval", "every", "s"), ("mean", "mean", ",.2f"),
        ("se_mean", "se", ",.2f"), ("std", "std", ",.1f"), ("std_derman_kamal", "DK std", ",.1f"),
        ("std_ratio_dk", "sim/DK", ".3f"), ("excess_kurtosis", "ex.kurt", ".2f"),
    ]))
    print(result.checks.report())
    for f in result.files:
        print(f"  wrote {f}")
