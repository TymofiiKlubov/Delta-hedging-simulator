"""Experiment 3 - transaction costs: hedging more often cuts risk but costs more.

With a proportional cost k, hedging N times costs ~ k sqrt(N) (each trade is ~ Gamma dS, of
size ~ sqrt(dt)) while the hedging-error variance falls like 1/N, so the mean-variance
objective

    J(N) = -E[PnL] + kappa * Var[PnL] / premium

is U-shaped with a minimum near N* ~ (2 kappa b / (a premium))^(2/3) for cost a sqrt(N) and
variance b / N. Band hedging (trade only when the delta drifts more than a band away from the
hedge) spends the trading budget where gamma is, and beats every fixed schedule.
"""

from __future__ import annotations

import math

import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter

from .. import bs
from .. import plotting as plt_
from .. import theory as th
from ..config import Config
from ..hedging import Costs, HedgeResult, Strategy, run_hedge
from ..paths import GBM
from ..stats import summarize
from .common import (Checks, ExperimentResult, Setup, Timer, common_grid, print_table,
                     simulate_chunks, write_outputs)


def run(cfg: Config, out_dir=None, verbose: bool = True) -> ExperimentResult:
    setup = Setup.from_config(cfg)
    ct = cfg.costs_tradeoff
    costs = Costs(ct.bps, ct.fixed)
    steps = sorted({int(s) for s in ct.steps})
    monitor = int(ct.monitor_steps)
    bands = sorted({float(b) for b in ct.bands})
    modes = list(dict.fromkeys(ct.band_modes))
    n_paths = int(ct.paths or cfg.simulation.paths)
    fine = common_grid(steps + [monitor])
    hedge_vol = setup.sigma_i if setup.hedge_vol is None else setup.hedge_vol
    r, q = setup.market.r, setup.market.q

    acc: dict[tuple, list] = {("fixed", n): [] for n in steps}
    acc.update({(mode, b): [] for mode in modes for b in bands})
    with Timer() as timer:
        for paths in simulate_chunks(GBM(setup.sigma_r), setup, fine, n_paths, cfg.simulation.seed):
            for n in steps:
                sub = paths.subsample(fine // n)
                acc[("fixed", n)].append(run_hedge(sub.S, sub.t, setup.option, setup.market,
                                                   setup.sigma_i, Strategy(setup.hedge_vol), costs))
            sub = paths.subsample(fine // monitor)
            delta = bs.delta(sub.S[:, :-1], setup.K, setup.T - sub.t[:-1], hedge_vol, r, q,
                             setup.option.kind)
            for mode in modes:
                for b in bands:
                    strategy = Strategy(setup.hedge_vol, band=b, band_mode=mode)
                    acc[(mode, b)].append(run_hedge(sub.S, sub.t, setup.option, setup.market,
                                                    setup.sigma_i, strategy, costs,
                                                    target_delta=delta))

    premium = setup.premium
    kappa = float(ct.risk_aversion)
    rows = [_row(setup, key, res, kappa, premium, monitor) for key, res in acc.items()]
    tradeoff = th.cost_tradeoff(steps, setup.S0, setup.K, setup.T, setup.sigma_i, setup.sigma_r,
                                ct.bps, ct.fixed, kappa, kind=setup.option.kind,
                                position=setup.option.position, **setup.theory_kwargs)
    for row in rows:
        if row["strategy"] == "fixed":
            i = steps.index(row["rebalances"])
            row.update(theory_cost=float(tradeoff.cost[i]), theory_std=float(tradeoff.std[i]),
                       theory_objective=float(tradeoff.objective[i]))
        else:
            row.update(theory_cost=float("nan"), theory_std=float("nan"),
                       theory_objective=float("nan"))
    optimum = th.optimal_rebalances(setup.S0, setup.K, setup.T, setup.sigma_i, setup.sigma_r,
                                    ct.bps, ct.fixed, kappa, kind=setup.option.kind,
                                    position=setup.option.position, **setup.theory_kwargs)
    checks, summary = _check(setup, rows, optimum, steps)
    summary.update(premium=premium, bps=ct.bps, fixed=ct.fixed, risk_aversion=kappa,
                   monitor_interval=setup.interval(monitor))
    fig = _plot(setup, rows, optimum, steps, kappa, premium, monitor)
    result = ExperimentResult("costs", rows, summary, checks, {"costs": fig}, seconds=timer.seconds)
    write_outputs(result, setup, out_dir)
    if verbose:
        _print(setup, result, n_paths)
    return result


def _row(setup: Setup, key: tuple, results: list, kappa: float, premium: float,
         monitor: int) -> dict:
    res = HedgeResult.concat(results)
    s = summarize(res.pnl)
    var = s["std"] ** 2
    objective = -s["mean"] + kappa * var / premium
    se_obj = math.sqrt(s["se_mean"] ** 2 + (kappa / premium * 2 * s["std"] * s["se_std"]) ** 2)
    strategy, param = key
    fixed = strategy == "fixed"
    return {
        "strategy": strategy,
        "rebalances": int(param) if fixed else monitor,
        "band": float("nan") if fixed else float(param),
        "label": f"every {setup.interval(int(param))}" if fixed else f"band {param:g} ({strategy})",
        "mean": s["mean"], "se_mean": s["se_mean"], "std": s["std"], "se_std": s["se_std"],
        "cost": float(-res.costs.mean()), "trades": float(res.n_trades.mean()),
        "objective": objective, "se_objective": se_obj,
    }


def _check(setup: Setup, rows: list[dict], optimum, steps: list[int]) -> tuple[Checks, dict]:
    checks = Checks()
    fixed = [r for r in rows if r["strategy"] == "fixed"]
    bands = [r for r in rows if r["strategy"] != "fixed"]

    worst = max(fixed, key=lambda r: abs(r["cost"] / r["theory_cost"] - 1))
    err = abs(worst["cost"] / worst["theory_cost"] - 1)
    checks.add("expected cost vs Leland-style theory", err < 0.05,
               f"sim/theory within {err:.1%} at every N (worst at N={worst['rebalances']})")
    large = [r for r in fixed if r["rebalances"] >= 30]
    if large:
        worst = max(large, key=lambda r: abs(r["std"] / r["theory_std"] - 1))
        err = abs(worst["std"] / worst["theory_std"] - 1)
        checks.add("std vs leading-order theory (N >= 30)", err < 0.05,
                   f"sim/theory within {err:.1%} (worst at N={worst['rebalances']})")

    best_fixed = min(fixed, key=lambda r: r["objective"])
    n_sim, n_th = best_fixed["rebalances"], optimum.best
    i = steps.index(n_sim)
    lo = steps[i - 1] if i > 0 else 1
    hi = steps[i + 1] if i + 1 < len(steps) else n_sim
    checks.add("optimal frequency: simulation vs theory", lo <= n_th <= hi,
               f"simulated optimum N={n_sim} (every {setup.interval(n_sim)}), theory "
               f"N*={n_th} (every {setup.interval(n_th)}); grid neighbours [{lo}, {hi}]")
    j_th = float(optimum.objective.min())
    err = abs(best_fixed["objective"] / j_th - 1)
    checks.add("minimum objective: simulation vs theory", err < 0.1,
               f"J_sim = {best_fixed['objective']:.1f} vs J_theory = {j_th:.1f} ({err:.1%})")
    summary = {"best_fixed": best_fixed, "theory_optimum": {"rebalances": n_th,
               "interval": setup.interval(n_th), "objective": j_th}}
    if bands:
        best_band = min(bands, key=lambda r: r["objective"])
        gain = 1 - best_band["objective"] / best_fixed["objective"]
        checks.add("band hedging beats the best fixed schedule", gain > 0,
                   f"best band {best_band['label']}: J = {best_band['objective']:.1f} vs "
                   f"{best_fixed['objective']:.1f} for every {setup.interval(n_sim)} "
                   f"({gain:.0%} lower)")
        summary["best_band"] = best_band
        summary["band_improvement"] = gain
    return checks, summary


def _plot(setup: Setup, rows: list[dict], optimum, steps, kappa, premium, monitor):
    fixed = [r for r in rows if r["strategy"] == "fixed"]
    blue, orange, aqua = plt_.SERIES[:3]
    ct = setup.cfg.costs_tradeoff
    n = np.array([r["rebalances"] for r in fixed], dtype=float)
    grid = np.unique(np.geomspace(n.min(), n.max(), 120).astype(int))
    curve = th.cost_tradeoff(grid, setup.S0, setup.K, setup.T, setup.sigma_i, setup.sigma_r,
                             ct.bps, ct.fixed, kappa, kind=setup.option.kind,
                             position=setup.option.position, **setup.theory_kwargs)
    risk = kappa * curve.std**2 / premium
    bands = {mode: sorted([r for r in rows if r["strategy"] == mode], key=lambda r: r["band"])
             for mode in dict.fromkeys(r["strategy"] for r in rows if r["strategy"] != "fixed")}
    band_style = {"target": (orange, "s", "band hedge, trade back to delta"),
                  "edge": (aqua, "^", "band hedge, trade only to the band edge")}
    best_band = min((r for rs in bands.values() for r in rs), key=lambda r: r["objective"],
                    default=None)

    with plt_.style():
        fig, axes = plt_.figure(1, 2, width=14, height=5.0)
        ax1, ax2 = axes.flat

        # (a) the U-shaped objective and its two components
        ax1.plot(grid, curve.cost, color=plt_.INK_2, lw=1.0)
        ax1.plot(grid, risk, color=plt_.MUTED, lw=1.0)
        # label the components where they are well apart: ~10% and ~50% along the N range
        i, j = int(0.1 * (len(grid) - 1)), int(0.5 * (len(grid) - 1))
        plt_.end_label(ax1, grid[i], curve.cost[i], "expected cost", dx=0, dy=9)
        plt_.end_label(ax1, grid[j], risk[j], "risk penalty \u03ba\u00b7Var / premium", dx=4, dy=-11)
        ax1.plot(grid, curve.objective, color=plt_.INK, lw=1.5,
                 label="theory  J = cost + \u03ba\u00b7Var / premium")
        ax1.errorbar(n, [r["objective"] for r in fixed], yerr=[2 * r["se_objective"] for r in fixed],
                     fmt="o", color=blue, ecolor=blue, elinewidth=1,
                     label="simulated J, fixed schedule (\u00b12 SE)", zorder=3)
        if best_band is not None:
            ax1.axhline(best_band["objective"], color=band_style[best_band["strategy"]][0], lw=1.2,
                        label=f"best band hedge (\u00b1{best_band['band']:g} delta, "
                              f"checked every {setup.interval(monitor)})")
        ax1.axvline(optimum.best, color=plt_.AXIS, lw=0.8)
        ax1.annotate(f"optimum N* = {optimum.best}\n(every {setup.interval(optimum.best)})",
                     (optimum.best, curve.objective.min()), xytext=(6, 34),
                     textcoords="offset points", fontsize=8.5, color=plt_.INK_2)
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        lo = min(np.min(curve.cost), np.min(risk[risk > 0])) * 0.7
        ax1.set_ylim(lo, np.max(curve.objective) * 3)
        ax1.set_xlabel("rebalances over the option's life, N")
        ax1.set_ylabel("objective J (currency)")
        plt_.money_axis(ax1)
        ax1.yaxis.set_minor_formatter(NullFormatter())
        marks = [m for m in (1, 4, 30, 120, 720, 2880) if n.min() <= m <= n.max()]
        ax1.set_xticks(marks)
        ax1.set_xticklabels([f"{m:,}\n{setup.interval(m)}" for m in marks])
        ax1.xaxis.set_minor_formatter(NullFormatter())
        ax1.set_title(f"{ct.bps:g} bp costs: too little hedging is risky, too much is expensive")
        ax1.legend(loc="upper right")

        # (b) cost vs risk for every strategy: down and to the left is better
        ax2.plot([r["std"] for r in fixed], [r["cost"] for r in fixed], "o-", color=blue, lw=1.2,
                 label="fixed schedule")
        for mode, rs in bands.items():
            color, marker, label = band_style.get(mode, (plt_.SERIES[3], "D", mode))
            ax2.plot([r["std"] for r in rs], [r["cost"] for r in rs], marker + "-", color=color,
                     lw=1.2, label=label)
        best_fixed = min(fixed, key=lambda r: r["objective"])
        for r in fixed:
            if r["rebalances"] in (1, 30, 120, 2880):
                plt_.end_label(ax2, r["std"], r["cost"], f"every {setup.interval(r['rebalances'])}",
                               dx=6, dy=9)
        leader = dict(arrowstyle="-", color=plt_.MUTED, lw=0.8, shrinkA=2, shrinkB=4)
        ax2.annotate(f"best fixed schedule (every {setup.interval(best_fixed['rebalances'])})\n"
                     f"J = {best_fixed['objective']:.0f}", (best_fixed["std"], best_fixed["cost"]),
                     xytext=(40, 55), textcoords="offset points", fontsize=8.5,
                     color=plt_.INK_2, arrowprops=leader)
        if best_band is not None:
            ax2.annotate(f"best band (\u00b1{best_band['band']:g} delta)\nJ = "
                         f"{best_band['objective']:.0f}", (best_band["std"], best_band["cost"]),
                         xytext=(-60, -34), textcoords="offset points", fontsize=8.5, ha="right",
                         color=plt_.INK_2, arrowprops=leader)
        ax2.set_xscale("log")
        ax2.xaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax2.set_ylim(0, max(r["cost"] for r in rows) * 1.12)
        ax2.set_xlabel("std of hedged P&L (risk)")
        ax2.set_ylabel("expected transaction cost")
        plt_.money_axis(ax2)
        plt_.money_axis(ax2, "x")
        ax2.xaxis.set_minor_formatter(NullFormatter())
        ax2.set_title("Bands buy more risk reduction per unit of cost")
        ax2.legend(loc="upper right")
        plt_.headline(fig, "With transaction costs there is an optimal hedge frequency "
                           "- and band hedging beats any fixed schedule")
    return fig


def _print(setup: Setup, result: ExperimentResult, n_paths: int) -> None:
    s = result.summary
    print(f"\n=== Experiment 3: costs vs frequency ({n_paths:,} paths, {s['bps']:g} bp + "
          f"{s['fixed']:g} fixed, kappa={s['risk_aversion']:g}, {result.seconds:.1f}s) ===")
    print(setup.describe())
    print(print_table(result.rows, [
        ("label", "strategy", "s"), ("mean", "mean", ",.1f"), ("std", "std", ",.1f"),
        ("cost", "E[cost]", ",.1f"), ("theory_cost", "theory", ",.1f"), ("trades", "trades", ",.0f"),
        ("objective", "J", ",.1f"), ("theory_objective", "J theory", ",.1f"),
    ]))
    print(result.checks.report())
    for f in result.files:
        print(f"  wrote {f}")
