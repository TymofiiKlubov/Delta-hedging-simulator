"""Experiment 5 - where the Black-Scholes hedge breaks: jumps and stochastic volatility.

Three models with the same expected realised variance (sigma_i^2 per year), hedged with the
Black-Scholes delta at sigma_i on the same schedules:

* GBM: the hedging error is pure discretisation noise and falls like 1/sqrt(N);
* Merton jumps: a jump moves the spot before the hedge can follow, so the loss
  V(S J) - V(S) - Delta S (J - 1) is taken in full however often one rebalances;
* Heston: realised variance itself is random, so the vol-gap P&L
  (1/2) int Gamma S^2 (sigma_i^2 - v_t) dt does not average out with more rebalancing.

Hedging more often removes the GBM part of the risk and leaves a fat-tailed residual.
"""

from __future__ import annotations

import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from .. import plotting as plt_
from .. import theory as th
from ..config import Config
from ..hedging import Strategy, run_hedge
from ..paths import GBM, Heston, Merton
from ..stats import summarize
from .common import (Checks, ExperimentResult, Setup, Timer, common_grid, print_table,
                     simulate_chunks, write_outputs)

LABELS = {"gbm": "GBM", "merton": "Merton jumps", "heston": "Heston stochastic vol"}


def build_models(cfg: Config, sigma: float) -> dict:
    m, h = cfg.models.merton, cfg.models.heston
    var = sigma**2
    return {
        "gbm": GBM(sigma),
        "merton": Merton.matched(sigma, m.lam, m.jump_mean, m.jump_std),
        "heston": Heston(var if h.v0 is None else h.v0, h.kappa, var if h.theta is None else h.theta,
                         h.xi, h.rho),
    }


def run(cfg: Config, out_dir=None, verbose: bool = True) -> ExperimentResult:
    setup = Setup.from_config(cfg)
    steps = sorted({int(s) for s in cfg.models.steps})
    fine = common_grid(steps)
    n_paths = int(cfg.models.paths or cfg.simulation.paths)
    show = min(steps, key=lambda s: abs(s - cfg.models.show_steps))
    models = build_models(cfg, setup.sigma_i)

    pnl: dict[str, dict[int, np.ndarray]] = {}
    with Timer() as timer:
        for name, model in models.items():
            acc = {n: [] for n in steps}
            for paths in simulate_chunks(model, setup, fine, n_paths, cfg.simulation.seed):
                for n in steps:
                    sub = paths.subsample(fine // n)
                    acc[n].append(run_hedge(sub.S, sub.t, setup.option, setup.market,
                                            setup.sigma_i, Strategy(), setup.costs).pnl)
            pnl[name] = {n: np.concatenate(v) for n, v in acc.items()}

    rows = []
    for name in models:
        for n in steps:
            s = summarize(pnl[name][n])
            rows.append({"model": name, "rebalances": n, "interval": setup.interval(n),
                         "mean": s["mean"], "se_mean": s["se_mean"], "std": s["std"],
                         "se_std": s["se_std"], "skew": s["skew"],
                         "excess_kurtosis": s["excess_kurtosis"], "q01": s["q01"],
                         "es01": s["es01"], "q99": s["q99"]})
    checks, summary = _check(setup, rows, steps, show)
    summary.update(models={k: repr(v) for k, v in models.items()}, show_steps=show,
                   show_interval=setup.interval(show))
    fig = _plot(setup, rows, pnl, steps, show)
    result = ExperimentResult("models", rows, summary, checks, {"models": fig}, seconds=timer.seconds)
    write_outputs(result, setup, out_dir)
    if verbose:
        _print(setup, result, n_paths)
    return result


def _check(setup: Setup, rows: list[dict], steps: list[int], show: int) -> tuple[Checks, dict]:
    checks = Checks()
    get = lambda m, n: next(r for r in rows if r["model"] == m and r["rebalances"] == n)  # noqa: E731
    gbm_large = [get("gbm", n) for n in steps if n >= 30]
    if len(gbm_large) >= 2:
        slope = float(np.polyfit(np.log([r["rebalances"] for r in gbm_large]),
                                 np.log([r["std"] for r in gbm_large]), 1)[0])
        checks.add("GBM: std ~ N^-1/2", abs(slope + 0.5) < 0.05, f"fitted slope {slope:.3f} for N >= 30")
        dk = [r["std"] / float(th.derman_kamal_std(r["rebalances"], setup.S0, setup.K, setup.T,
                                                   setup.sigma_i, position=setup.option.position))
              for r in gbm_large]
        worst = max(dk, key=lambda x: abs(x - 1))
        checks.add("GBM: std vs Derman-Kamal (N >= 30)", abs(worst - 1) < 0.05,
                   f"sim/theory within {abs(worst - 1):.1%}")
    summary: dict = {"plateau": {}}
    if len(steps) >= 2:
        hi, prev = steps[-1], steps[-2]
        expected = (prev / hi) ** 0.5
        for m in ("merton", "heston"):
            ratio = get(m, hi)["std"] / get(m, prev)["std"]
            summary["plateau"][m] = ratio
            checks.add(f"{LABELS[m]}: risk plateaus as hedging gets finer", ratio > 0.8,
                       f"std falls only to {ratio:.0%} from N={prev} to N={hi} "
                       f"(GBM: {get('gbm', hi)['std'] / get('gbm', prev)['std']:.0%}, "
                       f"theory {expected:.0%})")
    g = get("gbm", show)
    for m in ("merton", "heston"):
        r = get(m, show)
        checks.add(f"{LABELS[m]}: fatter left tail than GBM at {setup.interval(show)} hedging",
                   r["es01"] < 2 * g["es01"],
                   f"1% expected shortfall {r['es01']:,.0f} vs GBM {g['es01']:,.0f}; "
                   f"excess kurtosis {r['excess_kurtosis']:.1f} vs {g['excess_kurtosis']:.1f}")
    return checks, summary


def _plot(setup: Setup, rows: list[dict], pnl: dict, steps: list[int], show: int):
    styles = {"gbm": (plt_.SERIES[0], "o"), "merton": (plt_.SERIES[1], "s"),
              "heston": (plt_.SERIES[2], "^")}
    n = np.array(steps, dtype=float)
    grid = np.geomspace(n.min(), n.max(), 100)
    with plt_.style():
        fig, axes = plt_.figure(1, 2, width=14, height=5.0)
        ax1, ax2 = axes.flat
        dk = th.derman_kamal_std(grid, setup.S0, setup.K, setup.T, setup.sigma_i,
                                 position=setup.option.position)
        ax1.plot(grid, dk, color=plt_.INK, lw=1.5, label="Derman-Kamal (GBM theory)")
        for m, (color, marker) in styles.items():
            rs = [r for r in rows if r["model"] == m]
            ax1.errorbar([r["rebalances"] for r in rs], [r["std"] for r in rs],
                         yerr=[2 * r["se_std"] for r in rs], fmt=marker + "-", color=color,
                         ecolor=color, elinewidth=1, lw=1.2, label=LABELS[m], zorder=3)
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xticks(steps)
        ax1.set_xticklabels([f"{s:,}\n{setup.interval(s)}" for s in steps])
        ax1.xaxis.set_minor_formatter(NullFormatter())
        ax1.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax1.yaxis.set_minor_formatter(NullFormatter())
        plt_.money_axis(ax1)
        ax1.set_xlabel("rebalances over the option's life, N")
        ax1.set_ylabel("std of hedged P&L")
        ax1.set_title("Finer hedging removes diffusion risk, not jump or vol-of-vol risk")
        ax1.legend(loc="lower left")

        data = {m: pnl[m][show] for m in styles}
        lo = min(np.quantile(x, 0.0005) for x in data.values())
        hi = max(np.quantile(x, 0.9995) for x in data.values())
        bins = np.linspace(lo, hi, 120)
        for m, (color, _) in styles.items():
            x = data[m]
            s = summarize(x)
            ax2.hist(x, bins=bins, density=True, histtype="step", color=color, lw=1.5,
                     label=f"{LABELS[m]}: std {plt_.compact(s['std'])}, "
                           f"1% ES {plt_.compact(s['es01'])}")
        ax2.set_yscale("log")
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda *_: ""))
        ax2.yaxis.set_minor_formatter(NullFormatter())
        ax2.axvline(0.0, color=plt_.AXIS, lw=0.8)
        plt_.money_axis(ax2, "x")
        ax2.set_xlabel(f"hedged P&L at expiry, rebalanced every {setup.interval(show)}")
        ax2.set_ylabel("density (log scale)")
        ax2.set_title("Same expected variance, very different tails")
        ax2.legend(loc="upper left")
        plt_.headline(fig, "Jumps and stochastic volatility leave fat-tailed risk that "
                           "more frequent hedging cannot remove")
    return fig


def _print(setup: Setup, result: ExperimentResult, n_paths: int) -> None:
    print(f"\n=== Experiment 5: jumps and stochastic vol ({n_paths:,} paths per model, "
          f"{result.seconds:.1f}s) ===")
    print(setup.describe())
    for k, v in result.summary["models"].items():
        print(f"  {LABELS[k]}: {v}")
    print(print_table(result.rows, [
        ("model", "model", "s"), ("rebalances", "N", "d"), ("interval", "every", "s"),
        ("mean", "mean", ",.1f"), ("se_mean", "se", ",.1f"), ("std", "std", ",.1f"),
        ("skew", "skew", ".2f"), ("excess_kurtosis", "ex.kurt", ".1f"), ("q01", "q01", ",.0f"),
        ("es01", "ES1%", ",.0f"),
    ]))
    print(result.checks.report())
    for f in result.files:
        print(f"  wrote {f}")
