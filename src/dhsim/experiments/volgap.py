"""Experiment 2 - realised vs implied volatility: a delta-hedged short option is a volatility bet.

The option is sold at BS(sigma_i) and the paths realise sigma_r. Two hedges run on the same
paths:

* delta at sigma_r replicates the option, so it locks in BS(sigma_i) - BS(sigma_r) on every path;
* delta at sigma_i (what a desk without a vol forecast does) earns the same on average but
  path by path: PnL = (1/2) int Gamma_i S^2 (sigma_i^2 - sigma_r^2) dt, larger when the path
  spends its time near the strike close to expiry, where dollar gamma is concentrated.

The same Brownian increments are reused for every sigma_r (common random numbers).
"""

from __future__ import annotations

import numpy as np

from .. import bs
from .. import plotting as plt_
from .. import theory as th
from ..config import Config
from ..hedging import Strategy, run_hedge
from ..paths import GBM
from ..stats import summarize
from .common import (Checks, ExperimentResult, Setup, Timer, print_table, simulate_chunks,
                     write_outputs, z_threshold)


def run(cfg: Config, out_dir=None, verbose: bool = True) -> ExperimentResult:
    setup = Setup.from_config(cfg)
    sweep = sorted({round(float(s), 10) for s in cfg.volgap.sigma_r})
    n = int(cfg.volgap.steps)
    n_paths = int(cfg.volgap.paths or cfg.simulation.paths)
    show = sorted({min(sweep, key=lambda s: abs(s - x)) for x in cfg.volgap.show})

    samples: dict[float, dict[str, np.ndarray]] = {}
    with Timer() as timer:
        for sigma_r in sweep:
            out = {"implied": [], "realised": [], "pathwise": [], "implied_frictionless": []}
            for paths in simulate_chunks(GBM(sigma_r), setup, n, n_paths, cfg.simulation.seed):
                for key, hv in (("implied", None), ("realised", sigma_r)):
                    res = run_hedge(paths.S, paths.t, setup.option, setup.market, setup.sigma_i,
                                    Strategy(hedge_vol=hv), setup.costs)
                    out[key].append(res.pnl)
                    if key == "implied":
                        out["implied_frictionless"].append(res.pnl - res.costs)
                out["pathwise"].append(_pathwise_theory(setup, paths, sigma_r))
            samples[sigma_r] = {k: np.concatenate(v) for k, v in out.items()}

    rows = [_row(setup, sigma_r, n, samples[sigma_r]) for sigma_r in sweep]
    checks = _check(setup, rows)
    figures = {"volgap": _plot_summary(setup, rows, n),
               "volgap_distributions": _plot_distributions(setup, samples, show)}
    summary = {"premium": setup.premium, "steps": n, "interval": setup.interval(n),
               "show": show}
    result = ExperimentResult("volgap", rows, summary, checks, figures, seconds=timer.seconds)
    write_outputs(result, setup, out_dir)
    if verbose:
        _print(setup, result, n_paths)
    return result


def _pathwise_theory(setup: Setup, paths, sigma_r: float) -> np.ndarray:
    """-position * (1/2)(sigma_i^2 - sigma_r^2) sum_k Gamma_i(t_k, S_k) S_k^2 dt e^{r(T - t_k)}."""
    t, S = paths.t, paths.S[:, :-1]
    r, q = setup.market.r, setup.market.q
    dollar_gamma = S**2 * bs.gamma(S, setup.K, setup.T - t[:-1], setup.sigma_i, r, q)
    weights = np.diff(t) * np.exp(r * (setup.T - t[:-1]))
    return -setup.option.position * 0.5 * (setup.sigma_i**2 - sigma_r**2) * (dollar_gamma @ weights)


def _row(setup: Setup, sigma_r: float, n: int, x: dict[str, np.ndarray]) -> dict:
    S0, K, T, pos, kind = setup.S0, setup.K, setup.T, setup.option.position, setup.option.kind
    kw = setup.theory_kwargs
    imp, rea = summarize(x["implied"]), summarize(x["realised"])
    pw = x["pathwise"]
    corr = float(np.corrcoef(pw, x["implied"])[0, 1]) if pw.std() > 0 else float("nan")
    total = lambda hv: float(th.pnl_std(n, S0, K, T, setup.sigma_i, sigma_r, hv,  # noqa: E731
                                        setup.costs.bps, position=pos, **kw))
    return {
        "sigma_r": sigma_r,
        "theory_lockin": th.vol_gap_mean(S0, K, T, setup.sigma_i, sigma_r, kind=kind, position=pos,
                                         hedge="realised", **kw),
        "theory_mean_implied": th.vol_gap_mean(S0, K, T, setup.sigma_i, sigma_r, kind=kind,
                                               position=pos, hedge="implied", **kw),
        "mean_implied": imp["mean"], "se_implied": imp["se_mean"], "std_implied": imp["std"],
        "q05_implied": imp["q05"], "q95_implied": imp["q95"],
        "mean_realised": rea["mean"], "se_realised": rea["se_mean"], "std_realised": rea["std"],
        "q05_realised": rea["q05"], "q95_realised": rea["q95"],
        "theory_std_implied": total(None),
        "theory_std_realised": total(sigma_r),
        "std_pathwise": float(pw.std(ddof=1)),
        "corr_pathwise": corr,
        "std_residual": float(np.std(x["implied_frictionless"] - pw, ddof=1)),
        "theory_std_residual": float(th.discretisation_std(n, S0, K, T, sigma_r, setup.sigma_i,
                                                           position=pos, **kw)),
    }


def _check(setup: Setup, rows: list[dict]) -> Checks:
    checks = Checks()
    for key, theory, label in (("implied", "theory_mean_implied", "implied-vol delta"),
                               ("realised", "theory_lockin", "realised-vol delta")):
        z = z_threshold(len(rows))
        worst = max(abs(r[f"mean_{key}"] - r[theory]) / r[f"se_{key}"] for r in rows)
        bad = [r for r in rows if abs(r[f"mean_{key}"] - r[theory]) > z * r[f"se_{key}"]]
        checks.add(f"mean P&L vs theory ({label})", not bad,
                   f"|z| < {z:.2f} for every sigma_r (largest {worst:.2f})" if not bad else
                   f"|z| > {z:.2f} at sigma_r = " + ", ".join(
                       f"{r['sigma_r']:.2f} ({r[f'mean_{key}']:+.1f} vs {r[theory]:+.1f})" for r in bad))
    for key, label in (("realised", "realised-vol delta"), ("implied", "implied-vol delta")):
        ratio = [r[f"std_{key}"] / r[f"theory_std_{key}"] for r in rows]
        worst = max(range(len(rows)), key=lambda i: abs(ratio[i] - 1))
        checks.add(f"std vs theory ({label})", abs(ratio[worst] - 1) < 0.05,
                   f"sim/theory within {abs(ratio[worst] - 1):.1%} "
                   f"(worst at sigma_r={rows[worst]['sigma_r']:.2f})")
    ratio = [r["std_residual"] / r["theory_std_residual"] for r in rows]
    worst = max(range(len(rows)), key=lambda i: abs(ratio[i] - 1))
    checks.add("implied-vol hedge minus (1/2) int Gamma S^2 (sigma_i^2 - sigma_r^2) dt "
               "= discrete-hedging error", abs(ratio[worst] - 1) < 0.05,
               f"residual std within {abs(ratio[worst] - 1):.1%} of theory "
               f"(worst at sigma_r={rows[worst]['sigma_r']:.2f})")
    far = [r for r in rows if abs(r["sigma_r"] - setup.sigma_i) >= 0.1 - 1e-9]
    if far:
        tight = all(r["std_realised"] < r["std_implied"] for r in far)
        checks.add("realised-vol hedge has lower dispersion when |sigma_r - sigma_i| >= 10 vols",
                   tight, "std ratio realised/implied: " + ", ".join(
                       f"{r['std_realised'] / r['std_implied']:.2f}" for r in far))
    sign_ok = all(np.sign(r["mean_realised"]) == np.sign(setup.sigma_i - r["sigma_r"]) *
                  np.sign(-setup.option.position) for r in rows
                  if abs(r["sigma_r"] - setup.sigma_i) > 0.02)
    checks.add("short option earns when sigma_r < sigma_i and loses when sigma_r > sigma_i", sign_ok,
               "signs as expected" if sign_ok else "unexpected sign")
    return checks


def _plot_summary(setup: Setup, rows: list[dict], n: int):
    sig = np.array([r["sigma_r"] for r in rows])
    grid = np.linspace(sig.min() - 0.02, sig.max() + 0.02, 200)
    pos, kw = setup.option.position, setup.theory_kwargs
    lock = [th.vol_gap_mean(setup.S0, setup.K, setup.T, setup.sigma_i, s, kind=setup.option.kind,
                            position=pos, hedge="realised", **kw) for s in grid]
    blue, orange = plt_.SERIES[0], plt_.SERIES[1]
    dx = 0.004
    with plt_.style():
        fig, axes = plt_.figure(1, 2, width=13.5, height=4.8)
        ax1, ax2 = axes.flat
        ax1.axhline(0.0, color=plt_.AXIS, lw=0.8)
        ax1.axvline(setup.sigma_i, color=plt_.AXIS, lw=0.8)
        ax1.plot(grid, lock, color=plt_.INK, lw=1.5,
                 label="theory  BS(σᵢ) − BS(σᵣ)")
        for key, color, marker, label, shift in (
                ("implied", blue, "o", "hedge with implied-vol delta", -dx),
                ("realised", orange, "s", "hedge with realised-vol delta", dx)):
            lo = np.array([r[f"q05_{key}"] for r in rows])
            hi = np.array([r[f"q95_{key}"] for r in rows])
            ax1.fill_between(sig, lo, hi, color=color, alpha=0.10, lw=0)
            ax1.plot(sig + shift, [r[f"mean_{key}"] for r in rows], marker, color=color,
                     label=f"{label}: mean, 5–95% band", zorder=3)
        ax1.set_xlabel("realised volatility σᵣ")
        ax1.set_ylabel("hedged P&L of the short option")
        plt_.money_axis(ax1)
        ax1.xaxis.set_major_formatter(plt_.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax1.set_title(f"Sold at σᵢ = {setup.sigma_i:.0%}: P&L ≈ BS(σᵢ) − BS(σᵣ) on average")
        ax1.legend(loc="lower left")
        top = ax1.get_ylim()[1]
        ax1.text(setup.sigma_i - 0.01, top * 0.92, "short earns\nσᵣ < σᵢ", ha="right",
                 va="top", fontsize=8.5, color=plt_.INK_2)
        ax1.text(setup.sigma_i + 0.01, top * 0.92, "short loses\nσᵣ > σᵢ", ha="left",
                 va="top", fontsize=8.5, color=plt_.INK_2)

        fine = np.linspace(sig.min(), sig.max(), 60)
        bps = setup.costs.bps
        std_i = [th.pnl_std(n, setup.S0, setup.K, setup.T, setup.sigma_i, s, None, bps,
                            position=pos, **kw) for s in fine]
        std_r = [th.pnl_std(n, setup.S0, setup.K, setup.T, setup.sigma_i, s, s, bps,
                            position=pos, **kw) for s in fine]
        ax2.plot(fine, std_i, color=plt_.INK, lw=1.5,
                 label=f"theory ({setup.interval(n)} rebalancing)")
        ax2.plot(fine, std_r, color=plt_.INK, lw=1.5)
        ax2.plot(sig, [r["std_implied"] for r in rows], "o", color=blue, label="implied-vol delta",
                 zorder=3)
        ax2.plot(sig, [r["std_realised"] for r in rows], "s", color=orange,
                 label="realised-vol delta", zorder=3)
        ax2.axvline(setup.sigma_i, color=plt_.AXIS, lw=0.8)
        ax2.set_ylim(bottom=0)
        ax2.set_xlabel("realised volatility σᵣ")
        ax2.set_ylabel("std of hedged P&L")
        plt_.money_axis(ax2)
        ax2.xaxis.set_major_formatter(plt_.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax2.set_title("Realised-vol delta locks it in; implied-vol delta is path-dependent")
        ax2.legend(loc="upper left")
        plt_.headline(fig, "A delta-hedged short option is a bet that realised volatility "
                           "comes in below implied")
    return fig


def _plot_distributions(setup: Setup, samples: dict, show: list[float]):
    blue, orange = plt_.SERIES[0], plt_.SERIES[1]
    pos, kw = setup.option.position, setup.theory_kwargs
    rng = np.random.default_rng(0)
    with plt_.style():
        fig, axes = plt_.figure(2, len(show), width=6.6 * len(show), height=8.4)
        for j, sigma_r in enumerate(show):
            x = samples[sigma_r]
            lock = th.vol_gap_mean(setup.S0, setup.K, setup.T, setup.sigma_i, sigma_r,
                                   kind=setup.option.kind, position=pos, hedge="realised", **kw)
            ax = axes[0, j]
            lo, hi = np.quantile(np.concatenate([x["implied"], x["realised"]]), [0.002, 0.998])
            bins = np.linspace(lo, hi, 90)
            for key, color, label in (("realised", orange, "realised-vol delta"),
                                      ("implied", blue, "implied-vol delta")):
                v = x[key]
                ax.hist(v, bins=bins, density=True, histtype="stepfilled", color=color, alpha=0.12)
                ax.hist(v, bins=bins, density=True, histtype="step", color=color, lw=1.6,
                        label=f"{label}: mean {plt_.compact(v.mean())}, std {plt_.compact(v.std())}")
            ax.axvline(lock, color=plt_.INK, lw=1.5, label=f"BS(σᵢ) − BS(σᵣ) = {plt_.compact(lock)}")
            ax.set_yticks([])
            ax.set_xlabel("hedged P&L at expiry")
            plt_.money_axis(ax, "x")
            verdict = "earns" if (setup.sigma_i - sigma_r) * -pos > 0 else "loses"
            ax.set_title(f"σᵣ = {sigma_r:.0%} vs σᵢ = {setup.sigma_i:.0%}: the short {verdict}")
            ax.legend(loc="upper left")

            ax = axes[1, j]
            idx = rng.choice(x["implied"].size, size=min(4000, x["implied"].size), replace=False)
            pw, sim = x["pathwise"][idx], x["implied"][idx]
            ax.scatter(pw, sim, s=5, color=blue, alpha=0.35, lw=0)
            lim = np.quantile(np.concatenate([pw, sim]), [0.001, 0.999])
            ax.plot(lim, lim, color=plt_.INK, lw=1.2, label="y = x")
            corr = np.corrcoef(x["pathwise"], x["implied"])[0, 1]
            plt_.note(ax, f"correlation {corr:.3f}", loc="lower right")
            ax.set_xlabel("½ ∫ Γᵢ S² (σᵢ² − σᵣ²) dt  along the path")
            ax.set_ylabel("simulated P&L, implied-vol delta")
            plt_.money_axis(ax)
            plt_.money_axis(ax, "x")
            ax.set_title("Each path's P&L is its gamma-weighted variance gap")
            ax.legend(loc="upper left")
        plt_.headline(fig, "Same average, different risk: realised-vol delta locks in the vol "
                           "edge, implied-vol delta leaves it path-dependent")
    return fig


def _print(setup: Setup, result: ExperimentResult, n_paths: int) -> None:
    print(f"\n=== Experiment 2: realised vs implied vol ({n_paths:,} paths, "
          f"{result.summary['interval']} hedging, {result.seconds:.1f}s) ===")
    print(setup.describe())
    print(print_table(result.rows, [
        ("sigma_r", "sigma_r", ".2f"), ("theory_lockin", "BS(i)-BS(r)", ",.1f"),
        ("mean_implied", "mean(i-delta)", ",.1f"), ("se_implied", "se", ".1f"),
        ("std_implied", "std(i-delta)", ",.1f"), ("mean_realised", "mean(r-delta)", ",.1f"),
        ("se_realised", "se", ".1f"), ("std_realised", "std(r-delta)", ",.1f"),
        ("theory_std_realised", "theory", ",.1f"), ("theory_std_implied", "theory(i)", ",.1f"),
        ("std_residual", "resid std", ",.1f"), ("theory_std_residual", "theory", ",.1f"),
    ]))
    print(result.checks.report())
    for f in result.files:
        print(f"  wrote {f}")
