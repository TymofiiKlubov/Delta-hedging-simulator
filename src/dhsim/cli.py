"""Command-line interface: ``python -m dhsim <command> [options]``.

Every experiment reads the YAML config (``--config``, default: the built-in defaults that
``configs/default.yaml`` mirrors) and then applies command-line overrides.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from . import bs
from .config import Config, load_config, merge
from .experiments import EXPERIMENTS, get

UNITS_DAYS = {"w": 7.0, "d": 1.0, "h": 1 / 24, "min": 1 / 1440, "m": 1 / 1440}
NAMED = {"monthly": "30d", "weekly": "7d", "daily": "1d", "hourly": "1h"}
# config section holding each experiment's path count
PATHS_SECTION = {"frequency": "frequency", "volgap": "volgap", "costs": "costs_tradeoff",
                 "models": "models"}
DESCRIPTIONS = {
    "frequency": "Exp 1 - hedge frequency: std falls ~ 1/sqrt(N), mean stays ~ 0",
    "volgap": "Exp 2 - realised vs implied vol, implied- vs realised-vol delta",
    "costs": "Exp 3 - transaction costs vs frequency, optimal N, band hedging",
    "path": "Exp 4 - single-path walkthrough: theta vs gamma",
    "models": "Exp 5 - Merton jumps and Heston vol-of-vol: fat tails",
    "all": "run all five experiments",
    "price": "Black-Scholes price and Greeks",
}


def parse_steps(text: str, expiry_days: float) -> list[int]:
    """'1,4,30,720' or '1w,1d,1h,5min' or 'daily,hourly' -> rebalance counts over the option's life.

    Intervals are converted with the option's expiry and rounded to a whole number of rebalances.
    """
    out = []
    for token in str(text).replace(" ", "").split(","):
        if not token:
            continue
        token = NAMED.get(token.lower(), token.lower())
        if token.isdigit():
            out.append(int(token))
            continue
        for unit in ("min", "w", "d", "h", "m"):
            if token.endswith(unit):
                interval = float(token[: -len(unit)] or 1) * UNITS_DAYS[unit]
                out.append(max(1, int(round(expiry_days / interval))))
                break
        else:
            raise argparse.ArgumentTypeError(f"cannot parse rebalancing '{token}'")
    if not out:
        raise argparse.ArgumentTypeError("no rebalancing counts given")
    return out


def parse_floats(text: str) -> list[float]:
    """'0.4:0.8:0.05' (inclusive range) or '0.4,0.5,0.6' or '0.5'."""
    text = str(text).replace(" ", "")
    if ":" in text:
        lo, hi, step = (float(x) for x in text.split(":"))
        if step <= 0 or hi < lo:
            raise argparse.ArgumentTypeError(f"bad range '{text}'")
        return [round(v, 10) for v in np.arange(lo, hi + step / 2, step)]
    return [float(x) for x in text.split(",") if x]


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    g = common.add_argument_group("setup (overrides the config file)")
    g.add_argument("--config", help="YAML config (default: built-in defaults, see configs/default.yaml)")
    g.add_argument("--out", help="output directory (default: config output_dir = output)")
    g.add_argument("--paths", type=int, help="number of simulated paths")
    g.add_argument("--seed", type=int, help="random seed")
    g.add_argument("--spot", type=float)
    g.add_argument("--strike", type=float, help="default: at the money")
    g.add_argument("--expiry-days", type=float)
    g.add_argument("--kind", choices=["call", "put"])
    g.add_argument("--position", type=float, help="options held, -1 = short one (default)")
    g.add_argument("--r", type=float, help="interest rate")
    g.add_argument("--q", type=float, help="dividend / funding yield")
    g.add_argument("--mu", type=float, help="real-world drift of the spot")
    g.add_argument("--sigma-i", type=float, help="implied vol: sale price and default hedge vol")
    g.add_argument("--hedge-with", choices=["implied", "realised"], help="vol used for the hedge delta")
    g.add_argument("--bps", type=float, help="proportional cost, bp of traded notional "
                                            "(for 'costs': the cost level being studied)")
    g.add_argument("--fixed", type=float, help="fixed cost per trade")
    g.add_argument("--quiet", action="store_true", help="print only the checks")
    g.add_argument("--strict", action="store_true",
                   help="exit with status 1 if any simulation-vs-theory check is flagged")

    parser = argparse.ArgumentParser(
        prog="dhsim", description="Delta-hedging simulator: sell an option, hedge it, see why "
                                  "the hedge makes or loses money.")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    steps_help = "rebalance counts or intervals, e.g. 1,4,30,720 or weekly,daily,1h,5min"

    p = sub.add_parser("frequency", parents=[common], help=DESCRIPTIONS["frequency"])
    p.add_argument("--sigma-r", type=float, help="realised vol of the paths")
    p.add_argument("--steps", help=steps_help)

    p = sub.add_parser("volgap", parents=[common], help=DESCRIPTIONS["volgap"])
    p.add_argument("--sigma-r", help="realised vols: lo:hi:step or a comma list (default 0.4:0.8:0.05)")
    p.add_argument("--steps", help="rebalancing, e.g. 720 or 1h")
    p.add_argument("--show", help="realised vols whose P&L distributions are plotted")

    p = sub.add_parser("costs", parents=[common], help=DESCRIPTIONS["costs"])
    p.add_argument("--sigma-r", type=float, help="realised vol of the paths")
    p.add_argument("--steps", help=steps_help)
    p.add_argument("--bands", help="delta bands, comma list, e.g. 0.01,0.02,0.05")
    p.add_argument("--monitor", help="how often bands are checked, e.g. 2880 or 15min")
    p.add_argument("--risk-aversion", type=float, help="kappa in J = -E[PnL] + kappa Var/premium")

    p = sub.add_parser("path", parents=[common], help=DESCRIPTIONS["path"])
    p.add_argument("--sigma-r", type=float, help="realised vol of the path")
    p.add_argument("--steps", help="time grid, e.g. 720 or 1h")
    p.add_argument("--every", type=int, help="rebalance every k grid steps")

    p = sub.add_parser("models", parents=[common], help=DESCRIPTIONS["models"])
    p.add_argument("--steps", help=steps_help)
    p.add_argument("--show-steps", help="frequency whose distributions are plotted, e.g. 1h")
    p.add_argument("--lam", type=float, help="Merton jump intensity per year")
    p.add_argument("--jump-mean", type=float, help="Merton mean log jump")
    p.add_argument("--jump-std", type=float, help="Merton log-jump std")
    p.add_argument("--kappa", type=float, help="Heston mean reversion")
    p.add_argument("--xi", type=float, help="Heston vol of variance")
    p.add_argument("--rho", type=float, help="Heston spot-vol correlation")

    p = sub.add_parser("all", parents=[common], help=DESCRIPTIONS["all"])
    p.add_argument("--sigma-r", type=float, help="realised vol (experiments 1, 3, 4)")

    p = sub.add_parser("price", parents=[common], help=DESCRIPTIONS["price"])
    return parser


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Fold command-line options into the config."""
    cmd = args.command
    upd: dict = {}

    def put(section: str, key: str, value) -> None:
        if value is not None:
            upd.setdefault(section, {})[key] = value

    for key, attr in (("spot", "spot"), ("r", "r"), ("q", "q"), ("mu", "mu")):
        put("market", key, getattr(args, attr))
    for key, attr in (("strike", "strike"), ("expiry_days", "expiry_days"), ("kind", "kind"),
                      ("position", "position")):
        put("option", key, getattr(args, attr))
    put("vol", "implied", args.sigma_i)
    put("vol", "hedge_with", args.hedge_with)
    costs_section = "costs_tradeoff" if cmd == "costs" else "costs"
    put(costs_section, "bps", args.bps)
    put(costs_section, "fixed", args.fixed)
    if args.seed is not None:
        put("path" if cmd == "path" else "simulation", "seed", args.seed)
    if args.paths is not None:
        put("simulation", "paths", args.paths)
        targets = PATHS_SECTION.values() if cmd == "all" else [PATHS_SECTION.get(cmd)]
        for section in filter(None, targets):
            put(section, "paths", args.paths)
    if args.out is not None:
        upd["output_dir"] = args.out
    cfg = merge(cfg, upd)

    # experiment-specific options (parsed after the expiry is known)
    days = cfg.option.expiry_days
    sigma_r = getattr(args, "sigma_r", None)
    upd = {}
    if cmd == "volgap":
        if sigma_r is not None:
            put("volgap", "sigma_r", parse_floats(sigma_r))
        if args.steps is not None:
            put("volgap", "steps", parse_steps(args.steps, days)[0])
        if args.show is not None:
            put("volgap", "show", parse_floats(args.show))
    elif sigma_r is not None:
        put("vol", "realised", sigma_r)
    if cmd == "frequency" and args.steps is not None:
        put("frequency", "steps", parse_steps(args.steps, days))
    if cmd == "costs":
        if args.steps is not None:
            put("costs_tradeoff", "steps", parse_steps(args.steps, days))
        if args.bands is not None:
            put("costs_tradeoff", "bands", parse_floats(args.bands))
        if args.monitor is not None:
            put("costs_tradeoff", "monitor_steps", parse_steps(args.monitor, days)[0])
        put("costs_tradeoff", "risk_aversion", args.risk_aversion)
    if cmd == "path":
        if args.steps is not None:
            put("path", "steps", parse_steps(args.steps, days)[0])
        put("path", "every", args.every)
    if cmd == "models":
        if args.steps is not None:
            put("models", "steps", parse_steps(args.steps, days))
        if args.show_steps is not None:
            put("models", "show_steps", parse_steps(args.show_steps, days)[0])
        for key in ("lam", "jump_mean", "jump_std"):
            if getattr(args, key) is not None:
                upd.setdefault("models", {}).setdefault("merton", {})[key] = getattr(args, key)
        for key in ("kappa", "xi", "rho"):
            if getattr(args, key) is not None:
                upd.setdefault("models", {}).setdefault("heston", {})[key] = getattr(args, key)
    return merge(cfg, upd)


def price_report(cfg: Config) -> str:
    S, K, T = cfg.market.spot, cfg.strike, cfg.expiry
    r, q, sigma, kind = cfg.market.r, cfg.market.q, cfg.vol.implied, cfg.option.kind
    g = bs.greeks(S, K, T, sigma, r, q, kind)
    day = 1.0 / cfg.simulation.days_per_year
    move = S * sigma * math.sqrt(day)
    lines = [
        f"{cfg.option.expiry_days:g}-day {kind}, S={S:,.2f} K={K:,.2f} sigma={sigma:.2%} r={r:g} q={q:g}",
        f"  price            {float(g.price):,.4f}",
        f"  delta            {float(g.delta):.6f}",
        f"  gamma            {float(g.gamma):.6e}   (dollar gamma S^2 Gamma = {float(g.gamma) * S * S:,.2f})",
        f"  vega             {float(g.vega) / 100:,.4f} per vol point",
        f"  theta            {float(g.theta) * day:,.4f} per day",
        f"  breakeven move   +/-{move:,.2f} per day (S sigma sqrt(1 day)): the daily move at which "
        "a delta-hedged long option's gamma P&L pays its theta",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args)
    if args.command == "price":
        print(price_report(cfg))
        return 0
    out = Path(cfg.output_dir)
    names = list(EXPERIMENTS) if args.command == "all" else [args.command]
    results = []
    for name in names:
        result = get(name)(cfg, out_dir=out, verbose=not args.quiet)
        if args.quiet:
            print(f"{name}:\n{result.checks.report()}")
        results.append(result)
    if args.command == "all":
        path = write_summary(results, out)
        print(f"\nsummary of checks written to {path}")
    flagged = [c for r in results for c in r.checks.items if not c.ok]
    if flagged:
        print(f"\n{len(flagged)} check(s) flagged - see the [FLAG] lines above", file=sys.stderr)
    return 1 if (flagged and args.strict) else 0


def write_summary(results: list, out: Path) -> Path:
    lines = ["# Simulation vs theory checks", "",
             "Generated by `python -m dhsim all`.", "",
             "| experiment | check | result | detail |", "|---|---|---|---|"]
    for r in results:
        for c in r.checks.items:
            detail = c.detail.replace("|", "\\|")
            lines.append(f"| {r.name} | {c.name} | {'ok' if c.ok else '**FLAG**'} | {detail} |")
    lines += ["", "| experiment | runtime (s) |", "|---|---|"]
    lines += [f"| {r.name} | {r.seconds:.1f} |" for r in results]
    path = out / "checks.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "build_parser", "apply_overrides", "parse_steps", "parse_floats"]
