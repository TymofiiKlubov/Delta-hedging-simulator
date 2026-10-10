"""Plumbing shared by the experiments: setup from config, chunked simulation, checks, output."""

from __future__ import annotations

import csv
import json
import math
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import bs
from ..config import Config
from ..hedging import Costs, Market, Option
from ..paths import Model, PathSet, chunk_sizes


@dataclass(frozen=True)
class Setup:
    """The trade and market implied by a config."""

    cfg: Config
    option: Option
    market: Market
    sigma_i: float
    sigma_r: float
    mu: float
    costs: Costs

    @classmethod
    def from_config(cls, cfg: Config) -> Setup:
        if cfg.vol.hedge_with not in ("implied", "realised"):
            raise ValueError("vol.hedge_with must be 'implied' or 'realised'")
        option = Option(cfg.strike, cfg.expiry, cfg.option.kind, cfg.option.position)
        market = Market(cfg.market.spot, cfg.market.r, cfg.market.q)
        return cls(cfg, option, market, cfg.vol.implied, cfg.vol.realised, cfg.market.mu,
                   Costs(cfg.costs.bps, cfg.costs.fixed))

    @property
    def S0(self) -> float:
        return self.market.spot

    @property
    def K(self) -> float:
        return self.option.strike

    @property
    def T(self) -> float:
        return self.option.expiry

    @property
    def expiry_days(self) -> float:
        return self.cfg.option.expiry_days

    @property
    def hedge_vol(self) -> float | None:
        """Vol for the hedge delta: None means the implied vol."""
        return self.sigma_r if self.cfg.vol.hedge_with == "realised" else None

    @property
    def premium(self) -> float:
        """Premium of the whole position (|position| options) at the implied vol."""
        return abs(self.option.position) * float(
            bs.price(self.S0, self.K, self.T, self.sigma_i, self.market.r, self.market.q,
                     self.option.kind))

    @property
    def theory_kwargs(self) -> dict[str, Any]:
        return dict(r=self.market.r, q=self.market.q, mu=self.mu)

    @property
    def risk_neutral_drift(self) -> bool:
        return math.isclose(self.mu, self.market.r - self.market.q, abs_tol=1e-12)

    def interval(self, n_rebalances: int) -> str:
        return interval_label(self.expiry_days, n_rebalances)

    def describe(self) -> str:
        o = self.option
        side = "short" if o.position < 0 else "long"
        return (f"{side} {abs(o.position):g} x {self.expiry_days:g}-day {o.kind} K={self.K:,.0f} "
                f"on S0={self.S0:,.0f}, sigma_i={self.sigma_i:.0%}, r={self.market.r:g}, "
                f"q={self.market.q:g}, mu={self.mu:g}")


def interval_label(expiry_days: float, n_rebalances: int) -> str:
    """Human-readable rebalancing interval, e.g. 30 days / 720 -> '1h'."""
    minutes = expiry_days * 1440.0 / n_rebalances
    if minutes >= 1440 - 1e-9:
        value, unit = minutes / 1440, "d"
    elif minutes >= 60 - 1e-9:
        value, unit = minutes / 60, "h"
    else:
        value, unit = minutes, "min"
    return f"{value:.3g}{unit}"


def common_grid(steps: Sequence[int], cap: int = 100_000) -> int:
    """Smallest grid on which every rebalancing schedule in ``steps`` sits exactly."""
    fine = math.lcm(*[int(s) for s in steps])
    if fine > cap:
        raise ValueError(f"rebalance counts {sorted(steps)} need a {fine}-step common grid; "
                         "choose counts that divide a common number of steps (<= 100,000)")
    return fine


def simulate_chunks(model: Model, setup: Setup, n_steps: int, n_paths: int,
                    seed: int) -> Iterator[PathSet]:
    """Simulate ``n_paths`` paths in memory-bounded chunks from one random stream."""
    rng = np.random.default_rng(seed)
    for size in chunk_sizes(n_paths, n_steps, setup.cfg.simulation.max_chunk_elements):
        yield model.simulate(setup.S0, setup.T, n_steps, size, rng, mu=setup.mu)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class Checks:
    """Simulation-versus-theory checks; failures are flagged, not raised."""

    items: list[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str) -> bool:
        self.items.append(Check(name, bool(ok), detail))
        return bool(ok)

    @property
    def all_ok(self) -> bool:
        return all(c.ok for c in self.items)

    def report(self) -> str:
        return "\n".join(f"  [{'ok' if c.ok else 'FLAG'}] {c.name}: {c.detail}" for c in self.items)

    def to_list(self) -> list[dict[str, Any]]:
        return [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in self.items]


@dataclass
class ExperimentResult:
    name: str
    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    checks: Checks
    figures: dict[str, Any]  # file stem -> matplotlib Figure
    files: list[Path] = field(default_factory=list)
    seconds: float = 0.0


def write_outputs(result: ExperimentResult, setup: Setup, out_dir: str | Path | None) -> None:
    """Save charts (PNG), the table (CSV) and a JSON summary with checks and config."""
    from ..plotting import save

    if out_dir is None:
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stem, fig in result.figures.items():
        result.files.append(save(fig, out / f"{stem}.png"))
    if result.rows:
        path = out / f"{result.name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(result.rows[0]))
            writer.writeheader()
            writer.writerows({k: _csv_value(v) for k, v in row.items()} for row in result.rows)
        result.files.append(path)
    path = out / f"{result.name}.json"
    payload = {"experiment": result.name, "setup": setup.describe(), "summary": result.summary,
               "checks": result.checks.to_list(), "seconds": round(result.seconds, 2),
               "config": setup.cfg.to_dict()}
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")
    result.files.append(path)


def print_table(rows: list[dict[str, Any]], columns: list[tuple[str, str, str]]) -> str:
    """Fixed-width text table. ``columns`` = (key, header, format spec)."""
    cells = [[format(row[key], fmt) if isinstance(row[key], (int, float, np.floating, np.integer))
              else str(row[key]) for key, _, fmt in columns] for row in rows]
    heads = [h for _, h, _ in columns]
    widths = [max(len(h), *(len(c[i]) for c in cells)) for i, h in enumerate(heads)]
    lines = ["  ".join(h.rjust(w) for h, w in zip(heads, widths))]
    lines += ["  ".join(c.rjust(w) for c, w in zip(row, widths)) for row in cells]
    return "\n".join(lines)


class Timer:
    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self.start


def _csv_value(v: Any) -> Any:
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.10g}"
    return v


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj
