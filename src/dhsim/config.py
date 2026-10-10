"""Configuration: dataclass defaults, YAML loading and overrides.

The defaults describe a crypto-like setup: spot 60,000, 60% implied volatility, a 30-day
at-the-money call, zero rates and a 365-day calendar year (the market trades 24/7).
``configs/default.yaml`` mirrors these defaults one-to-one (a test enforces that).
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class MarketConfig:
    spot: float = 60_000.0
    r: float = 0.0  # continuously-compounded interest rate (cash account)
    q: float = 0.0  # dividend / funding yield earned by holding the underlying
    mu: float = 0.0  # real-world drift of the spot price (excluding q)


@dataclass(frozen=True)
class OptionConfig:
    kind: str = "call"
    strike: float | None = None  # None -> at the money (strike = spot)
    expiry_days: float = 30.0
    position: float = -1.0  # options held; -1 = short one option


@dataclass(frozen=True)
class VolConfig:
    implied: float = 0.60  # sigma_i: the option is sold at BS(sigma_i)
    realised: float = 0.60  # sigma_r: volatility of the simulated paths
    hedge_with: str = "implied"  # delta computed with "implied" or "realised" vol


@dataclass(frozen=True)
class CostConfig:
    bps: float = 0.0  # proportional cost, basis points of traded notional
    fixed: float = 0.0  # fixed cost per trade (currency units)


@dataclass(frozen=True)
class SimulationConfig:
    paths: int = 20_000
    seed: int = 42
    days_per_year: float = 365.0
    max_chunk_elements: int = 4_000_000  # paths x steps simulated at once (memory cap)


@dataclass(frozen=True)
class FrequencyConfig:
    # Rebalances over the option's life. For a 30-day option: 1 = monthly, 4 ~ weekly,
    # 30 = daily, 720 = hourly, 2880 = every 15 minutes, 8640 = every 5 minutes.
    steps: tuple[int, ...] = (1, 2, 4, 8, 15, 30, 60, 120, 240, 480, 720, 1440, 2880, 8640)
    paths: int | None = None  # None -> simulation.paths


@dataclass(frozen=True)
class VolgapConfig:
    sigma_r: tuple[float, ...] = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
    steps: int = 720  # hourly hedging
    paths: int | None = 10_000
    show: tuple[float, ...] = (0.45, 0.75)  # realised vols whose distributions are plotted


@dataclass(frozen=True)
class CostTradeoffConfig:
    bps: float = 5.0
    fixed: float = 0.0
    risk_aversion: float = 1.0  # kappa in J = -E[PnL] + kappa * Var[PnL] / premium
    steps: tuple[int, ...] = (1, 2, 4, 8, 15, 30, 60, 120, 240, 480, 960, 1440, 2880)
    bands: tuple[float, ...] = (0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3)
    band_modes: tuple[str, ...] = ("target", "edge")
    monitor_steps: int = 2880  # band hedging checks the band every 15 minutes
    paths: int | None = 10_000


@dataclass(frozen=True)
class PathConfig:
    seed: int = 3
    steps: int = 720  # hourly grid
    every: int = 1  # rebalance every `every` grid steps


@dataclass(frozen=True)
class MertonConfig:
    lam: float = 5.0  # jumps per year
    jump_mean: float = -0.05  # mean of log jump size
    jump_std: float = 0.15  # std of log jump size


@dataclass(frozen=True)
class HestonConfig:
    kappa: float = 3.0
    xi: float = 1.5  # vol of variance
    rho: float = -0.3
    v0: float | None = None  # None -> implied vol squared
    theta: float | None = None  # None -> implied vol squared


@dataclass(frozen=True)
class ModelsConfig:
    steps: tuple[int, ...] = (8, 30, 120, 720, 2880)
    show_steps: int = 720  # hedge frequency whose P&L distributions are plotted
    paths: int | None = 10_000
    merton: MertonConfig = field(default_factory=MertonConfig)
    heston: HestonConfig = field(default_factory=HestonConfig)


@dataclass(frozen=True)
class Config:
    market: MarketConfig = field(default_factory=MarketConfig)
    option: OptionConfig = field(default_factory=OptionConfig)
    vol: VolConfig = field(default_factory=VolConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    frequency: FrequencyConfig = field(default_factory=FrequencyConfig)
    volgap: VolgapConfig = field(default_factory=VolgapConfig)
    costs_tradeoff: CostTradeoffConfig = field(default_factory=CostTradeoffConfig)
    path: PathConfig = field(default_factory=PathConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    output_dir: str = "output"

    # -- derived quantities -------------------------------------------------------------

    @property
    def strike(self) -> float:
        return self.market.spot if self.option.strike is None else float(self.option.strike)

    @property
    def expiry(self) -> float:
        """Time to expiry in years."""
        return self.option.expiry_days / self.simulation.days_per_year

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(dataclasses.asdict(self))


# -- loading / overriding -------------------------------------------------------------


def load_config(path: str | Path | None = None) -> Config:
    """Default config, optionally updated from a YAML file (unknown keys raise)."""
    cfg = Config()
    if path is None:
        return cfg
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return merge(cfg, data)


def merge(obj: Any, updates: dict[str, Any]) -> Any:
    """Return a copy of dataclass ``obj`` with ``updates`` (nested dict) applied."""
    hints = typing.get_type_hints(type(obj))
    names = {f.name for f in dataclasses.fields(obj)}
    changes: dict[str, Any] = {}
    for key, value in updates.items():
        if key not in names:
            raise KeyError(f"unknown config key '{key}' for {type(obj).__name__}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"config key '{key}' expects a mapping")
            changes[key] = merge(current, value)
        else:
            changes[key] = _coerce(value, hints[key])
    return dataclasses.replace(obj, **changes)


def _coerce(value: Any, hint: Any) -> Any:
    """Coerce YAML scalars/lists to the annotated field type (tuples, floats, ints)."""
    if value is None:
        return None
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin is typing.Union or origin is types.UnionType:
        inner = [a for a in args if a is not type(None)]
        return _coerce(value, inner[0]) if len(inner) == 1 else value
    if origin is tuple:
        items = value if isinstance(value, (list, tuple)) else [value]
        return tuple(_coerce(v, args[0]) for v in items)
    if hint is float:
        return float(value)
    if hint is int:
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"expected an integer, got {value}")
        return int(value)
    if hint is str:
        return str(value)
    return value


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj
