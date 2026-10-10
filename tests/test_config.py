from pathlib import Path

import pytest

from dhsim.config import Config, load_config, merge

ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_mirrors_dataclass_defaults():
    assert load_config(ROOT / "configs" / "default.yaml") == Config()


def test_merge_nested_and_coerces_types():
    cfg = merge(Config(), {"vol": {"implied": 1}, "frequency": {"steps": [1, 30]},
                           "models": {"heston": {"xi": 2}}})
    assert cfg.vol.implied == 1.0 and isinstance(cfg.vol.implied, float)
    assert cfg.frequency.steps == (1, 30)
    assert cfg.models.heston.xi == 2.0
    assert cfg.models.merton == Config().models.merton  # untouched siblings survive


def test_unknown_key_raises():
    with pytest.raises(KeyError):
        merge(Config(), {"vol": {"implid": 0.5}})


def test_derived_strike_and_expiry():
    cfg = Config()
    assert cfg.strike == cfg.market.spot
    assert cfg.expiry == pytest.approx(30 / 365)
    assert merge(cfg, {"option": {"strike": 65000}}).strike == 65000.0
