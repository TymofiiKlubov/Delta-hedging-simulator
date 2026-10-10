import json
import subprocess
import sys

import pytest

from dhsim.cli import apply_overrides, build_parser, main, parse_floats, parse_steps
from dhsim.config import Config


def test_parse_steps_accepts_counts_intervals_and_names():
    assert parse_steps("1,4,30,720", 30) == [1, 4, 30, 720]
    assert parse_steps("weekly,daily,1h,15min,5m", 30) == [4, 30, 720, 2880, 8640]
    assert parse_steps("monthly", 30) == [1]
    assert parse_steps("1d", 7) == [7]
    with pytest.raises(Exception):
        parse_steps("fortnightly", 30)


def test_parse_floats_range_is_inclusive():
    assert parse_floats("0.4:0.8:0.05") == [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    assert parse_floats("0.5,0.7") == [0.5, 0.7]


def _cfg(argv):
    return apply_overrides(Config(), build_parser().parse_args(argv))


def test_overrides_route_to_the_right_config_sections():
    cfg = _cfg(["costs", "--bps", "7", "--paths", "123", "--monitor", "1h"])
    assert cfg.costs_tradeoff.bps == 7.0 and cfg.costs.bps == 0.0
    assert cfg.costs_tradeoff.paths == 123 and cfg.costs_tradeoff.monitor_steps == 720
    cfg = _cfg(["frequency", "--bps", "2", "--sigma-r", "0.5", "--steps", "daily,hourly"])
    assert cfg.costs.bps == 2.0 and cfg.vol.realised == 0.5 and cfg.frequency.steps == (30, 720)
    cfg = _cfg(["path", "--seed", "9", "--expiry-days", "7", "--steps", "1h"])
    assert cfg.path.seed == 9 and cfg.simulation.seed == Config().simulation.seed
    assert cfg.path.steps == 168  # hourly over a 7-day option
    cfg = _cfg(["volgap", "--sigma-r", "0.3:0.5:0.1", "--steps", "1d"])
    assert cfg.volgap.sigma_r == (0.3, 0.4, 0.5) and cfg.volgap.steps == 30
    cfg = _cfg(["all", "--paths", "50"])
    assert {cfg.volgap.paths, cfg.costs_tradeoff.paths, cfg.models.paths, cfg.simulation.paths} == {50}
    cfg = _cfg(["models", "--lam", "2", "--xi", "0.8"])
    assert cfg.models.merton.lam == 2.0 and cfg.models.heston.xi == 0.8


SMOKE = {
    "frequency": ["--paths", "300", "--steps", "1,4,30"],
    "volgap": ["--paths", "300", "--sigma-r", "0.5,0.7", "--steps", "30", "--show", "0.5,0.7"],
    "costs": ["--paths", "300", "--steps", "1,4,30", "--bands", "0.05,0.2", "--monitor", "60"],
    "path": ["--steps", "48"],
    "models": ["--paths", "300", "--steps", "8,30", "--show-steps", "30"],
}


@pytest.mark.parametrize("command", list(SMOKE))
def test_each_command_writes_charts_table_and_summary(command, tmp_path, capsys):
    assert main([command, *SMOKE[command], "--out", str(tmp_path), "--quiet"]) == 0
    stem = command
    assert (tmp_path / f"{stem}.png").stat().st_size > 10_000
    assert (tmp_path / f"{stem}.csv").read_text().count("\n") >= 2
    payload = json.loads((tmp_path / f"{stem}.json").read_text())
    assert payload["experiment"] == stem and payload["checks"]
    assert payload["config"]["output_dir"] == str(tmp_path)
    assert "[ok]" in capsys.readouterr().out


def test_price_command(capsys):
    assert main(["price", "--sigma-i", "0.6"]) == 0
    out = capsys.readouterr().out
    assert "4,112.36" in out and "breakeven move" in out


def test_module_entry_point():
    res = subprocess.run([sys.executable, "-m", "dhsim", "--help"], capture_output=True, text=True,
                         check=True)
    assert "frequency" in res.stdout and "volgap" in res.stdout
