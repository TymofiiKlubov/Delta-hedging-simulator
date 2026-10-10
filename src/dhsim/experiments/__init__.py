"""The five experiments. Each module's ``run(cfg, out_dir, verbose)`` returns an
ExperimentResult and, when ``out_dir`` is given, writes PNG charts, a CSV table and a JSON
summary there."""

from __future__ import annotations

import importlib
from collections.abc import Callable

# CLI name -> module (imported lazily)
EXPERIMENTS = {
    "frequency": "frequency",
    "volgap": "volgap",
    "costs": "costs",
    "path": "walkthrough",
    "models": "models",
}


def get(name: str) -> Callable:
    """The ``run`` function of experiment ``name``."""
    return importlib.import_module(f"{__name__}.{EXPERIMENTS[name]}").run
