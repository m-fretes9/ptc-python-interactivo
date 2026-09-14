"""Regression tests for selectable solvers and factored radiation."""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from defaults import default_config, default_fluid_database
from ptc_model import PTCSimulator, fourth_power_difference


def test_factored_fourth_power_difference_matches_direct() -> None:
    for t1, t2 in [(450.0, 300.0), (350.00001, 350.0), (1000.0, 250.0)]:
        direct = t1**4 - t2**4
        factored = fourth_power_difference(t1, t2)
        assert np.isclose(factored, direct, rtol=2e-15, atol=1e-6)


def _small_config(method: str):
    cfg = default_config()
    cfg["geometry"]["Nseg"] = 3
    cfg["operation"]["t_start_s"] = 12.0 * 3600.0
    cfg["operation"]["t_end_s"] = 12.05 * 3600.0
    cfg["operation"]["output_step_s"] = 30.0
    cfg["solver"]["method"] = method
    cfg["solver"]["max_step_s"] = 10.0
    return cfg


def test_rk45_and_bdf_agree_on_small_case() -> None:
    db = default_fluid_database()
    bdf = PTCSimulator(_small_config("BDF"), db).simulate()
    rk = PTCSimulator(_small_config("RK45"), db).simulate()
    assert np.isclose(rk.Tout_C[-1], bdf.Tout_C[-1], rtol=2e-4, atol=2e-3)
    assert np.isclose(rk.scalar_diag["Quseful_W"][-1], bdf.scalar_diag["Quseful_W"][-1], rtol=5e-4, atol=5e-2)
