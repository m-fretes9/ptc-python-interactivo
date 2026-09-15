from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from defaults import default_fluid_database
from rea_meteorological_wind_test import (
    adjust_wind_height,
    foz_weatherspark_wind_10m,
    run_monthly_wind_hypothesis,
)
from validations import inverse_parameter_options


def test_foz_weatherspark_template_and_height_adjustment():
    df = foz_weatherspark_wind_10m()
    assert len(df) == 12
    assert (df["Viento_10m_m_s"] > 1.5).all()
    adjusted = adjust_wind_height(
        df["Viento_10m_m_s"], receiver_height_m=1.0, alpha=0.14, apply_adjustment=True
    )
    assert adjusted.shape == (12,)
    assert np.all(adjusted < df["Viento_10m_m_s"].to_numpy())


def test_monthly_wind_hypothesis_two_months_runs_without_calibration():
    df = foz_weatherspark_wind_10m()
    out = run_monthly_wind_hypothesis(
        "rea_foz",
        default_fluid_database(),
        df["Viento_10m_m_s"],
        months=[1, 7],
    )
    table = out["table"]
    assert len(table) == 2
    assert {"Eta_ref_pct", "Eta_base_pct", "Eta_meteo_pct", "Qconv_meteo_W"}.issubset(table.columns)
    assert np.isfinite(out["metrics"]["eta_base"]["rmse"])
    assert np.isfinite(out["metrics"]["eta_meteo"]["rmse"])


def test_wind_is_not_calibrable_in_rea_monthly_registry():
    db = default_fluid_database()
    assert "wind_m_s" not in inverse_parameter_options("rea_foz", db)
    assert "wind_m_s" not in inverse_parameter_options("rea_alvorada", db)
    assert "wind_m_s" in inverse_parameter_options("bhambare", db)
