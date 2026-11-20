"""Pruebas de campaña real T1/T2 sin lectura de otros sensores."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from defaults import default_config, default_fluid_database
import experimental_validation as exp_validation
from experimental_validation import (
    apply_measurements, load_bundled_measurements, normalise_measurements,
    run_experimental_validation,
)
from presets import build_preset
from ptc_model import PTCSimulator


def test_bundled_experiment_uses_495_thermocouple_records_only():
    df = load_bundled_measurements()
    assert len(df) == 495
    assert list(df.columns) == ["datetime_local", "DATE", "TIME", "t_s", "Tin_exp_C", "Tout_exp_C"]
    assert df.iloc[0]["TIME"] == "10:04:48"
    assert df.iloc[-1]["TIME"] == "14:12:20"
    assert np.isclose(df.iloc[0]["Tin_exp_C"], 40.36)
    assert np.isclose(df.iloc[0]["Tout_exp_C"], 57.39)
    assert int((df["Tout_exp_C"] < df["Tin_exp_C"]).sum()) == 12


def test_ignores_radiation_and_other_columns_and_selects_correct_t1():
    raw = pd.DataFrame({
        "DATE": ["16/11/2026", "16/11/2026"],
        "TIME": ["10:00:00", "10:00:30"],
        "radiação": [1e10, -1e10],
        "temopar_1": [None, None],
        "termopar_1": ["40,10", "40,50"],
        "termopar_2": ["46,20", "46,50"],
        "vazao_REG1": ["0,00", "100,00"],
    })
    df = normalise_measurements(raw)
    assert np.allclose(df["Tin_exp_C"], [40.1, 40.5])
    assert np.allclose(df["Tout_exp_C"], [46.2, 46.5])
    assert "radiação" not in df.columns and "vazao_REG1" not in df.columns


def test_experimental_preset_does_not_modify_existing_presets():
    cfg, db = build_preset("experimental_foz_2026_11_16")
    assert cfg["solar"]["mode"] == "fijo_horizontal"
    assert cfg["solar"]["fixed_dni_source"] == "clear_sky"
    assert cfg["solar"]["date_origin"] == "2026-11-16"
    assert cfg["solar"]["clock_time_correction"] is True
    assert cfg["solver"]["method"] == "BDF"
    assert len(cfg["operation"]["output_times_s"]) == 495
    foz, _ = build_preset("rea_foz_monthly", month=1)
    assert "Tin_profile" not in foz["operation"]
    assert foz["solar"]["mode"] == "constante"
    assert cfg["model"]["has_glass"] is False


def test_input_interpolation_and_solarmodel_date_rollover():
    raw = pd.DataFrame({
        "DATE": ["16/11/2026", "17/11/2026"],
        "TIME": ["23:59:30", "00:00:30"],
        "termopar_1": ["40", "44"],
        "termopar_2": ["48", "50"],
    })
    cfg = default_config()
    cfg["solar"].update({"mode": "fijo_horizontal", "latitude_deg": -25.516,
                          "longitude_deg": -54.585, "fixed_dni_source": "clear_sky",
                          "clock_time_correction": True, "utc_offset_h": -3})
    cfg = apply_measurements(cfg, normalise_measurements(raw), source="test.csv")
    sim = PTCSimulator(cfg, default_fluid_database())
    assert abs(sim.inlet_temperature_K((cfg["operation"]["t_start_s"] + cfg["operation"]["t_end_s"]) / 2) - (42 + 273.15)) < 1e-8
    s1 = sim.solar_model(12*3600)
    s2 = sim.solar_model(36*3600)
    assert s1["DNI_W_m2"] != s2["DNI_W_m2"]


def test_full_real_validation_time_alignment():
    cfg, db = build_preset("experimental_foz_2026_11_16")
    report = run_experimental_validation(cfg, db)
    table = report["table"]
    assert len(table) == 495
    assert report["metrics"]["n"] == 495
    assert np.allclose(table["Tin_exp_C"], np.array(cfg["experimental_validation"]["Tin_exp_C"]))
    assert np.allclose(table["Tout_exp_C"], np.array(cfg["experimental_validation"]["Tout_exp_C"]))
    assert np.all(np.isfinite(table["DNI_synthetic_W_m2"]))
    assert report["metrics"]["RMSE_Tout_C"] > 0
    assert "radiação" not in table.columns


def test_bundled_experiment_survives_missing_csv_on_streamlit_cloud(monkeypatch, tmp_path):
    """En GitHub/Streamlit Cloud puede faltar data/ pero el preset no cae."""
    original = load_bundled_measurements()
    monkeypatch.setattr(exp_validation, "BUNDLED_FOZ_CSV", tmp_path / "missing_data.csv")
    loaded = load_bundled_measurements()
    pd.testing.assert_frame_equal(loaded, original)
    assert len(loaded) == 495
    cfg, fluids = build_preset("experimental_foz_2026_11_16")
    assert len(cfg["experimental_validation"]["TIME"]) == 495
    assert cfg["geometry"]["L"] == 2.0
    assert cfg["geometry"]["W"] == 1.10
    assert cfg["geometry"]["D2"] == 0.039
    assert cfg["geometry"]["D3"] == 0.042
    assert cfg["geometry"]["f"] == 0.16
    assert cfg["geometry"]["Nseg"] == 12
    assert cfg["model"]["has_glass"] is False
    assert "Rea Quille" in cfg["preset_meta"]["geometry_source"]

