import numpy as np

from defaults import default_fluid_database
from rea_temporal_aggregation_test import (
    build_equal_energy_dni_profile,
    run_temporal_aggregation_hypothesis,
)


def test_profile_preserves_monthly_mean_dni():
    p = build_equal_energy_dni_profile(-25.43816, 7, 146.87, n_bins=7, atmospheric_B=0.15)
    assert len(p) == 7
    assert np.isclose(p["DNI_W_m2"].mean(), 146.87, rtol=0, atol=1e-9)
    assert p["DNI_W_m2"].max() > p["DNI_W_m2"].min()


def test_temporal_aggregation_runs_without_calibration():
    out = run_temporal_aggregation_hypothesis(
        "rea_foz", default_fluid_database(), n_bins=5, atmospheric_B=0.15, months=[7]
    )
    table = out["table"]
    assert len(table) == 1
    assert np.isfinite(table.loc[0, "Eta_DNI_medio_pct"])
    assert np.isfinite(table.loc[0, "Eta_agregacion_temporal_pct"])
    assert np.isclose(
        table.loc[0, "DNI_publicado_W_m2"],
        table.loc[0, "DNI_perfil_medio_W_m2"],
        rtol=0,
        atol=1e-9,
    )
