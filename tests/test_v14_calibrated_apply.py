import pandas as pd

from defaults import default_fluid_database
from presets import build_preset
from validations import apply_calibrated_parameters


def test_apply_calibrated_parameters_to_active_project():
    cfg, db = build_preset("rea_foz_monthly", month=1)
    calibration = {
        "case": "rea_foz",
        "score_before_pct": 10.0,
        "score_after_pct": 5.0,
        "parameter_table": pd.DataFrame(
            [
                {"ID": "eta_opt_eff", "Identificado": 0.60},
                {"ID": "wind_m_s", "Identificado": 1.25},
            ]
        ),
    }
    apply_calibrated_parameters(cfg, db, calibration)
    assert abs(cfg["environment"]["wind_m_s"] - 1.25) < 1e-12
    assert cfg["calibration_meta"]["case"] == "rea_foz"
    assert cfg["calibration_meta"]["parameters"]["eta_opt_eff"] == 0.60
