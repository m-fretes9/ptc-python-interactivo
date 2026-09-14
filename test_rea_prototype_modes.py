from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presets import (
    REA_PROTOTYPE_USER_EXPORT_2026_09_14,
    build_rea_prototype_preset,
)
from ptc_model import PTCSimulator
from validations import prototype_user_export_template, validate_rea_prototype_mode


def test_trnsys_published_optics_are_nominal():
    cfg, db = build_rea_prototype_preset("trnsys_published")
    sim = PTCSimulator(cfg, db)
    solar = sim.solar_model(12.0 * 3600.0)
    assert cfg["solar"]["mode"] == "constante"
    assert abs(solar["DNI_W_m2"] - 905.0) < 1e-12
    assert abs(solar["theta_deg"]) < 1e-12
    assert abs(solar["IAM"] - 1.0) < 1e-12
    assert abs(solar["endLoss"] - 1.0) < 1e-12


def test_physical_fixed_mode_changes_incidence_hourly():
    cfg, db = build_rea_prototype_preset("physical_fixed_ns", dni_source="nominal")
    sim = PTCSimulator(cfg, db)
    s9 = sim.solar_model(9.0 * 3600.0)
    s12 = sim.solar_model(12.0 * 3600.0)
    assert cfg["solar"]["mode"] == "fijo_horizontal"
    assert s9["DNI_W_m2"] == 905.0
    assert s12["DNI_W_m2"] == 905.0
    assert s9["theta_deg"] > s12["theta_deg"]
    assert not np.isclose(s9["IAM"], s12["IAM"])


def test_user_csv_template_is_exact_and_validation_runs():
    template = prototype_user_export_template()["table"]
    assert len(template) == 8
    assert abs(template.iloc[0]["Eta_modelo_inicial_CSV_pct"] - 54.458875343535006) < 1e-12
    assert abs(template.iloc[-1]["Eta_modelo_identificado_CSV_pct"] - 28.889673058221174) < 1e-12
    assert template["Eta_referencia_pct"].tolist() == REA_PROTOTYPE_USER_EXPORT_2026_09_14["eta_reference_pct"]

    cfg, db = build_rea_prototype_preset("trnsys_published")
    out = validate_rea_prototype_mode(
        "trnsys_published",
        db,
        target_key="csv_initial",
        dni_source="nominal",
        config=cfg,
    )
    assert len(out["table"]) == 8
    assert np.isfinite(out["metrics"]["RMSE_pp"])
