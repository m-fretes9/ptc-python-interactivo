from defaults import default_fluid_database
from bhambare_radial_audit import run_bhambare_radial_audit


def test_bhambare_radial_audit_closure():
    out = run_bhambare_radial_audit(default_fluid_database())
    assert out["verdict"] == "externo_ok_sukhatme_incompatible"
    targets = out["target_table"].set_index("Referencia")
    assert abs(float(targets.loc["Bhambare", "Delta_Tglass_K"])) < 1.0
    assert abs(float(targets.loc["Sukhatme", "Delta_Tglass_K"])) < 1.0
    scenarios = out["scenario_table"].set_index("Escenario")
    assert abs(float(scenarios.loc["Bhambare", "Residual_glass_pct_Qout"])) < 5.0
    assert abs(float(scenarios.loc["Sukhatme", "Residual_glass_pct_Qout"])) > 20.0
