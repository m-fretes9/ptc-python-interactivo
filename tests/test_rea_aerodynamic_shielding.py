from defaults import default_fluid_database
from rea_aerodynamic_shielding_test import run_aerodynamic_shielding_hypothesis


def test_shielding_uses_one_factor_and_keeps_holdout_separate():
    out = run_aerodynamic_shielding_hypothesis(default_fluid_database(), sv_min=0.4, sv_max=1.0)
    assert 0.4 <= out["S_v_star"] <= 1.0
    table = out["table"]
    foz = table[table["Caso"] == "rea_foz"]
    alv = table[table["Caso"] == "rea_alvorada"]
    assert len(foz) == 12
    assert len(alv) == 12
    assert set(foz[foz["Conjunto"] == "calibración"]["Mes"]) == {"Ene", "Abr", "Jul", "Oct"}
    assert len(foz[foz["Conjunto"] == "hold-out"]) == 8
    assert (alv["S_v"] == out["S_v_star"]).all()
