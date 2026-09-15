import math

from defaults import default_fluid_database
from fluid_properties import FluidPropertyEvaluator
from presets import build_rea_monthly_preset
from ptc_model import internal_convection


def test_rea_water_is_laminar_and_developing_correlations_differ():
    cfg, _ = build_rea_monthly_preset("foz", 1)
    db = default_fluid_database()
    fluid = FluidPropertyEvaluator("Agua", db)
    Tin = float(cfg["operation"]["Tin_K"])
    prop = fluid(Tin)
    prop_wall = fluid(Tin + 10.0)
    mdot = float(cfg["operation"]["mdot"])
    D = float(cfg["geometry"]["D2"])
    L = float(cfg["geometry"]["L"])

    re = 4.0 * mdot / (math.pi * D * prop.mu)
    assert re < 2300.0

    fully = internal_convection(mdot, D, prop, prop_wall, "laminar_436_forzado", characteristic_length_m=L)
    hausen = internal_convection(mdot, D, prop, prop_wall, "hausen_laminar", characteristic_length_m=L)
    sieder = internal_convection(mdot, D, prop, prop_wall, "sieder_tate_laminar", characteristic_length_m=L)

    assert abs(fully["Nu"] - 4.36) < 1e-12
    assert hausen["Nu"] > fully["Nu"]
    assert sieder["Nu"] > fully["Nu"]
    assert fully["Graetz"] > 0.0
