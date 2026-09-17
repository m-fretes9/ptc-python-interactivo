import math

from defaults import default_config, default_fluid_database
from fluid_properties import FluidPropertyEvaluator
from presets import build_rea_monthly_preset
from ptc_model import internal_convection


def _foz_january_inputs():
    cfg, _ = build_rea_monthly_preset("foz", 1)
    db = default_fluid_database()
    fluid = FluidPropertyEvaluator("Agua", db)
    Tin = float(cfg["operation"]["Tin_K"])
    prop = fluid(Tin)
    prop_wall = fluid(Tin + 10.0)
    return cfg, prop, prop_wall


def test_main_default_uses_automatic_local_hausen():
    cfg = default_config()
    assert cfg["model"]["internal_correlation"] == "automatica_hausen"
    assert math.isclose(float(cfg["model"]["thermal_entrance_factor"]), 0.05)


def test_local_hausen_is_stronger_near_inlet_and_436_after_lth():
    cfg, prop, prop_wall = _foz_january_inputs()
    mdot = float(cfg["operation"]["mdot"])
    D = float(cfg["geometry"]["D2"])

    near = internal_convection(
        mdot, D, prop, prop_wall, "automatica_hausen",
        x_start_m=0.0, x_end_m=0.10,
    )
    assert near["Re"] < 2300.0
    assert near["Nu"] > 4.36
    assert near["thermal_entrance_length_m"] > 0.10

    lth = near["thermal_entrance_length_m"]
    developed = internal_convection(
        mdot, D, prop, prop_wall, "automatica_hausen",
        x_start_m=lth + 0.10, x_end_m=lth + 0.20,
    )
    assert math.isclose(developed["Nu"], 4.36, rel_tol=0.0, abs_tol=1e-12)
    assert developed["x_over_Lth"] > 1.0


def test_fv_segment_average_recovers_hausen_cumulative_mean_inside_entrance():
    cfg, prop, prop_wall = _foz_january_inputs()
    mdot = float(cfg["operation"]["mdot"])
    D = float(cfg["geometry"]["D2"])
    dx = 0.10

    seg1 = internal_convection(
        mdot, D, prop, prop_wall, "automatica_hausen",
        x_start_m=0.0, x_end_m=dx,
    )
    seg2 = internal_convection(
        mdot, D, prop, prop_wall, "automatica_hausen",
        x_start_m=dx, x_end_m=2.0 * dx,
    )
    mean_from_segments = (seg1["Nu"] * dx + seg2["Nu"] * dx) / (2.0 * dx)

    global_hausen_to_x = internal_convection(
        mdot, D, prop, prop_wall, "hausen_laminar",
        characteristic_length_m=2.0 * dx,
    )
    assert math.isclose(mean_from_segments, global_hausen_to_x["Nu"], rel_tol=1e-12, abs_tol=1e-12)
