"""Casos de comparación y validación reproducidos desde el código MATLAB."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np
import pandas as pd

from fluid_properties import FluidPropertyEvaluator
from ptc_model import PTCSimulator, SimulationResult


def validate_bhambare(
    base_config: Mapping[str, Any],
    fluid_database: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    cfg = deepcopy(base_config)
    cfg["geometry"]["Nseg"] = 12
    cfg["operation"].update(
        {
            "name": "Validación Bhambare/Sukhatme",
            "fluid": "ParathermNF",
            "mdot": 0.0986,
            "Tin_K": 150.0 + 273.15,
            "t_start_s": 0.0,
            "t_end_s": 8.0 * 3600.0,
            "output_step_s": 120.0,
        }
    )
    cfg["environment"].update({"Tamb_K": 31.9 + 273.15, "wind_m_s": 5.3})
    cfg["solar"].update(
        {
            "mode": "constante",
            "DNI_constant_W_m2": 705.0,
            "angle_constant_deg": 0.0,
        }
    )
    cfg["model"].update(
        {
            "has_glass": True,
            "annulus": "vacio_ideal",
            "internal_correlation": "dittusboelter_forzado",
        }
    )
    cfg["solver"]["max_step_s"] = 20.0

    result = PTCSimulator(cfg, fluid_database).simulate()
    k = len(result.t_s) - 1

    Tref_glass_K = 333.39
    Tref_abs_K = 441.13
    Tref_out_C = 155.34
    Qref_loss_W = 857.6
    properties = FluidPropertyEvaluator("ParathermNF", fluid_database)
    prop_mean_ref = properties(0.5 * (150.0 + Tref_out_C) + 273.15)
    Qref_useful_W = cfg["operation"]["mdot"] * prop_mean_ref.Cp * (Tref_out_C - 150.0)
    eta_ref_pct = 100.0 * Qref_useful_W / (
        705.0 * cfg["geometry"]["W"] * cfg["geometry"]["L"]
    )

    names = [
        "T_vidrio_K",
        "T_absorbedor_K",
        "T_salida_C",
        "Q_perdidas_W",
        "Q_util_derivado_W",
        "eta_derivada_pct",
    ]
    reference = np.array(
        [Tref_glass_K, Tref_abs_K, Tref_out_C, Qref_loss_W, Qref_useful_W, eta_ref_pct],
        dtype=float,
    )
    article_model = np.array([331.4, 465.4, 154.1, 813.8, np.nan, np.nan], dtype=float)
    simulation = np.array(
        [
            result.Tglass_mean_C[k] + 273.15,
            result.Tabs_mean_C[k] + 273.15,
            result.Tout_C[k],
            result.scalar_diag["Qloss_W"][k],
            result.scalar_diag["Quseful_W"][k],
            result.scalar_diag["eta_pct"][k],
        ],
        dtype=float,
    )
    error_pct = 100.0 * np.abs(simulation - reference) / np.maximum(np.abs(reference), np.finfo(float).eps)
    table = pd.DataFrame(
        {
            "Magnitud": names,
            "Referencia": reference,
            "Modelo_articulo": article_model,
            "Modelo_Python": simulation,
            "Error_rel_pct": error_pct,
        }
    )
    return {
        "table": table,
        "result": result,
        "note": "Q_util y eta de referencia se derivan de Tout, mdot y Cp(T); no están tabulados directamente.",
    }


def validate_tcc_monthly(
    base_config: Mapping[str, Any],
    fluid_database: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    months = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    Tin_C = np.array([34.73, 34.06, 34.17, 34.02, 32.89, 32.71, 32.31, 33.07, 31.74, 32.57, 34.60, 34.92])
    Tout_ref_C = np.array([41.95, 40.91, 41.25, 40.71, 38.57, 38.11, 37.57, 38.46, 36.07, 38.25, 41.96, 42.48])
    Tamb_C = np.array([27.46, 25.13, 24.27, 24.56, 20.24, 18.79, 17.14, 17.80, 18.33, 23.93, 24.12, 25.83])
    DNI = np.array([252.38, 224.26, 228.56, 224.70, 174.61, 164.10, 146.87, 181.28, 120.74, 157.03, 247.59, 262.18])
    mdot = np.array([0.0087, 0.0080, 0.0082, 0.0076, 0.0063, 0.0065, 0.0060, 0.0067, 0.0054, 0.0064, 0.0087, 0.0090])
    eta_ref_pct = np.array([47.63, 46.25, 48.14, 43.04, 38.97, 40.55, 40.54, 37.98, 36.50, 43.86, 49.25, 49.41])

    Tout_sim_C = np.full(12, np.nan)
    Quseful_sim_W = np.full(12, np.nan)
    Qloss_sim_W = np.full(12, np.nan)
    eta_sim_pct = np.full(12, np.nan)
    Quseful_ref_W = np.full(12, np.nan)
    water = FluidPropertyEvaluator("Agua", fluid_database)

    for m in range(12):
        cfg = deepcopy(base_config)
        cfg["geometry"].update(
            {
                "W": 1.10,
                "L": 2.00,
                "f": 0.16,
                "D2": 0.039,
                "D3": 0.042,
                "D4": 0.050,
                "D5": 0.055,
                "Nseg": 10,
            }
        )
        cfg["materials"]["absorber"].update({"alpha": 0.97, "eps": 0.90})
        cfg["optics"].update({"reflectivity": 0.85, "intercept_factor": 0.95})
        cfg["model"].update({"has_glass": False, "internal_correlation": "automatica"})
        cfg["operation"].update(
            {
                "name": f"TCC mensual - {months[m]}",
                "fluid": "Agua",
                "mdot": float(mdot[m]),
                "Tin_K": float(Tin_C[m] + 273.15),
                "t_start_s": 0.0,
                "t_end_s": 4.0 * 3600.0,
                "output_step_s": 120.0,
            }
        )
        cfg["environment"].update({"Tamb_K": float(Tamb_C[m] + 273.15), "wind_m_s": 1.0})
        cfg["solar"].update(
            {
                "mode": "constante",
                "DNI_constant_W_m2": float(DNI[m]),
                "angle_constant_deg": 0.0,
            }
        )
        cfg["solver"]["max_step_s"] = 20.0

        result = PTCSimulator(cfg, fluid_database).simulate()
        k = len(result.t_s) - 1
        Tout_sim_C[m] = result.Tout_C[k]
        Quseful_sim_W[m] = result.scalar_diag["Quseful_W"][k]
        Qloss_sim_W[m] = result.scalar_diag["Qloss_W"][k]
        eta_sim_pct[m] = result.scalar_diag["eta_pct"][k]
        prop_ref = water(0.5 * (Tin_C[m] + Tout_ref_C[m]) + 273.15)
        Quseful_ref_W[m] = mdot[m] * prop_ref.Cp * (Tout_ref_C[m] - Tin_C[m])

    err_Tout_pct = 100.0 * np.abs(Tout_sim_C - Tout_ref_C) / np.maximum(np.abs(Tout_ref_C), np.finfo(float).eps)
    err_Quseful_pct = 100.0 * np.abs(Quseful_sim_W - Quseful_ref_W) / np.maximum(np.abs(Quseful_ref_W), np.finfo(float).eps)
    err_eta_pct = 100.0 * np.abs(eta_sim_pct - eta_ref_pct) / np.maximum(np.abs(eta_ref_pct), np.finfo(float).eps)

    table = pd.DataFrame(
        {
            "Mes": months,
            "Tin_C": Tin_C,
            "Tout_ref_C": Tout_ref_C,
            "Tout_sim_C": Tout_sim_C,
            "Err_Tout_pct": err_Tout_pct,
            "Tamb_C": Tamb_C,
            "DNI_W_m2": DNI,
            "mdot_kg_s": mdot,
            "Qutil_ref_W": Quseful_ref_W,
            "Qutil_sim_W": Quseful_sim_W,
            "Qloss_sim_W": Qloss_sim_W,
            "Eta_ref_pct": eta_ref_pct,
            "Eta_sim_pct": eta_sim_pct,
            "Err_Qutil_pct": err_Quseful_pct,
            "Err_Eta_pct": err_eta_pct,
        }
    )
    metrics = {
        "MAPE_Tout_pct": float(np.mean(err_Tout_pct)),
        "MAPE_Qutil_pct": float(np.mean(err_Quseful_pct)),
        "MAPE_eta_pct": float(np.mean(err_eta_pct)),
    }
    return {
        "table": table,
        "metrics": metrics,
        "note": "La Tabla 10 suministra promedios mensuales; cada mes se modela como estado cuasiestacionario.",
    }


def prototype_tcc_table() -> dict[str, Any]:
    hours = np.arange(9, 17)
    eta_exp = np.array([28.70, 34.40, 26.51, 30.06, 27.50, 30.01, 28.60, 27.10])
    eta_trnsys = np.array([27.400, 27.300, 26.500, 26.200, 35.300, 32.900, 32.100, 30.000])
    difference_pp = np.abs(eta_trnsys - eta_exp)
    error_relative_pct = 100.0 * difference_pp / np.maximum(np.abs(eta_exp), np.finfo(float).eps)
    rmse = float(np.sqrt(np.mean((eta_trnsys - eta_exp) ** 2)))
    mape = float(np.mean(error_relative_pct))
    table = pd.DataFrame(
        {
            "Hora": [f"{hour:02d}:00" for hour in hours],
            "Eta_experimental_pct": eta_exp,
            "Eta_TRNSYS_pct": eta_trnsys,
            "Diferencia_pp": difference_pp,
            "Error_rel_pct": error_relative_pct,
        }
    )
    return {
        "table": table,
        "RMSE_pp": rmse,
        "MAPE_pct": mape,
        "note": "No se fuerza una simulación propia porque la fuente no informa Tin por hora.",
    }
