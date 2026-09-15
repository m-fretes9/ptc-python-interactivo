"""Auditoría diagnóstica del cierre radial Bhambare/Sukhatme.

La prueba no calibra parámetros. Ejecuta el caso nominal una sola vez y luego
recalcula, con las mismas correlaciones externas, qué pérdidas resultarían si
la temperatura del vidrio fuese la publicada por Bhambare o Sukhatme.
También evalúa el cierre aproximado del balance del vidrio con los pares
(Tabs, Tglass) publicados.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from fluid_properties import FluidPropertyEvaluator
from presets import build_bhambare_sukhatme_preset
from ptc_model import (
    PTCSimulator,
    effective_sky_temperature,
    external_convection,
    fourth_power_difference,
)

SIGMA = 5.670374419e-8


def run_bhambare_radial_audit(
    fluid_database: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    cfg, _ = build_bhambare_sukhatme_preset()
    result = PTCSimulator(cfg, fluid_database).simulate()
    k = len(result.t_s) - 1

    g = cfg["geometry"]
    env = cfg["environment"]
    absorber = cfg["materials"]["absorber"]
    glass = cfg["materials"]["glass"]
    op = cfg["operation"]

    Tamb_K = float(env["Tamb_K"])
    Tsky_K = float(effective_sky_temperature(float(op["t_end_s"]), env)["Tsky_K"])
    wind = float(env["wind_m_s"])
    pressure = float(env["pressure_Pa"])
    A5 = np.pi * float(g["D5"]) * float(g["L"])
    A3 = np.pi * float(g["D3"]) * float(g["L"])

    def external_loss(Tg_K: float) -> dict[str, float]:
        conv = external_convection(0.5 * (Tg_K + Tamb_K), wind, float(g["D5"]), pressure)
        qconv = conv["h_W_m2K"] * A5 * (Tg_K - Tamb_K)
        qrad = float(glass["eps"]) * SIGMA * A5 * fourth_power_difference(Tg_K, Tsky_K)
        return {
            "Tglass_K": float(Tg_K),
            "h_ext_W_m2K": float(conv["h_W_m2K"]),
            "Qconv_W": float(qconv),
            "Qrad_sky_W": float(qrad),
            "Qloss_W": float(qconv + qrad),
        }

    denominator = (
        1.0 / float(absorber["eps"])
        + (float(g["D3"]) / float(g["D4"])) * (1.0 / float(glass["eps"]) - 1.0)
    )

    def q_abs_to_glass(Tabs_K: float, Tg_K: float) -> float:
        return float(
            SIGMA
            * A3
            * fourth_power_difference(Tabs_K, Tg_K)
            / max(denominator, np.finfo(float).eps)
        )

    model_Tg_K = float(result.Tglass_mean_C[k] + 273.15)
    model_Tabs_K = float(result.Tabs_mean_C[k] + 273.15)
    model_Tout_C = float(result.Tout_C[k])
    model_Qloss_W = float(result.scalar_diag["Qloss_W"][k])
    model_Qsolar_abs_W = float(result.scalar_diag["QsolarAbs_W"][k])
    Qsolar_glass_W = float(result.scalar_diag["QsolarGlass_W"][k])
    model_Quseful_W = float(result.scalar_diag["Quseful_W"][k])
    model_Qrad_ag_W = float(np.sum(result.node_diag["Qrad_abs_glass_W"][k]))
    model_Qconv_ext_W = float(np.sum(result.node_diag["Qconv_external_W"][k]))
    model_Qrad_sky_W = float(np.sum(result.node_diag["Qrad_sky_W"][k]))

    refs = {
        "Modelo Python": {
            "Tglass_K": model_Tg_K,
            "Tabs_K": model_Tabs_K,
            "Tout_C": model_Tout_C,
            "Qloss_ref_W": model_Qloss_W,
        },
        "Bhambare": {
            "Tglass_K": 331.4,
            "Tabs_K": 465.4,
            "Tout_C": 154.1,
            "Qloss_ref_W": 813.8,
        },
        "Sukhatme": {
            "Tglass_K": 333.39,
            "Tabs_K": 441.13,
            "Tout_C": 155.34,
            "Qloss_ref_W": 857.6,
        },
    }

    rows: list[dict[str, float | str]] = []
    fluid = FluidPropertyEvaluator("ParathermNF", fluid_database)
    Tin_C = float(op["Tin_K"]) - 273.15
    mdot = float(op["mdot"])

    for name, ref in refs.items():
        Tg = float(ref["Tglass_K"])
        Tabs = float(ref["Tabs_K"])
        Tout = float(ref["Tout_C"])
        ext = external_loss(Tg)
        qag = model_Qrad_ag_W if name == "Modelo Python" else q_abs_to_glass(Tabs, Tg)
        qin_glass = qag + Qsolar_glass_W  # anular = vacío ideal
        qout_glass = ext["Qloss_W"]
        residual_glass = qin_glass - qout_glass
        cp_mean = 0.5 * (fluid(Tin_C + 273.15).Cp + fluid(Tout + 273.15).Cp)
        quse = mdot * cp_mean * (Tout - Tin_C)
        qabs_req_qs = quse + qag  # aproximación estacionaria, soportes/anular=0
        rows.append(
            {
                "Escenario": name,
                "Tglass_K": Tg,
                "Tabs_K": Tabs,
                "Tout_C": Tout,
                "h_ext_W_m2K": ext["h_ext_W_m2K"],
                "Qconv_ext_W": ext["Qconv_W"],
                "Qrad_sky_W": ext["Qrad_sky_W"],
                "Qloss_recalculado_W": ext["Qloss_W"],
                "Qloss_referencia_W": float(ref["Qloss_ref_W"]),
                "Error_Qloss_pct": 100.0 * (ext["Qloss_W"] - float(ref["Qloss_ref_W"])) / float(ref["Qloss_ref_W"]),
                "Qrad_abs_glass_W": qag,
                "Qsolar_glass_W": Qsolar_glass_W,
                "Qin_glass_W": qin_glass,
                "Qout_glass_W": qout_glass,
                "Residual_glass_W": residual_glass,
                "Residual_glass_pct_Qout": 100.0 * residual_glass / max(abs(qout_glass), 1e-12),
                "Quseful_derivado_W": quse,
                "Qabs_requerido_qs_W": qabs_req_qs,
                "QsolarAbs_modelo_W": model_Qsolar_abs_W,
                "Ratio_Qabs_req_modelo": qabs_req_qs / model_Qsolar_abs_W,
            }
        )

    scenario_table = pd.DataFrame(rows)

    target_rows: list[dict[str, float | str]] = []
    for name in ("Bhambare", "Sukhatme"):
        target = float(refs[name]["Qloss_ref_W"])
        Tg_req = float(brentq(lambda T: external_loss(T)["Qloss_W"] - target, Tamb_K + 1e-6, 600.0))
        Tg_pub = float(refs[name]["Tglass_K"])
        forced = external_loss(Tg_pub)
        target_rows.append(
            {
                "Referencia": name,
                "Qloss_objetivo_W": target,
                "Tglass_publicada_K": Tg_pub,
                "Tglass_requerida_por_bloque_externo_K": Tg_req,
                "Delta_Tglass_K": Tg_req - Tg_pub,
                "Qloss_a_Tglass_publicada_W": forced["Qloss_W"],
                "Error_Qloss_a_Tglass_publicada_pct": 100.0 * (forced["Qloss_W"] - target) / target,
            }
        )
    target_table = pd.DataFrame(target_rows)

    # Curva para visualización: mismo bloque externo, sólo varía Tglass.
    Tmin = max(Tamb_K + 0.1, min(model_Tg_K, 331.4, 333.39) - 6.0)
    Tmax = max(model_Tg_K, 331.4, 333.39) + 8.0
    Tgrid = np.linspace(Tmin, Tmax, 90)
    curve = pd.DataFrame(
        {
            "Tglass_K": Tgrid,
            "Qloss_W": [external_loss(float(T))["Qloss_W"] for T in Tgrid],
        }
    )

    bh = target_table.loc[target_table["Referencia"] == "Bhambare"].iloc[0]
    su = target_table.loc[target_table["Referencia"] == "Sukhatme"].iloc[0]
    bh_pair = scenario_table.loc[scenario_table["Escenario"] == "Bhambare"].iloc[0]
    su_pair = scenario_table.loc[scenario_table["Escenario"] == "Sukhatme"].iloc[0]

    external_compatible = (
        abs(float(bh["Delta_Tglass_K"])) <= 1.0
        and abs(float(su["Delta_Tglass_K"])) <= 1.0
    )
    bh_radial_compatible = abs(float(bh_pair["Residual_glass_pct_Qout"])) <= 5.0
    suk_radial_compatible = abs(float(su_pair["Residual_glass_pct_Qout"])) <= 10.0

    if external_compatible and bh_radial_compatible and not suk_radial_compatible:
        verdict = "externo_ok_sukhatme_incompatible"
        diagnosis = (
            "El bloque de pérdidas externas reproduce los Qloss publicados si se impone la Tglass publicada, y el par de temperaturas de Bhambare "
            "cierra razonablemente el balance del vidrio. En cambio, el par Tabs/Tglass de Sukhatme no cierra con estas mismas ecuaciones. "
            "La discrepancia principal no apunta al solver ni a la convección/radiación externa, sino al reparto radial/upstream o a diferencias entre el problema del libro y el modelo de Bhambare."
        )
    elif external_compatible:
        verdict = "externo_ok"
        diagnosis = (
            "El bloque externo es compatible con las pérdidas publicadas cuando se impone Tglass. La causa principal de la Tglass demasiado baja debe buscarse aguas arriba del vidrio: "
            "intercambio absorbedor-vidrio, potencia absorbida o transferencia al HTF."
        )
    else:
        verdict = "externo_revisar"
        diagnosis = (
            "Incluso imponiendo las temperaturas de vidrio publicadas, el bloque externo no reproduce suficientemente las pérdidas objetivo. "
            "Conviene revisar primero la formulación de convección/radiación al ambiente."
        )

    return {
        "scenario_table": scenario_table,
        "target_table": target_table,
        "loss_curve": curve,
        "verdict": verdict,
        "diagnosis": diagnosis,
        "metrics": {
            "Tglass_model_K": model_Tg_K,
            "Tglass_error_vs_sukhatme_K": model_Tg_K - 333.39,
            "Qloss_model_W": model_Qloss_W,
            "Qloss_error_vs_sukhatme_pct": 100.0 * (model_Qloss_W - 857.6) / 857.6,
            "Qconv_model_W": model_Qconv_ext_W,
            "Qrad_sky_model_W": model_Qrad_sky_W,
            "Qrad_abs_glass_model_W": model_Qrad_ag_W,
            "Quseful_model_W": model_Quseful_W,
            "QsolarAbs_model_W": model_Qsolar_abs_W,
            "Tsky_K": Tsky_K,
        },
        "note": (
            "La prueba es diagnóstica y no calibra parámetros. Los balances con pares publicados son cuasiestacionarios y usan las mismas correlaciones del código actual. "
            "Bhambare y Sukhatme reportan resultados de modelos/problemas distintos; una incompatibilidad entre ambos no implica por sí sola que una de las fuentes sea incorrecta."
        ),
    }
