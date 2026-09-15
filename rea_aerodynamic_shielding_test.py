"""Prueba diagnóstica de apantallamiento aerodinámico del receptor.

Hipótesis
---------
La correlación de convección externa puede ser adecuada para un cilindro en
flujo cruzado, pero el receiver de un PTC no está necesariamente expuesto a la
velocidad ambiente completa porque la propia calha/reflector modifica el campo
de velocidades alrededor del tubo.

Se introduce un único factor estructural, constante entre meses y ciudades:

    v_eff = S_v * v_amb

Esta prueba NO calibra un viento por mes. Ajusta un solo S_v usando únicamente
Ene/Abr/Jul/Oct de Foz do Iguaçu, lo congela y lo prueba en:

* Foz hold-out: Feb/Mar/May/Jun/Ago/Sep/Nov/Dic.
* Alvorada do Norte: los 12 meses, sin recalibración.

El viento ambiente de esta prueba se mantiene en el baseline documental del
preset Rea (1 m/s). Por tanto, el experimento aísla exclusivamente un posible
factor geométrico/aerodinámico del receptor; no intenta reconstruir la
meteorología real.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from presets import (
    MONTH_ABBR_ES,
    REA_ALVORADA_MONTHLY,
    REA_FOZ_MONTHLY,
    build_rea_monthly_preset,
)
from ptc_model import PTCSimulator


CAL_MONTHS = (1, 4, 7, 10)
HOLDOUT_MONTHS = tuple(m for m in range(1, 13) if m not in CAL_MONTHS)
BASE_AMBIENT_WIND_M_S = 1.0


def _prep_diagnostic(cfg: dict[str, Any]) -> None:
    """Acelera el ensayo sin alterar la estructura física del modelo."""
    cfg["geometry"]["Nseg"] = min(int(cfg["geometry"].get("Nseg", 12)), 6)
    cfg["solver"]["method"] = "BDF"
    cfg["solver"]["max_step_s"] = max(float(cfg["solver"].get("max_step_s", 20.0)), 900.0)
    cfg["operation"]["output_step_s"] = max(float(cfg["operation"].get("output_step_s", 60.0)), 900.0)


def _city_data(city_key: str) -> tuple[str, Mapping[str, Any]]:
    if city_key == "rea_foz":
        return "Foz do Iguaçu", REA_FOZ_MONTHLY
    if city_key == "rea_alvorada":
        return "Alvorada do Norte", REA_ALVORADA_MONTHLY
    raise ValueError(f"Ciudad no soportada: {city_key}")


def _metric(reference: Sequence[float], prediction: Sequence[float]) -> dict[str, float]:
    ref = np.asarray(reference, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    err = pred - ref
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(ref), 1e-12)) * 100.0)
    corr = float(np.corrcoef(ref, pred)[0, 1]) if len(ref) > 1 and np.std(ref) > 0 and np.std(pred) > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "bias": bias, "mape_pct": mape, "corr": corr}


def _improvement_pct(before: float, after: float) -> float:
    if not np.isfinite(before) or abs(before) < 1e-12:
        return float("nan")
    return float(100.0 * (before - after) / before)


def run_aerodynamic_shielding_hypothesis(
    fluid_database: Mapping[str, Mapping[str, Any]],
    *,
    sv_min: float = 0.20,
    sv_max: float = 1.00,
) -> dict[str, Any]:
    """Calibra un único factor S_v en Foz y valida sin reajuste.

    La función objetivo usa exclusivamente la eficiencia de los cuatro meses de
    calibración de Foz. Tout se reporta como variable de comprobación, pero no
    entra de nuevo en la función objetivo porque para las tablas Rea ambas
    magnitudes están fuertemente ligadas por la Ec. (10).
    """
    sv_min = float(sv_min)
    sv_max = float(sv_max)
    if not (0.0 < sv_min < sv_max <= 1.0):
        raise ValueError("Se requiere 0 < sv_min < sv_max <= 1.")

    db0 = deepcopy(fluid_database)
    cache: dict[tuple[str, int, float], dict[str, float]] = {}

    def simulate(city_key: str, month: int, sv: float) -> dict[str, float]:
        sv = float(np.clip(sv, sv_min, sv_max))
        key = (city_key, int(month), round(sv, 8))
        if key in cache:
            return cache[key]
        city_name, data = _city_data(city_key)
        cfg, _ = build_rea_monthly_preset(city_name, int(month))
        _prep_diagnostic(cfg)
        cfg["environment"]["wind_m_s"] = BASE_AMBIENT_WIND_M_S * sv
        result = PTCSimulator(cfg, deepcopy(db0)).simulate()
        k = len(result.t_s) - 1
        out = {
            "eta_pct": float(result.scalar_diag["eta_dni_basis_pct"][k]),
            "tout_C": float(result.Tout_C[k]),
            "qconv_W": float(np.sum(result.node_diag["Qconv_external_W"][k, :])),
            "qrad_W": float(np.sum(result.node_diag["Qrad_sky_W"][k, :])),
            "qsolar_W": float(result.scalar_diag["QsolarAbs_W"][k] + result.scalar_diag["QsolarGlass_W"][k]),
            "quseful_W": float(result.scalar_diag["Quseful_W"][k]),
        }
        cache[key] = out
        return out

    def calibration_rmse(sv: float) -> float:
        preds = [simulate("rea_foz", m, sv)["eta_pct"] for m in CAL_MONTHS]
        refs = [float(REA_FOZ_MONTHLY["eta_ref_pct"][m - 1]) for m in CAL_MONTHS]
        return _metric(refs, preds)["rmse"]

    # Búsqueda determinista de dos etapas. Además de ser robusta, deja una curva
    # objetivo explícita y auditable para la interfaz.
    coarse = np.linspace(sv_min, sv_max, 7)
    coarse_scores = [(float(s), float(calibration_rmse(float(s)))) for s in coarse]
    best_coarse = min(coarse_scores, key=lambda x: x[1])[0]
    step = float(coarse[1] - coarse[0]) if len(coarse) > 1 else 0.1
    lo = max(sv_min, best_coarse - step)
    hi = min(sv_max, best_coarse + step)
    fine = np.linspace(lo, hi, 7)
    factors = sorted({round(float(x), 8) for x in np.concatenate([coarse, fine])})
    trace_rows = []
    for s in factors:
        trace_rows.append({"S_v": s, "RMSE_cal_Foz_pp": float(calibration_rmse(s))})
    trace = pd.DataFrame(trace_rows).sort_values("S_v").reset_index(drop=True)
    idx_best = int(trace["RMSE_cal_Foz_pp"].idxmin())
    sv_star = float(trace.loc[idx_best, "S_v"])

    # Evalúa baseline y factor congelado en Foz y Alvorada.
    rows: list[dict[str, Any]] = []
    for city_key in ("rea_foz", "rea_alvorada"):
        city_name, data = _city_data(city_key)
        for month in range(1, 13):
            base = simulate(city_key, month, 1.0)
            shield = simulate(city_key, month, sv_star)
            subset = (
                "calibración" if city_key == "rea_foz" and month in CAL_MONTHS
                else "hold-out" if city_key == "rea_foz"
                else "validación externa"
            )
            eta_ref = float(data["eta_ref_pct"][month - 1])
            tout_ref = float(data["Tout_ref_C"][month - 1])
            rows.append(
                {
                    "Ciudad": city_name,
                    "Caso": city_key,
                    "Mes": MONTH_ABBR_ES[month - 1],
                    "Mes_num": month,
                    "Conjunto": subset,
                    "S_v": sv_star,
                    "V_amb_baseline_m_s": BASE_AMBIENT_WIND_M_S,
                    "V_eff_apantallado_m_s": BASE_AMBIENT_WIND_M_S * sv_star,
                    "Eta_ref_pct": eta_ref,
                    "Eta_baseline_pct": base["eta_pct"],
                    "Eta_apantallada_pct": shield["eta_pct"],
                    "Error_eta_baseline_pp": base["eta_pct"] - eta_ref,
                    "Error_eta_apantallada_pp": shield["eta_pct"] - eta_ref,
                    "Tout_ref_C": tout_ref,
                    "Tout_baseline_C": base["tout_C"],
                    "Tout_apantallada_C": shield["tout_C"],
                    "Error_Tout_baseline_C": base["tout_C"] - tout_ref,
                    "Error_Tout_apantallada_C": shield["tout_C"] - tout_ref,
                    "Qconv_baseline_W": base["qconv_W"],
                    "Qconv_apantallada_W": shield["qconv_W"],
                    "Qrad_baseline_W": base["qrad_W"],
                    "Qrad_apantallada_W": shield["qrad_W"],
                    "Qsolar_baseline_W": base["qsolar_W"],
                    "Qsolar_apantallada_W": shield["qsolar_W"],
                    "Quseful_baseline_W": base["quseful_W"],
                    "Quseful_apantallada_W": shield["quseful_W"],
                }
            )

    table = pd.DataFrame(rows)

    def metrics_for(city: str, subset: str | None = None) -> dict[str, Any]:
        df = table[table["Caso"] == city].copy()
        if subset is not None:
            df = df[df["Conjunto"] == subset]
        eta_b = _metric(df["Eta_ref_pct"], df["Eta_baseline_pct"])
        eta_s = _metric(df["Eta_ref_pct"], df["Eta_apantallada_pct"])
        tout_b = _metric(df["Tout_ref_C"], df["Tout_baseline_C"])
        tout_s = _metric(df["Tout_ref_C"], df["Tout_apantallada_C"])
        return {
            "eta_baseline": eta_b,
            "eta_shielded": eta_s,
            "tout_baseline": tout_b,
            "tout_shielded": tout_s,
            "eta_rmse_improvement_pct": _improvement_pct(eta_b["rmse"], eta_s["rmse"]),
            "tout_rmse_improvement_pct": _improvement_pct(tout_b["rmse"], tout_s["rmse"]),
        }

    metrics = {
        "foz_cal": metrics_for("rea_foz", "calibración"),
        "foz_holdout": metrics_for("rea_foz", "hold-out"),
        "foz_all": metrics_for("rea_foz", None),
        "alvorada_external": metrics_for("rea_alvorada", "validación externa"),
    }

    imp_f = metrics["foz_holdout"]["eta_rmse_improvement_pct"]
    imp_a = metrics["alvorada_external"]["eta_rmse_improvement_pct"]
    near_bound = (sv_star - sv_min) <= 0.03 * (sv_max - sv_min) or (sv_max - sv_star) <= 0.03 * (sv_max - sv_min)

    if imp_f >= 10.0 and imp_a >= 10.0 and not near_bound:
        verdict = "apoya"
        diagnosis = (
            "Un único factor de apantallamiento, calibrado solo con cuatro meses de Foz, mejora al menos 10 % el RMSE "
            "tanto en el hold-out de Foz como en Alvorada sin reajuste. La hipótesis aerodinámica merece incorporarse como "
            "corrección estructural candidata y someterse a validación adicional."
        )
    elif imp_f > 0.0 and imp_a > 0.0:
        verdict = "parcial"
        diagnosis = (
            "El mismo factor mejora ambas ciudades, pero la ganancia fuera de muestra no supera 10 % en los dos conjuntos. "
            "La hipótesis es plausible pero todavía no explica por sí sola la discrepancia."
        )
    elif imp_f >= 10.0 and imp_a <= 0.0:
        verdict = "rechaza_transferencia"
        diagnosis = (
            "El factor mejora claramente el hold-out de Foz pero no generaliza a Alvorada. Eso indica que estaría actuando "
            "como corrección específica de Foz, no como propiedad aerodinámica universal del colector."
        )
    else:
        verdict = "rechaza"
        diagnosis = (
            "Un único factor de apantallamiento no mejora de forma consistente la validación fuera de muestra. La convección "
            "no puede corregirse de manera defendible con una reducción geométrica global de la velocidad."
        )

    return {
        "table": table,
        "objective_trace": trace,
        "S_v_star": sv_star,
        "V_eff_star_m_s": BASE_AMBIENT_WIND_M_S * sv_star,
        "calibration_months": [MONTH_ABBR_ES[m - 1] for m in CAL_MONTHS],
        "holdout_months": [MONTH_ABBR_ES[m - 1] for m in HOLDOUT_MONTHS],
        "metrics": metrics,
        "verdict": verdict,
        "diagnosis": diagnosis,
        "near_bound": bool(near_bound),
        "settings": {"sv_min": sv_min, "sv_max": sv_max, "ambient_wind_baseline_m_s": BASE_AMBIENT_WIND_M_S},
        "note": (
            "Prueba controlada. S_v se identifica únicamente con Ene/Abr/Jul/Oct de Foz. Después queda congelado. "
            "Foz hold-out y los 12 meses de Alvorada nunca participan del ajuste. El viento ambiente se mantiene en el "
            "baseline de 1 m/s del preset para aislar únicamente el posible apantallamiento geométrico del receptor."
        ),
    }
