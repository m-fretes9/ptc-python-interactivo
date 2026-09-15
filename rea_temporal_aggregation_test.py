"""Prueba diagnóstica de agregación temporal para Rea Quille mensual.

Hipótesis
---------
Las Tablas 10/11 contienen promedios mensuales, pero el colector es no lineal.
Por tanto, evaluar el modelo una sola vez con el DNI medio puede diferir de
promediar la respuesta del modelo a un perfil horario cuya energía total sea la
misma.

Esta prueba aísla SOLO ese efecto. Para cada mes:

1. Ejecuta el caso mensual actual con DNI constante igual al publicado.
2. Construye un día solar representativo con una forma de cielo claro basada en
   la geometría solar.
3. Escala el perfil para que su media durante las horas de sol sea exactamente
   el DNI mensual publicado.
4. Resuelve varios estados cuasiestacionarios a lo largo del día manteniendo
   Tin, Tamb, caudal, viento y todas las constantes del colector sin cambios.
5. Integra/promedia Qutil y calcula

       eta_temporal = 100 * <Qutil> / (A_apertura * <DNI>)

No intenta reconstruir el TRNSYS hora a hora y no calibra ningún parámetro. Es
una prueba controlada de ``f(promedio)`` frente a ``promedio(f)``.
"""
from __future__ import annotations

from calendar import monthrange
from copy import deepcopy
from datetime import date
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from presets import MONTH_ABBR_ES, REA_ALVORADA_MONTHLY, REA_FOZ_MONTHLY, build_rea_monthly_preset
from ptc_model import PTCSimulator


CITY_INFO = {
    "rea_foz": {"name": "Foz do Iguaçu", "latitude_deg": -25.43816},
    "rea_alvorada": {"name": "Alvorada do Norte", "latitude_deg": -14.600},
}


def _case_info(case: str) -> tuple[str, str, Mapping[str, Any], float]:
    key = str(case).strip().lower()
    if key in {"rea_foz", "foz", "foz do iguaçu", "foz do iguacu"}:
        return "rea_foz", "Foz do Iguaçu", REA_FOZ_MONTHLY, CITY_INFO["rea_foz"]["latitude_deg"]
    if key in {"rea_alvorada", "alvorada", "alvorada do norte"}:
        return "rea_alvorada", "Alvorada do Norte", REA_ALVORADA_MONTHLY, CITY_INFO["rea_alvorada"]["latitude_deg"]
    raise ValueError(f"Caso no soportado: {case}")


def _midmonth_doy(month: int, year: int = 2021) -> int:
    day = min(15, monthrange(year, month)[1])
    return date(year, month, day).timetuple().tm_yday


def _solar_declination_deg(day_of_year: int) -> float:
    return float(23.45 * np.sin(np.deg2rad(360.0 * (284.0 + day_of_year) / 365.0)))


def _sunrise_sunset_solar_time(latitude_deg: float, day_of_year: int) -> tuple[float, float]:
    phi = np.deg2rad(float(latitude_deg))
    delta = np.deg2rad(_solar_declination_deg(day_of_year))
    arg = -np.tan(phi) * np.tan(delta)
    arg = float(np.clip(arg, -1.0, 1.0))
    omega_s_deg = float(np.rad2deg(np.arccos(arg)))
    half_day_h = omega_s_deg / 15.0
    return 12.0 - half_day_h, 12.0 + half_day_h


def build_equal_energy_dni_profile(
    latitude_deg: float,
    month: int,
    target_mean_dni_W_m2: float,
    *,
    n_bins: int = 9,
    atmospheric_B: float = 0.15,
) -> pd.DataFrame:
    """Crea un perfil de DNI con la misma media diurna que el valor publicado.

    Se usan bins de igual duración entre salida y puesta del sol. La forma base
    es ``exp(-B/cos(z))``; luego se reescala para que la media discreta sea
    exactamente ``target_mean_dni_W_m2``. El ángulo de incidencia NO se cambia:
    esta prueba modifica solo la distribución temporal del DNI.
    """
    if int(n_bins) < 5 or int(n_bins) > 25:
        raise ValueError("n_bins debe estar entre 5 y 25.")
    B = float(atmospheric_B)
    if not 0.0 <= B <= 1.0:
        raise ValueError("atmospheric_B debe estar entre 0 y 1.")
    target = float(target_mean_dni_W_m2)
    if target <= 0.0:
        raise ValueError("El DNI medio objetivo debe ser positivo.")

    doy = _midmonth_doy(int(month))
    sunrise, sunset = _sunrise_sunset_solar_time(float(latitude_deg), doy)
    edges = np.linspace(sunrise, sunset, int(n_bins) + 1)
    times = 0.5 * (edges[:-1] + edges[1:])

    phi = np.deg2rad(float(latitude_deg))
    delta = np.deg2rad(_solar_declination_deg(doy))
    omega = np.deg2rad(15.0 * (12.0 - times))
    cos_z = np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.cos(omega)
    cos_z = np.clip(cos_z, 1e-8, 1.0)

    raw = np.exp(-B / cos_z)
    raw_mean = float(np.mean(raw))
    if raw_mean <= 0.0 or not np.isfinite(raw_mean):
        raise RuntimeError("No fue posible construir el perfil solar representativo.")
    dni = target * raw / raw_mean

    return pd.DataFrame(
        {
            "Hora_solar_h": times,
            "cos_z": cos_z,
            "DNI_W_m2": dni,
            "Peso": np.full(len(times), 1.0 / len(times)),
            "DNI_medio_objetivo_W_m2": target,
        }
    )


def _prep_diagnostic(cfg: dict[str, Any]) -> None:
    # La prueba necesita muchos estados cuasiestacionarios. N=6 conserva bien la
    # referencia numérica de las validaciones anteriores y reduce el costo; no modifica el modelo guardado.
    cfg["geometry"]["Nseg"] = min(int(cfg["geometry"].get("Nseg", 12)), 6)
    cfg["solver"]["method"] = "BDF"
    cfg["solver"]["max_step_s"] = max(float(cfg["solver"].get("max_step_s", 20.0)), 900.0)
    cfg["operation"]["output_step_s"] = max(float(cfg["operation"].get("output_step_s", 60.0)), 900.0)


def _metric(reference: Sequence[float], prediction: Sequence[float]) -> dict[str, float]:
    ref = np.asarray(reference, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    err = pred - ref
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(ref), 1e-12)) * 100.0)
    corr = float(np.corrcoef(ref, pred)[0, 1]) if np.std(ref) > 0 and np.std(pred) > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "bias": bias, "mape_pct": mape, "corr": corr}


def run_temporal_aggregation_hypothesis(
    case: str,
    fluid_database: Mapping[str, Mapping[str, Any]],
    *,
    n_bins: int = 9,
    atmospheric_B: float = 0.15,
    months: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Compara entrada mensual media vs promedio de respuestas cuasiestacionarias."""
    case_key, city_name, data, latitude = _case_info(case)
    month_numbers = list(range(1, 13)) if months is None else [int(m) for m in months]
    if not month_numbers or any(m < 1 or m > 12 for m in month_numbers):
        raise ValueError("months debe contener valores entre 1 y 12.")

    rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    db0 = deepcopy(fluid_database)

    for month in month_numbers:
        i = month - 1
        cfg_const, _ = build_rea_monthly_preset(city_name, month)
        _prep_diagnostic(cfg_const)
        # Mantener exactamente la interpretación mensual histórica: DNI
        # constante, incidencia normal. Solo se compara contra una distribución
        # temporal con la MISMA media de DNI.
        cfg_const["solar"]["mode"] = "constante"
        cfg_const["solar"]["DNI_constant_W_m2"] = float(data["DNI_W_m2"][i])
        cfg_const["solar"]["angle_constant_deg"] = 0.0
        base = PTCSimulator(cfg_const, deepcopy(db0)).simulate()
        kb = len(base.t_s) - 1

        eta_const = float(base.scalar_diag["eta_dni_basis_pct"][kb])
        tout_const = float(base.Tout_C[kb])
        q_const = float(base.scalar_diag["Quseful_W"][kb])

        profile = build_equal_energy_dni_profile(
            latitude,
            month,
            float(data["DNI_W_m2"][i]),
            n_bins=int(n_bins),
            atmospheric_B=float(atmospheric_B),
        )

        q_samples: list[float] = []
        tout_samples: list[float] = []
        eta_samples: list[float] = []
        for _, p in profile.iterrows():
            cfg_slice, _ = build_rea_monthly_preset(city_name, month)
            _prep_diagnostic(cfg_slice)
            cfg_slice["solar"]["mode"] = "constante"
            cfg_slice["solar"]["DNI_constant_W_m2"] = float(p["DNI_W_m2"])
            cfg_slice["solar"]["angle_constant_deg"] = 0.0
            result = PTCSimulator(cfg_slice, deepcopy(db0)).simulate()
            k = len(result.t_s) - 1
            q_samples.append(float(result.scalar_diag["Quseful_W"][k]))
            tout_samples.append(float(result.Tout_C[k]))
            eta_samples.append(float(result.scalar_diag["eta_dni_basis_pct"][k]))

            profile_rows.append(
                {
                    "Caso": case_key,
                    "Mes": MONTH_ABBR_ES[i],
                    "Mes_num": month,
                    "Hora_solar_h": float(p["Hora_solar_h"]),
                    "cos_z": float(p["cos_z"]),
                    "DNI_W_m2": float(p["DNI_W_m2"]),
                    "Qutil_W": q_samples[-1],
                    "Tout_C": tout_samples[-1],
                    "Eta_inst_pct": eta_samples[-1],
                }
            )

        weights = profile["Peso"].to_numpy(dtype=float)
        dni_values = profile["DNI_W_m2"].to_numpy(dtype=float)
        q_values = np.asarray(q_samples, dtype=float)
        tout_values = np.asarray(tout_samples, dtype=float)
        aperture = float(cfg_const["geometry"]["W"]) * float(cfg_const["geometry"]["L"])

        dni_temporal_mean = float(np.sum(weights * dni_values))
        q_temporal_mean = float(np.sum(weights * q_values))
        tout_temporal_mean = float(np.sum(weights * tout_values))
        eta_temporal = 100.0 * q_temporal_mean / max(aperture * dni_temporal_mean, 1e-12)

        rows.append(
            {
                "Mes": MONTH_ABBR_ES[i],
                "Mes_num": month,
                "DNI_publicado_W_m2": float(data["DNI_W_m2"][i]),
                "DNI_perfil_medio_W_m2": dni_temporal_mean,
                "DNI_perfil_min_W_m2": float(np.min(dni_values)),
                "DNI_perfil_max_W_m2": float(np.max(dni_values)),
                "Eta_ref_pct": float(data["eta_ref_pct"][i]),
                "Eta_DNI_medio_pct": eta_const,
                "Eta_agregacion_temporal_pct": eta_temporal,
                "Delta_eta_agregacion_pp": eta_temporal - eta_const,
                "Tout_ref_C": float(data["Tout_ref_C"][i]),
                "Tout_DNI_medio_C": tout_const,
                "Tout_agregacion_temporal_C": tout_temporal_mean,
                "Delta_Tout_agregacion_C": tout_temporal_mean - tout_const,
                "Qutil_DNI_medio_W": q_const,
                "Qutil_temporal_medio_W": q_temporal_mean,
                "Error_eta_DNI_medio_pp": eta_const - float(data["eta_ref_pct"][i]),
                "Error_eta_temporal_pp": eta_temporal - float(data["eta_ref_pct"][i]),
            }
        )

    table = pd.DataFrame(rows)
    profiles = pd.DataFrame(profile_rows)
    eta_const_metrics = _metric(table["Eta_ref_pct"], table["Eta_DNI_medio_pct"])
    eta_temp_metrics = _metric(table["Eta_ref_pct"], table["Eta_agregacion_temporal_pct"])
    tout_const_metrics = _metric(table["Tout_ref_C"], table["Tout_DNI_medio_C"])
    tout_temp_metrics = _metric(table["Tout_ref_C"], table["Tout_agregacion_temporal_C"])

    improvement_pct = (
        100.0 * (eta_const_metrics["rmse"] - eta_temp_metrics["rmse"]) / eta_const_metrics["rmse"]
        if eta_const_metrics["rmse"] > 1e-12 else float("nan")
    )
    corr_delta = eta_temp_metrics["corr"] - eta_const_metrics["corr"]

    if np.isfinite(improvement_pct) and improvement_pct >= 10.0:
        verdict = "apoya"
        diagnosis = (
            "Distribuir temporalmente el mismo DNI medio reduce el RMSE de eficiencia al menos 10 %. "
            "La agregación mensual estaba ocultando una parte relevante de la no linealidad del colector."
        )
    elif np.isfinite(improvement_pct) and improvement_pct <= -10.0:
        verdict = "rechaza"
        diagnosis = (
            "La distribución temporal de igual energía empeora el RMSE al menos 10 %. "
            "El sesgo por usar el DNI mensual medio no explica la discrepancia principal con Rea."
        )
    else:
        verdict = "inconclusa"
        diagnosis = (
            "El cambio de RMSE es menor al 10 %. Con esta prueba controlada, la agregación temporal del DNI "
            "no explica por sí sola la discrepancia."
        )

    return {
        "case": case_key,
        "city": city_name,
        "table": table,
        "profiles": profiles,
        "metrics": {
            "eta_mean_input": eta_const_metrics,
            "eta_temporal": eta_temp_metrics,
            "tout_mean_input": tout_const_metrics,
            "tout_temporal": tout_temp_metrics,
            "eta_rmse_improvement_pct": float(improvement_pct),
            "eta_corr_delta": float(corr_delta),
            "mean_abs_jensen_gap_pp": float(np.mean(np.abs(table["Delta_eta_agregacion_pp"]))),
            "max_abs_jensen_gap_pp": float(np.max(np.abs(table["Delta_eta_agregacion_pp"]))),
            "max_profile_dni_W_m2": float(profiles["DNI_W_m2"].max()),
        },
        "verdict": verdict,
        "diagnosis": diagnosis,
        "settings": {"n_bins": int(n_bins), "atmospheric_B": float(atmospheric_B)},
        "note": (
            "Prueba diagnóstica sin calibración. El perfil temporal conserva exactamente el DNI medio publicado durante "
            "las horas solares representativas. Tin, Tamb, caudal, viento, óptica e incidencia permanecen fijos. Cada bin "
            "se resuelve cuasiestacionariamente; no es una reconstrucción transitoria completa del TRNSYS."
        ),
    }
