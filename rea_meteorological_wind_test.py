"""Prueba diagnóstica de la hipótesis de viento meteorológico mensual.

Objetivo
--------
Comparar el supuesto histórico ``v = 1 m/s`` usado en los presets mensuales de
Rea Quille contra una serie mensual de viento independiente. La prueba NO
calibra parámetros ni modifica correlaciones de transferencia de calor.

Se ejecutan dos escenarios con idénticas constantes físicas:

1. baseline: viento fijo de 1 m/s (preset actual);
2. meteorológico: viento mensual proporcionado por una fuente independiente.

El viento climatológico suele venir referido a 10 m. Opcionalmente se corrige a
la altura efectiva del receptor con una ley de potencia:

    v(z) = v_10 (z / 10)^alpha

La altura y ``alpha`` son hipótesis explícitas del ensayo, no datos de Rea.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json

import numpy as np
import pandas as pd

from presets import (
    MONTH_ABBR_ES,
    REA_ALVORADA_MONTHLY,
    REA_FOZ_MONTHLY,
    build_rea_monthly_preset,
)
from ptc_model import PTCSimulator


# WeatherSpark, Foz do Iguaçu. Valores publicados como velocidad media horaria
# a 10 m, en mph. WeatherSpark indica que el viento proviene de NASA MERRA-2.
WEATHERSPARK_FOZ_WIND_MPH = [3.7, 3.6, 3.6, 3.9, 4.0, 4.1, 4.3, 4.4, 4.4, 4.3, 4.0, 3.8]
MPH_TO_M_S = 0.44704

CITY_COORDS = {
    "rea_foz": {"name": "Foz do Iguaçu", "lat": -25.43816, "lon": -54.59679},
    # Coordenadas de sede municipal de Alvorada do Norte (aprox. IBGE).
    "rea_alvorada": {"name": "Alvorada do Norte", "lat": -14.4808, "lon": -46.4922},
}

NASA_MONTH_KEYS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def _case_info(case: str) -> tuple[str, str, Mapping[str, Any]]:
    key = str(case).strip().lower()
    if key in {"rea_foz", "foz", "foz do iguaçu", "foz do iguacu"}:
        return "rea_foz", "Foz do Iguaçu", REA_FOZ_MONTHLY
    if key in {"rea_alvorada", "alvorada", "alvorada do norte"}:
        return "rea_alvorada", "Alvorada do Norte", REA_ALVORADA_MONTHLY
    raise ValueError(f"Caso no soportado: {case}")


def foz_weatherspark_wind_10m() -> pd.DataFrame:
    values = np.asarray(WEATHERSPARK_FOZ_WIND_MPH, dtype=float) * MPH_TO_M_S
    return pd.DataFrame(
        {
            "Mes": MONTH_ABBR_ES,
            "Viento_10m_m_s": values,
            "Fuente": ["WeatherSpark / NASA MERRA-2"] * 12,
        }
    )


def fetch_nasa_power_ws10m_climatology(
    case: str,
    *,
    start_year: int = 2001,
    end_year: int = 2020,
    timeout_s: float = 25.0,
) -> pd.DataFrame:
    """Descarga WS10M climatológico mensual desde NASA POWER.

    La función se usa solo por acción explícita del usuario en la interfaz. Si
    el servidor no tiene salida a Internet, levanta una excepción legible y la
    interfaz conserva la serie editable/manual.
    """
    key, _, _ = _case_info(case)
    coords = CITY_COORDS[key]
    query = urlencode(
        {
            "parameters": "WS10M",
            "community": "RE",
            "longitude": coords["lon"],
            "latitude": coords["lat"],
            "format": "JSON",
            "start": int(start_year),
            "end": int(end_year),
        }
    )
    url = f"https://power.larc.nasa.gov/api/temporal/climatology/point?{query}"
    req = Request(url, headers={"User-Agent": "PTC-validation/14.7"})
    with urlopen(req, timeout=float(timeout_s)) as response:  # nosec B310 - URL fija NASA
        payload = json.loads(response.read().decode("utf-8"))
    ws = payload.get("properties", {}).get("parameter", {}).get("WS10M", {})
    values = [float(ws[k]) for k in NASA_MONTH_KEYS]
    if any((not np.isfinite(v)) or v < 0 for v in values):
        raise ValueError("NASA POWER devolvió valores WS10M inválidos.")
    return pd.DataFrame(
        {
            "Mes": MONTH_ABBR_ES,
            "Viento_10m_m_s": values,
            "Fuente": [f"NASA POWER WS10M climatología {start_year}-{end_year}"] * 12,
        }
    )


def adjust_wind_height(
    wind_10m_m_s: Sequence[float],
    *,
    receiver_height_m: float,
    alpha: float,
    apply_adjustment: bool,
) -> np.ndarray:
    v10 = np.asarray(wind_10m_m_s, dtype=float)
    if v10.size != 12:
        raise ValueError("La serie de viento debe contener exactamente 12 meses.")
    if np.any(~np.isfinite(v10)) or np.any(v10 < 0):
        raise ValueError("La serie de viento contiene valores inválidos.")
    if not apply_adjustment:
        return v10.copy()
    z = float(receiver_height_m)
    a = float(alpha)
    if not (0.05 <= z <= 10.0):
        raise ValueError("La altura efectiva del receptor debe estar entre 0.05 y 10 m.")
    if not (0.0 <= a <= 0.6):
        raise ValueError("El exponente alpha debe estar entre 0 y 0.6.")
    return v10 * (z / 10.0) ** a


def _prep_diagnostic(cfg: dict[str, Any]) -> None:
    """Acelera el ensayo sin cambiar su estructura física."""
    cfg["geometry"]["Nseg"] = min(int(cfg["geometry"].get("Nseg", 12)), 6)
    cfg["solver"]["method"] = "BDF"
    cfg["solver"]["max_step_s"] = max(float(cfg["solver"].get("max_step_s", 20.0)), 600.0)
    cfg["operation"]["output_step_s"] = max(float(cfg["operation"].get("output_step_s", 60.0)), 600.0)


def _metric(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    err = pred - ref
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(ref), 1e-12)) * 100.0)
    corr = float(np.corrcoef(ref, pred)[0, 1]) if np.std(ref) > 0 and np.std(pred) > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "bias": bias, "mape_pct": mape, "corr": corr}


def run_monthly_wind_hypothesis(
    case: str,
    fluid_database: Mapping[str, Mapping[str, Any]],
    wind_10m_m_s: Sequence[float],
    *,
    apply_height_adjustment: bool = True,
    receiver_height_m: float = 1.0,
    alpha: float = 0.14,
    baseline_wind_m_s: float = 1.0,
    months: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Compara viento fijo vs viento meteorológico mensual sin calibración."""
    case_key, city_name, data = _case_info(case)
    wind10 = np.asarray(wind_10m_m_s, dtype=float)
    wind_used = adjust_wind_height(
        wind10,
        receiver_height_m=receiver_height_m,
        alpha=alpha,
        apply_adjustment=apply_height_adjustment,
    )

    month_numbers = list(range(1, 13)) if months is None else [int(m) for m in months]
    if not month_numbers or any(m < 1 or m > 12 for m in month_numbers):
        raise ValueError("months debe contener valores entre 1 y 12.")

    rows: list[dict[str, Any]] = []
    db0 = deepcopy(fluid_database)
    for month in month_numbers:
        i = month - 1

        cfg_base, _ = build_rea_monthly_preset(city_name, month)
        _prep_diagnostic(cfg_base)
        cfg_base["environment"]["wind_m_s"] = float(baseline_wind_m_s)
        base = PTCSimulator(cfg_base, deepcopy(db0)).simulate()

        cfg_met, _ = build_rea_monthly_preset(city_name, month)
        _prep_diagnostic(cfg_met)
        cfg_met["environment"]["wind_m_s"] = float(wind_used[i])
        met = PTCSimulator(cfg_met, deepcopy(db0)).simulate()

        kb = len(base.t_s) - 1
        km = len(met.t_s) - 1
        eta_ref = float(data["eta_ref_pct"][i])
        tout_ref = float(data["Tout_ref_C"][i])
        eta_base = float(base.scalar_diag["eta_pct"][kb])
        eta_met = float(met.scalar_diag["eta_pct"][km])
        tout_base = float(base.Tout_C[kb])
        tout_met = float(met.Tout_C[km])
        qconv_base = float(np.sum(base.node_diag["Qconv_external_W"][kb, :]))
        qconv_met = float(np.sum(met.node_diag["Qconv_external_W"][km, :]))

        rows.append(
            {
                "Mes": MONTH_ABBR_ES[i],
                "Mes_num": month,
                "Viento_base_m_s": float(baseline_wind_m_s),
                "Viento_10m_m_s": float(wind10[i]),
                "Viento_usado_m_s": float(wind_used[i]),
                "Eta_ref_pct": eta_ref,
                "Eta_base_pct": eta_base,
                "Eta_meteo_pct": eta_met,
                "Error_eta_base_pp": eta_base - eta_ref,
                "Error_eta_meteo_pp": eta_met - eta_ref,
                "Tout_ref_C": tout_ref,
                "Tout_base_C": tout_base,
                "Tout_meteo_C": tout_met,
                "Error_Tout_base_C": tout_base - tout_ref,
                "Error_Tout_meteo_C": tout_met - tout_ref,
                "Qconv_base_W": qconv_base,
                "Qconv_meteo_W": qconv_met,
                "Delta_Qconv_W": qconv_met - qconv_base,
            }
        )

    table = pd.DataFrame(rows)
    eta_base_metrics = _metric(table["Eta_ref_pct"], table["Eta_base_pct"])
    eta_met_metrics = _metric(table["Eta_ref_pct"], table["Eta_meteo_pct"])
    tout_base_metrics = _metric(table["Tout_ref_C"], table["Tout_base_C"])
    tout_met_metrics = _metric(table["Tout_ref_C"], table["Tout_meteo_C"])

    rmse_change = eta_met_metrics["rmse"] - eta_base_metrics["rmse"]
    improvement_pct = (
        100.0 * (eta_base_metrics["rmse"] - eta_met_metrics["rmse"]) / eta_base_metrics["rmse"]
        if eta_base_metrics["rmse"] > 1e-12
        else float("nan")
    )
    corr_change = eta_met_metrics["corr"] - eta_base_metrics["corr"]

    if np.isfinite(improvement_pct) and improvement_pct >= 10.0:
        verdict = "apoya"
        diagnosis = (
            "La serie meteorológica reduce el RMSE de eficiencia al menos 10 %. "
            "La hipótesis de viento mensual merece incorporarse como entrada ambiental independiente."
        )
    elif np.isfinite(improvement_pct) and improvement_pct <= -10.0:
        verdict = "rechaza"
        diagnosis = (
            "La serie meteorológica independiente empeora el RMSE de eficiencia al menos 10 %. "
            "El supuesto de viento fijo no explica por sí solo el déficit del modelo; no conviene calibrar el viento para forzar el ajuste."
        )
    else:
        verdict = "inconclusa"
        diagnosis = (
            "El cambio de RMSE es menor al 10 %. El viento mensual, con esta fuente y tratamiento de altura, "
            "no explica por sí solo la discrepancia."
        )

    return {
        "case": case_key,
        "city": city_name,
        "table": table,
        "metrics": {
            "eta_base": eta_base_metrics,
            "eta_meteo": eta_met_metrics,
            "tout_base": tout_base_metrics,
            "tout_meteo": tout_met_metrics,
            "eta_rmse_delta_pp": float(rmse_change),
            "eta_rmse_improvement_pct": float(improvement_pct),
            "eta_corr_delta": float(corr_change),
            "wind_10m_mean_m_s": float(np.mean(wind10)),
            "wind_used_mean_m_s": float(np.mean(wind_used)),
            "wind_used_min_m_s": float(np.min(wind_used)),
            "wind_used_max_m_s": float(np.max(wind_used)),
            "qconv_delta_mean_W": float(np.mean(table["Delta_Qconv_W"])),
        },
        "verdict": verdict,
        "diagnosis": diagnosis,
        "settings": {
            "apply_height_adjustment": bool(apply_height_adjustment),
            "receiver_height_m": float(receiver_height_m),
            "alpha": float(alpha),
            "baseline_wind_m_s": float(baseline_wind_m_s),
        },
        "note": (
            "Prueba diagnóstica sin calibración. El viento a 10 m es una variable meteorológica; la corrección de altura "
            "es una hipótesis de capa límite. Los resultados usan N=6 para acelerar el ensayo y no modifican el modelo guardado."
        ),
    }
