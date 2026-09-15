"""Auditoría diagnóstica del flujo externo alrededor del receptor PTC.

Esta etapa sucede a la separación convección/radiación. No calibra ni modifica
el modelo. Mantiene congeladas las temperaturas de superficie obtenidas en cada
simulación mensual y abre el término convectivo externo para inspeccionar:

- temperatura de superficie y de película;
- propiedades del aire usadas por el modelo;
- Reynolds, Prandtl, Nusselt y h externos;
- h requerido para cerrar el calor útil documental manteniendo los demás
  términos del balance sin cambios;
- velocidad de viento implícita que, con la MISMA correlación actual
  (Churchill-Bernstein), produciría ese h requerido sobre el campo térmico
  congelado.

La velocidad requerida es un diagnóstico local, no una reconstrucción
meteorológica: al cambiar el viento en una simulación acoplada también cambiarían
las temperaturas de superficie.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from fluid_properties import FluidPropertyEvaluator
from presets import MONTH_ABBR_ES, REA_ALVORADA_MONTHLY, REA_FOZ_MONTHLY, build_rea_monthly_preset
from ptc_model import PTCSimulator, air_properties, external_convection
from validations import apply_calibrated_parameters


def _city_data(city: str) -> tuple[str, str, Mapping[str, Any]]:
    key = city.strip().lower()
    if key.startswith("foz"):
        return "Foz do Iguaçu", "rea_foz", REA_FOZ_MONTHLY
    if key.startswith("alvorada"):
        return "Alvorada do Norte", "rea_alvorada", REA_ALVORADA_MONTHLY
    raise ValueError(f"Ciudad no soportada: {city}")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.std(x) <= 1e-15 or np.std(y) <= 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _cv(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    mean = float(np.mean(arr))
    if abs(mean) <= 1e-15:
        return float("nan")
    return float(np.std(arr, ddof=0) / abs(mean))


def _external_field_at_wind(
    surface_K: np.ndarray,
    Tamb_K: float,
    diameter_m: float,
    length_m: float,
    pressure_Pa: float,
    wind_m_s: float,
) -> dict[str, Any]:
    surface = np.asarray(surface_K, dtype=float)
    n = surface.size
    dx = float(length_m) / max(n, 1)
    area_node = np.pi * float(diameter_m) * dx
    deltaT = surface - float(Tamb_K)
    film = 0.5 * (surface + float(Tamb_K))

    h = np.empty(n, dtype=float)
    Re = np.empty(n, dtype=float)
    Nu = np.empty(n, dtype=float)
    rho = np.empty(n, dtype=float)
    mu = np.empty(n, dtype=float)
    k_air = np.empty(n, dtype=float)
    Cp = np.empty(n, dtype=float)
    Pr = np.empty(n, dtype=float)

    for i, Tf in enumerate(film):
        conv = external_convection(float(Tf), float(wind_m_s), float(diameter_m), float(pressure_Pa))
        air = air_properties(float(Tf), float(pressure_Pa))
        h[i] = conv["h_W_m2K"]
        Re[i] = conv["Re"]
        Nu[i] = conv["Nu"]
        rho[i] = air["rho"]
        mu[i] = air["mu"]
        k_air[i] = air["k"]
        Cp[i] = air["Cp"]
        Pr[i] = air["Pr"]

    q_nodes = h * area_node * deltaT
    conductance_temperature = float(np.sum(area_node * deltaT))
    q_total = float(np.sum(q_nodes))
    h_effective = q_total / conductance_temperature if abs(conductance_temperature) > 1e-12 else float("nan")

    return {
        "q_total_W": q_total,
        "h_effective_W_m2K": h_effective,
        "surface_mean_K": float(np.mean(surface)),
        "film_mean_K": float(np.mean(film)),
        "deltaT_mean_K": float(np.mean(deltaT)),
        "Re_mean": float(np.mean(Re)),
        "Re_min": float(np.min(Re)),
        "Re_max": float(np.max(Re)),
        "Nu_mean": float(np.mean(Nu)),
        "h_mean_W_m2K": float(np.mean(h)),
        "rho_mean_kg_m3": float(np.mean(rho)),
        "mu_mean_Pa_s": float(np.mean(mu)),
        "k_mean_W_mK": float(np.mean(k_air)),
        "Cp_mean_J_kgK": float(np.mean(Cp)),
        "Pr_mean": float(np.mean(Pr)),
        "area_deltaT_W_K": conductance_temperature,
        "node": {
            "surface_K": surface,
            "film_K": film,
            "deltaT_K": deltaT,
            "h_W_m2K": h,
            "Re": Re,
            "Nu": Nu,
            "q_W": q_nodes,
        },
    }


def _required_wind_for_q(
    q_required_W: float,
    surface_K: np.ndarray,
    Tamb_K: float,
    diameter_m: float,
    length_m: float,
    pressure_Pa: float,
    *,
    max_wind_m_s: float = 30.0,
) -> tuple[float, str]:
    """Viento que reproduce Qconv requerido con temperaturas congeladas."""
    if not np.isfinite(q_required_W):
        return float("nan"), "Q requerido no finito"
    if q_required_W < 0.0:
        return float("nan"), "Qconv requerida negativa; viento no puede cerrar el balance"

    def f(v: float) -> float:
        return _external_field_at_wind(
            surface_K, Tamb_K, diameter_m, length_m, pressure_Pa, v
        )["q_total_W"] - q_required_W

    f0 = f(0.0)
    fmax = f(float(max_wind_m_s))
    if abs(f0) <= 1e-8:
        return 0.0, "solución en viento nulo"
    if f0 > 0.0:
        return float("nan"), "menor que la pérdida mínima del modelo a v=0"
    if fmax < 0.0:
        return float("nan"), f"requiere viento > {max_wind_m_s:g} m/s"
    try:
        value = float(brentq(f, 0.0, float(max_wind_m_s), xtol=1e-9, rtol=1e-9, maxiter=100))
        return value, "solución dentro del rango"
    except Exception as exc:  # pragma: no cover - protección de interfaz
        return float("nan"), f"sin convergencia: {exc}"


def audit_rea_external_flow(
    city: str,
    fluid_database: Mapping[str, Mapping[str, Any]],
    calibration: Mapping[str, Any] | None = None,
    *,
    months: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Audita la correlación de convección externa usada en los presets mensuales Rea."""
    city_name, expected_case, data = _city_data(city)
    month_numbers = list(range(1, 13)) if months is None else [int(m) for m in months]
    if not month_numbers or any(m < 1 or m > 12 for m in month_numbers):
        raise ValueError("months debe contener valores entre 1 y 12.")

    parameter_source = "Parámetros nominales del preset Rea Quille"
    if calibration is not None:
        if not isinstance(calibration, Mapping) or calibration.get("case") != expected_case:
            raise ValueError("La calibración suministrada no corresponde a la ciudad seleccionada.")
        parameter_source = "Última calibración guardada para esta ciudad"

    water = FluidPropertyEvaluator("Agua", fluid_database)
    rows: list[dict[str, Any]] = []

    for month in month_numbers:
        i = month - 1
        cfg, _ = build_rea_monthly_preset(city_name, month)
        db = deepcopy(fluid_database)
        if calibration is not None:
            apply_calibrated_parameters(cfg, db, calibration)

        result = PTCSimulator(cfg, db).simulate()
        k = len(result.t_s) - 1
        has_glass = bool(cfg["model"]["has_glass"])
        surface_C = result.Tglass_C[k, :] if has_glass else result.Tabs_C[k, :]
        surface_K = np.asarray(surface_C, dtype=float) + 273.15
        Tamb_K = float(cfg["environment"]["Tamb_K"])
        wind_model = float(cfg["environment"]["wind_m_s"])
        pressure = float(cfg["environment"]["pressure_Pa"])
        diameter = float(cfg["geometry"]["D5"] if has_glass else cfg["geometry"]["D3"])
        length = float(cfg["geometry"]["L"])

        flow = _external_field_at_wind(surface_K, Tamb_K, diameter, length, pressure, wind_model)

        tin_c = float(data["Tin_C"][i])
        tout_ref_c = float(data["Tout_ref_C"][i])
        mdot = float(data["mdot_kg_s"][i])
        eta_ref = float(data["eta_ref_pct"][i])
        cp_ref = float(water(0.5 * (tin_c + tout_ref_c) + 273.15).Cp)
        q_use_ref = mdot * cp_ref * (tout_ref_c - tin_c)

        q_solar = float(result.scalar_diag["QsolarAbs_W"][k] + result.scalar_diag["QsolarGlass_W"][k])
        q_use_model = float(result.scalar_diag["Quseful_W"][k])
        q_storage = float(result.scalar_diag["Qstorage_est_W"][k])
        q_conv_model = float(np.sum(result.node_diag["Qconv_external_W"][k, :]))
        q_rad = float(np.sum(result.node_diag["Qrad_sky_W"][k, :]))
        q_support = float(np.sum(result.node_diag["Qsupports_W"][k, :]))

        # Si únicamente la convección absorbiera todo el cierre del balance:
        q_conv_required = q_solar - q_storage - q_use_ref - q_rad - q_support
        h_required = (
            q_conv_required / flow["area_deltaT_W_K"]
            if abs(flow["area_deltaT_W_K"]) > 1e-12
            else float("nan")
        )
        h_model = float(flow["h_effective_W_m2K"])
        h_factor = h_required / h_model if np.isfinite(h_required) and abs(h_model) > 1e-12 else float("nan")

        wind_required, wind_status = _required_wind_for_q(
            q_conv_required,
            surface_K,
            Tamb_K,
            diameter,
            length,
            pressure,
        )
        flow_req = (
            _external_field_at_wind(surface_K, Tamb_K, diameter, length, pressure, wind_required)
            if np.isfinite(wind_required)
            else None
        )

        eta_denominator = q_use_ref / (eta_ref / 100.0) if eta_ref > 1e-12 else float("nan")
        eta_model = 100.0 * q_use_model / eta_denominator if eta_denominator > 1e-12 else float("nan")

        rows.append(
            {
                "Mes": MONTH_ABBR_ES[i],
                "Mes_num": month,
                "Tin_C": tin_c,
                "Tout_ref_C": tout_ref_c,
                "Tamb_C": float(data["Tamb_C"][i]),
                "DNI_W_m2": float(data["DNI_W_m2"][i]),
                "mdot_kg_s": mdot,
                "Eta_ref_pct": eta_ref,
                "Eta_modelo_pct": eta_model,
                "Qsolar_W": q_solar,
                "Qutil_modelo_W": q_use_model,
                "Qutil_ref_W": q_use_ref,
                "Qrad_cielo_W": q_rad,
                "Qsoportes_W": q_support,
                "Qstorage_W": q_storage,
                "Qconv_modelo_W": q_conv_model,
                "Qconv_requerida_W": q_conv_required,
                "Superficie_externa": "vidrio" if has_glass else "absorbedor",
                "D_externo_m": diameter,
                "Tsuperficie_media_C": flow["surface_mean_K"] - 273.15,
                "Tfilm_media_C": flow["film_mean_K"] - 273.15,
                "DeltaT_superficie_amb_K": flow["deltaT_mean_K"],
                "Viento_modelo_m_s": wind_model,
                "Viento_requerido_m_s": wind_required,
                "Estado_viento_requerido": wind_status,
                "h_modelo_W_m2K": h_model,
                "h_requerido_W_m2K": h_required,
                "Factor_h_requerido": h_factor,
                "Cambio_h_requerido_pct": 100.0 * (h_factor - 1.0) if np.isfinite(h_factor) else float("nan"),
                "Re_modelo": flow["Re_mean"],
                "Nu_modelo": flow["Nu_mean"],
                "Pr_modelo": flow["Pr_mean"],
                "rho_aire_kg_m3": flow["rho_mean_kg_m3"],
                "mu_aire_Pa_s": flow["mu_mean_Pa_s"],
                "k_aire_W_mK": flow["k_mean_W_mK"],
                "Cp_aire_J_kgK": flow["Cp_mean_J_kgK"],
                "Re_requerido": flow_req["Re_mean"] if flow_req else float("nan"),
                "Nu_requerido": flow_req["Nu_mean"] if flow_req else float("nan"),
                "h_recalculado_v_req_W_m2K": flow_req["h_effective_W_m2K"] if flow_req else float("nan"),
                "Qconv_recalculada_v_req_W": flow_req["q_total_W"] if flow_req else float("nan"),
            }
        )

    table = pd.DataFrame(rows)
    finite_wind = table["Viento_requerido_m_s"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    hfactor = table["Factor_h_requerido"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)

    metrics = {
        "current_wind_mean_m_s": float(table["Viento_modelo_m_s"].mean()),
        "required_wind_mean_m_s": float(np.mean(finite_wind)) if finite_wind.size else float("nan"),
        "required_wind_min_m_s": float(np.min(finite_wind)) if finite_wind.size else float("nan"),
        "required_wind_max_m_s": float(np.max(finite_wind)) if finite_wind.size else float("nan"),
        "required_wind_cv_pct": 100.0 * _cv(finite_wind),
        "wind_solution_months": int(np.isfinite(table["Viento_requerido_m_s"]).sum()),
        "h_model_mean_W_m2K": float(table["h_modelo_W_m2K"].mean()),
        "h_required_mean_W_m2K": float(table["h_requerido_W_m2K"].mean()),
        "h_factor_mean": float(np.mean(hfactor)) if hfactor.size else float("nan"),
        "h_factor_cv_pct": 100.0 * _cv(hfactor),
        "Re_model_mean": float(table["Re_modelo"].mean()),
        "Nu_model_mean": float(table["Nu_modelo"].mean()),
        "film_temp_range_K": float(table["Tfilm_media_C"].max() - table["Tfilm_media_C"].min()),
        "corr_hfactor_Re": _corr(table["Factor_h_requerido"].to_numpy(float), table["Re_modelo"].to_numpy(float)),
        "corr_hfactor_Tfilm": _corr(table["Factor_h_requerido"].to_numpy(float), table["Tfilm_media_C"].to_numpy(float)),
        "corr_hfactor_DeltaT": _corr(table["Factor_h_requerido"].to_numpy(float), table["DeltaT_superficie_amb_K"].to_numpy(float)),
        "corr_vreq_DNI": _corr(table["Viento_requerido_m_s"].to_numpy(float), table["DNI_W_m2"].to_numpy(float)),
    }

    diagnosis: list[str] = []
    diagnosis.append(
        "El preset mensual de Rea Quille usa viento constante de "
        f"{metrics['current_wind_mean_m_s']:.2f} m/s porque las Tablas 10/11 no publican viento."
    )
    if finite_wind.size:
        diagnosis.append(
            "Con las temperaturas de superficie congeladas, la misma correlación Churchill–Bernstein requeriría "
            f"viento entre {metrics['required_wind_min_m_s']:.2f} y {metrics['required_wind_max_m_s']:.2f} m/s "
            f"(media {metrics['required_wind_mean_m_s']:.2f} m/s) para cerrar el balance mensual."
        )
    if metrics["wind_solution_months"] < len(table):
        diagnosis.append(
            f"En {len(table) - metrics['wind_solution_months']} mes(es) el viento por sí solo no puede reproducir la pérdida requerida dentro del rango 0–30 m/s."
        )
    if np.isfinite(metrics["h_factor_cv_pct"]):
        if metrics["h_factor_cv_pct"] < 10.0:
            diagnosis.append(
                f"El factor h requerido es relativamente estable (CV={metrics['h_factor_cv_pct']:.1f}%). Esto favorece una corrección casi global de la convección."
            )
        else:
            diagnosis.append(
                f"El factor h requerido cambia de forma apreciable entre meses (CV={metrics['h_factor_cv_pct']:.1f}%). Una sola constante convectiva no parece suficiente."
            )
    if np.isfinite(metrics["required_wind_cv_pct"]) and metrics["required_wind_cv_pct"] > 20.0:
        diagnosis.append(
            f"El viento implícito requerido es claramente estacional (CV={metrics['required_wind_cv_pct']:.1f}%). La hipótesis fija de 1 m/s merece ser revisada antes de cambiar la correlación."
        )

    # Señal heurística sobre qué dependencia revisar primero.
    dependencies = {
        "Reynolds": abs(metrics["corr_hfactor_Re"]) if np.isfinite(metrics["corr_hfactor_Re"]) else -1.0,
        "temperatura de película": abs(metrics["corr_hfactor_Tfilm"]) if np.isfinite(metrics["corr_hfactor_Tfilm"]) else -1.0,
        "ΔT superficie-ambiente": abs(metrics["corr_hfactor_DeltaT"]) if np.isfinite(metrics["corr_hfactor_DeltaT"]) else -1.0,
    }
    dominant = max(dependencies, key=dependencies.get)
    dominant_corr = dependencies[dominant]
    if dominant_corr >= 0.6:
        diagnosis.append(
            f"El factor requerido presenta su asociación más fuerte con {dominant} (|r|≈{dominant_corr:.2f}); conviene auditar esa dependencia de la correlación en la siguiente etapa."
        )
    else:
        diagnosis.append(
            "Ninguna variable interna aislada (Re, Tfilm o ΔT) explica fuertemente el factor requerido; esto favorece la hipótesis de una entrada meteorológica faltante antes que un error simple de forma en Churchill–Bernstein."
        )

    return {
        "case": expected_case,
        "city": city_name,
        "parameter_source": parameter_source,
        "table": table,
        "metrics": metrics,
        "diagnosis": diagnosis,
        "note": (
            "Esta auditoría mantiene congelado el campo de temperaturas obtenido por el modelo. "
            "h requerido y viento requerido son diagnósticos locales; no son parámetros calibrados ni una reconstrucción meteorológica. "
            "La correlación evaluada es exactamente la usada por el modelo: Churchill–Bernstein sobre cilindro en flujo cruzado."
        ),
    }
