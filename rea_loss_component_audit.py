"""Auditoría diagnóstica de pérdidas convectivas vs radiativas para Rea Quille.

Esta etapa sucede a la auditoría global del balance energético. No calibra el
modelo. Mantiene fija la potencia solar absorbida y pregunta, para cada mes,
qué multiplicador tendría que aplicarse *a un único componente de pérdida* para
alcanzar el calor útil de referencia, dejando el otro componente sin cambios.

Convección como único bloque corregido:

    k_conv = (Qsolar - Qstorage - Quse_ref - Qrad - Qsupport) / Qconv

Radiación al cielo como único bloque corregido:

    k_rad = (Qsolar - Qstorage - Quse_ref - Qconv - Qsupport) / Qrad

Interpretación:
    k = 1   -> el componente actual ya tiene la magnitud requerida.
    0 < k < 1 -> el modelo necesita menos pérdida por ese mecanismo.
    k > 1   -> el modelo necesita más pérdida por ese mecanismo.
    k < 0   -> eliminar por completo ese mecanismo todavía no basta para cerrar
               la referencia; el componente no puede explicar solo el error.

Además se calcula, por mínimos cuadrados, el mejor multiplicador *global* para
cada componente por separado. Es sólo un contrafactual diagnóstico, no un
parámetro físico identificado.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from fluid_properties import FluidPropertyEvaluator
from presets import MONTH_ABBR_ES, REA_ALVORADA_MONTHLY, REA_FOZ_MONTHLY, build_rea_monthly_preset
from ptc_model import PTCSimulator
from validations import apply_calibrated_parameters


def _cv(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    mean = float(np.mean(arr))
    if abs(mean) <= np.finfo(float).eps:
        return float("nan")
    return float(np.std(arr, ddof=0) / abs(mean))


def _rmse(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(arr))))


def _city_data(city: str) -> tuple[str, str, Mapping[str, Any]]:
    key = city.strip().lower()
    if key.startswith("foz"):
        return "Foz do Iguaçu", "rea_foz", REA_FOZ_MONTHLY
    if key.startswith("alvorada"):
        return "Alvorada do Norte", "rea_alvorada", REA_ALVORADA_MONTHLY
    raise ValueError(f"Ciudad no soportada: {city}")


def _least_squares_multiplier(component: np.ndarray, required_component: np.ndarray) -> float:
    x = np.asarray(component, dtype=float)
    y = np.asarray(required_component, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (np.abs(x) > 1e-12)
    if not np.any(mask):
        return float("nan")
    denom = float(np.dot(x[mask], x[mask]))
    if denom <= 1e-20:
        return float("nan")
    return float(np.dot(x[mask], y[mask]) / denom)


def audit_rea_loss_components(
    city: str,
    fluid_database: Mapping[str, Mapping[str, Any]],
    calibration: Mapping[str, Any] | None = None,
    *,
    months: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Separa el error de pérdidas entre convección externa y radiación al cielo."""
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

        tin_c = float(data["Tin_C"][i])
        tout_ref_c = float(data["Tout_ref_C"][i])
        mdot = float(data["mdot_kg_s"][i])
        eta_ref = float(data["eta_ref_pct"][i])
        cp_ref = float(water(0.5 * (tin_c + tout_ref_c) + 273.15).Cp)
        q_use_ref = mdot * cp_ref * (tout_ref_c - tin_c)

        q_solar = float(result.scalar_diag["QsolarAbs_W"][k] + result.scalar_diag["QsolarGlass_W"][k])
        q_use_model = float(result.scalar_diag["Quseful_W"][k])
        q_storage = float(result.scalar_diag["Qstorage_est_W"][k])
        q_conv = float(np.sum(result.node_diag["Qconv_external_W"][k, :]))
        q_rad = float(np.sum(result.node_diag["Qrad_sky_W"][k, :]))
        q_support = float(np.sum(result.node_diag["Qsupports_W"][k, :]))
        q_loss = q_conv + q_rad + q_support

        # Pérdida total compatible con la referencia si Qsolar y Qstorage quedan fijos.
        q_loss_required = q_solar - q_storage - q_use_ref
        delta_q_use = q_use_ref - q_use_model

        # Cuánto debería valer cada componente si él solo absorbiera toda la corrección.
        q_conv_required = q_loss_required - q_rad - q_support
        q_rad_required = q_loss_required - q_conv - q_support
        k_conv = q_conv_required / q_conv if abs(q_conv) > 1e-12 else float("nan")
        k_rad = q_rad_required / q_rad if abs(q_rad) > 1e-12 else float("nan")

        # Base energética de la Eq. (10) publicada, inferida de la propia referencia.
        eta_denominator = q_use_ref / (eta_ref / 100.0) if eta_ref > 1e-12 else float("nan")
        eta_model_from_q = 100.0 * q_use_model / eta_denominator if eta_denominator > 1e-12 else float("nan")

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
                "Eta_modelo_pct": eta_model_from_q,
                "Qsolar_W": q_solar,
                "Qutil_modelo_W": q_use_model,
                "Qutil_ref_W": q_use_ref,
                "Delta_Qutil_requerida_W": delta_q_use,
                "Qloss_modelo_W": q_loss,
                "Qloss_requerida_W": q_loss_required,
                "Qconv_ext_W": q_conv,
                "Qrad_cielo_W": q_rad,
                "Qsoportes_W": q_support,
                "Qstorage_W": q_storage,
                "Participacion_conv_pct": 100.0 * q_conv / q_loss if q_loss > 1e-12 else float("nan"),
                "Participacion_rad_pct": 100.0 * q_rad / q_loss if q_loss > 1e-12 else float("nan"),
                "Qconv_requerida_si_sola_W": q_conv_required,
                "Qrad_requerida_si_sola_W": q_rad_required,
                "Factor_conv_requerido": k_conv,
                "Factor_rad_requerido": k_rad,
                "Cambio_conv_requerido_pct": 100.0 * (k_conv - 1.0),
                "Cambio_rad_requerido_pct": 100.0 * (k_rad - 1.0),
                "Conv_no_puede_cerrar_sola": bool(np.isfinite(k_conv) and k_conv < 0.0),
                "Rad_no_puede_cerrar_sola": bool(np.isfinite(k_rad) and k_rad < 0.0),
                "Eta_denominador_W": eta_denominator,
            }
        )

    table = pd.DataFrame(rows)

    conv = table["Qconv_ext_W"].to_numpy(float)
    rad = table["Qrad_cielo_W"].to_numpy(float)
    support = table["Qsoportes_W"].to_numpy(float)
    solar = table["Qsolar_W"].to_numpy(float)
    storage = table["Qstorage_W"].to_numpy(float)
    qref = table["Qutil_ref_W"].to_numpy(float)
    eta_ref_arr = table["Eta_ref_pct"].to_numpy(float)
    denom = table["Eta_denominador_W"].to_numpy(float)

    required_conv = solar - storage - qref - rad - support
    required_rad = solar - storage - qref - conv - support
    k_conv_global = _least_squares_multiplier(conv, required_conv)
    k_rad_global = _least_squares_multiplier(rad, required_rad)

    q_use_conv_global = solar - storage - (k_conv_global * conv + rad + support)
    q_use_rad_global = solar - storage - (conv + k_rad_global * rad + support)
    eta_conv_global = 100.0 * q_use_conv_global / denom
    eta_rad_global = 100.0 * q_use_rad_global / denom

    table["Qutil_kconv_global_W"] = q_use_conv_global
    table["Qutil_krad_global_W"] = q_use_rad_global
    table["Eta_kconv_global_pct"] = eta_conv_global
    table["Eta_krad_global_pct"] = eta_rad_global

    conv_factors = table["Factor_conv_requerido"].replace([np.inf, -np.inf], np.nan).dropna()
    rad_factors = table["Factor_rad_requerido"].replace([np.inf, -np.inf], np.nan).dropna()
    eta_base = table["Eta_modelo_pct"].to_numpy(float)
    q_base = table["Qutil_modelo_W"].to_numpy(float)

    metrics = {
        "eta_rmse_base_pp": _rmse(eta_base - eta_ref_arr),
        "eta_rmse_kconv_global_pp": _rmse(eta_conv_global - eta_ref_arr),
        "eta_rmse_krad_global_pp": _rmse(eta_rad_global - eta_ref_arr),
        "q_rmse_base_W": _rmse(q_base - qref),
        "q_rmse_kconv_global_W": _rmse(q_use_conv_global - qref),
        "q_rmse_krad_global_W": _rmse(q_use_rad_global - qref),
        "conv_factor_mean": float(conv_factors.mean()) if len(conv_factors) else float("nan"),
        "conv_factor_min": float(conv_factors.min()) if len(conv_factors) else float("nan"),
        "conv_factor_max": float(conv_factors.max()) if len(conv_factors) else float("nan"),
        "conv_factor_cv_pct": 100.0 * _cv(conv_factors),
        "rad_factor_mean": float(rad_factors.mean()) if len(rad_factors) else float("nan"),
        "rad_factor_min": float(rad_factors.min()) if len(rad_factors) else float("nan"),
        "rad_factor_max": float(rad_factors.max()) if len(rad_factors) else float("nan"),
        "rad_factor_cv_pct": 100.0 * _cv(rad_factors),
        "kconv_global_ls": float(k_conv_global),
        "krad_global_ls": float(k_rad_global),
        "conv_share_mean_pct": float(table["Participacion_conv_pct"].mean()),
        "rad_share_mean_pct": float(table["Participacion_rad_pct"].mean()),
        "conv_infeasible_months": int(table["Conv_no_puede_cerrar_sola"].sum()),
        "rad_infeasible_months": int(table["Rad_no_puede_cerrar_sola"].sum()),
        "max_storage_fraction_pct": float(
            100.0 * np.max(np.abs(storage) / np.maximum(np.abs(solar), 1e-12))
        ),
    }

    # Criterio diagnóstico: si ninguna corrección global mejora siquiera el RMSE
    # de eficiencia, la conclusión correcta es que no basta una constante global.
    # Si alguna mejora, preferimos el mecanismo con menor RMSE penalizado por la
    # variación mensual de su factor requerido.
    conv_score = metrics["eta_rmse_kconv_global_pp"] * (1.0 + metrics["conv_factor_cv_pct"] / 100.0)
    rad_score = metrics["eta_rmse_krad_global_pp"] * (1.0 + metrics["rad_factor_cv_pct"] / 100.0)
    if metrics["conv_infeasible_months"]:
        conv_score *= 2.0
    if metrics["rad_infeasible_months"]:
        rad_score *= 2.0
    base_rmse = metrics["eta_rmse_base_pp"]
    if min(metrics["eta_rmse_kconv_global_pp"], metrics["eta_rmse_krad_global_pp"]) >= base_rmse:
        preferred = "ninguno · revisar dependencia estacional"
    else:
        preferred = "convección externa" if conv_score <= rad_score else "radiación al cielo"
    metrics["preferred_component"] = preferred

    messages: list[str] = []
    messages.append(
        f"La convección representa en promedio {metrics['conv_share_mean_pct']:.1f}% de las pérdidas externas y la radiación al cielo {metrics['rad_share_mean_pct']:.1f}%."
    )
    messages.append(
        f"Un único multiplicador convectivo por mínimos cuadrados sería k_conv={metrics['kconv_global_ls']:.3f} y llevaría el RMSE de η de "
        f"{metrics['eta_rmse_base_pp']:.2f} a {metrics['eta_rmse_kconv_global_pp']:.2f} pp."
    )
    messages.append(
        f"Un único multiplicador radiativo sería k_rad={metrics['krad_global_ls']:.3f} y llevaría el RMSE de η a {metrics['eta_rmse_krad_global_pp']:.2f} pp."
    )
    if metrics["conv_factor_cv_pct"] + 2.0 < metrics["rad_factor_cv_pct"]:
        messages.append(
            "El factor convectivo requerido es claramente más estable entre meses que el radiativo; la formulación de convección externa merece revisarse primero."
        )
    elif metrics["rad_factor_cv_pct"] + 2.0 < metrics["conv_factor_cv_pct"]:
        messages.append(
            "El factor radiativo requerido es claramente más estable entre meses que el convectivo; la formulación radiativa/cielo merece revisarse primero."
        )
    else:
        messages.append(
            "Los dos factores tienen una variación estacional comparable; no hay un único componente claramente identificable con esta prueba."
        )
    if metrics["conv_infeasible_months"] or metrics["rad_infeasible_months"]:
        messages.append(
            "Al menos un mecanismo requiere factor negativo en algún mes, por lo que ese mecanismo no puede explicar por sí solo toda la discrepancia."
        )
    if preferred.startswith("ninguno"):
        messages.append(
            "Ningún multiplicador global de convección o radiación mejora la eficiencia anual: la siguiente revisión debe centrarse en la dependencia estacional/ambiental de las pérdidas, no en otra constante."
        )
    else:
        messages.append(f"Prioridad diagnóstica sugerida: revisar {preferred} antes de introducir otra constante global.")

    return {
        "city": city_name,
        "case": expected_case,
        "table": table,
        "metrics": metrics,
        "parameter_source": parameter_source,
        "diagnosis": messages,
        "note": (
            "Prueba sin optimización física. Los factores mensuales responden a un contrafactual de un solo bloque: se mantiene todo el modelo igual y se pregunta cuánto tendría que multiplicarse sólo Qconv o sólo Qrad para cerrar Qútil de referencia. "
            "Los multiplicadores globales LS son únicamente resúmenes diagnósticos; no se aplican al simulador ni se guardan como parámetros calibrados."
        ),
    }
