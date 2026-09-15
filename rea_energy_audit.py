"""Auditoría diagnóstica mensual del balance energético para Rea Quille.

La prueba no calibra parámetros. Ejecuta cada mes con las entradas publicadas y
pregunta qué corrección sería necesaria, *manteniendo fijo el resto del modelo*,
para alcanzar el calor útil de referencia.

Dos factores contrafactuales se calculan en régimen cuasiestacionario:

    F_opt = (Quse_ref + Qloss_model) / Qsolar_model

    F_loss = (Qsolar_model - Quse_ref) / Qloss_model

F_opt indica cuánto habría que multiplicar la potencia solar absorbida si se
mantuvieran las pérdidas actuales. F_loss indica cuánto habría que multiplicar
las pérdidas térmicas si se mantuviera fija la potencia solar absorbida.

No deben interpretarse como parámetros físicos identificados: son herramientas
de diagnóstico para decidir qué bloque del modelo merece la siguiente revisión.
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


def _relative_error(sim: float, ref: float) -> float:
    return 100.0 * abs(float(sim) - float(ref)) / max(abs(float(ref)), np.finfo(float).eps)


def _cv(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    mean = float(np.mean(arr))
    if abs(mean) <= np.finfo(float).eps:
        return float("nan")
    return float(np.std(arr, ddof=0) / abs(mean))


def _city_data(city: str) -> tuple[str, str, Mapping[str, Any]]:
    key = city.strip().lower()
    if key.startswith("foz"):
        return "Foz do Iguaçu", "rea_foz", REA_FOZ_MONTHLY
    if key.startswith("alvorada"):
        return "Alvorada do Norte", "rea_alvorada", REA_ALVORADA_MONTHLY
    raise ValueError(f"Ciudad no soportada: {city}")


def audit_rea_monthly_energy_balance(
    city: str,
    fluid_database: Mapping[str, Mapping[str, Any]],
    calibration: Mapping[str, Any] | None = None,
    *,
    months: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Ejecuta una auditoría mensual de potencia sin optimizar nada.

    Parameters
    ----------
    city:
        ``Foz do Iguaçu`` o ``Alvorada do Norte``.
    fluid_database:
        Base de propiedades usada por el simulador.
    calibration:
        Opcionalmente, una calibración ya existente de la misma ciudad. La
        función sólo la aplica; nunca vuelve a optimizar.
    months:
        Meses 1..12 a ejecutar. Por defecto, los doce. El argumento existe
        también para poder hacer pruebas rápidas de regresión.
    """
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
        tout_model_c = float(result.Tout_C[k])
        mdot = float(data["mdot_kg_s"][i])
        dni = float(data["DNI_W_m2"][i])
        eta_ref = float(data["eta_ref_pct"][i])

        cp_ref = float(water(0.5 * (tin_c + tout_ref_c) + 273.15).Cp)
        q_use_ref = mdot * cp_ref * (tout_ref_c - tin_c)
        q_solar = float(result.scalar_diag["QsolarAbs_W"][k] + result.scalar_diag["QsolarGlass_W"][k])
        q_use_model = float(result.scalar_diag["Quseful_W"][k])
        q_loss = float(result.scalar_diag["Qloss_W"][k])
        q_storage = float(result.scalar_diag["Qstorage_est_W"][k])
        q_conv = float(np.sum(result.node_diag["Qconv_external_W"][k, :]))
        q_rad = float(np.sum(result.node_diag["Qrad_sky_W"][k, :]))
        q_support = float(np.sum(result.node_diag["Qsupports_W"][k, :]))

        # Dos preguntas contrafactuales de diagnóstico.
        # 1) Si las pérdidas del modelo fueran correctas, ¿cuánto debería cambiar
        #    la potencia solar absorbida para alcanzar Quse_ref?
        f_opt = (q_use_ref + q_loss) / q_solar if q_solar > 1e-12 else float("nan")

        # 2) Si la potencia solar absorbida fuera correcta, ¿cuánto deberían
        #    cambiar las pérdidas térmicas para alcanzar Quse_ref?
        q_loss_required = q_solar - q_use_ref
        f_loss = q_loss_required / q_loss if q_loss > 1e-12 else float("nan")

        eta_model = float(result.scalar_diag.get("eta_dni_basis_pct", result.scalar_diag["eta_pct"])[k])
        energy_closure = q_solar - q_loss - q_use_model - q_storage

        rows.append(
            {
                "Mes": MONTH_ABBR_ES[i],
                "Mes_num": month,
                "Tin_C": tin_c,
                "Tout_ref_C": tout_ref_c,
                "Tout_modelo_C": tout_model_c,
                "Tamb_C": float(data["Tamb_C"][i]),
                "DNI_W_m2": dni,
                "mdot_kg_s": mdot,
                "Eta_ref_pct": eta_ref,
                "Eta_modelo_pct": eta_model,
                "Err_Eta_pct": _relative_error(eta_model, eta_ref),
                "Qsolar_absorbida_W": q_solar,
                "Qutil_modelo_W": q_use_model,
                "Qutil_referencia_W": q_use_ref,
                "Qloss_modelo_W": q_loss,
                "Qconv_ext_W": q_conv,
                "Qrad_cielo_W": q_rad,
                "Qsoportes_W": q_support,
                "Qstorage_W": q_storage,
                "Cierre_balance_W": energy_closure,
                "Perdidas_sobre_Qsolar_pct": 100.0 * q_loss / q_solar if q_solar > 1e-12 else float("nan"),
                "Util_modelo_sobre_Qsolar_pct": 100.0 * q_use_model / q_solar if q_solar > 1e-12 else float("nan"),
                "Util_ref_sobre_Qsolar_pct": 100.0 * q_use_ref / q_solar if q_solar > 1e-12 else float("nan"),
                "Qloss_requerido_W": q_loss_required,
                "Factor_optico_requerido": f_opt,
                "Factor_perdidas_requerido": f_loss,
                "Optica_insuficiente_aun_sin_perdidas": bool(q_loss_required < 0.0),
            }
        )

    table = pd.DataFrame(rows)
    valid_loss_factor = table.loc[table["Factor_perdidas_requerido"] >= 0.0, "Factor_perdidas_requerido"]
    opt_factor = table["Factor_optico_requerido"]

    metrics = {
        "eta_rmse_pp": float(np.sqrt(np.mean(np.square(table["Eta_modelo_pct"] - table["Eta_ref_pct"])))),
        "eta_mae_pp": float(np.mean(np.abs(table["Eta_modelo_pct"] - table["Eta_ref_pct"]))),
        "opt_factor_mean": float(np.mean(opt_factor)),
        "opt_factor_min": float(np.min(opt_factor)),
        "opt_factor_max": float(np.max(opt_factor)),
        "opt_factor_cv_pct": 100.0 * _cv(opt_factor),
        "loss_factor_mean": float(np.mean(valid_loss_factor)) if len(valid_loss_factor) else float("nan"),
        "loss_factor_min": float(np.min(valid_loss_factor)) if len(valid_loss_factor) else float("nan"),
        "loss_factor_max": float(np.max(valid_loss_factor)) if len(valid_loss_factor) else float("nan"),
        "loss_factor_cv_pct": 100.0 * _cv(valid_loss_factor),
        "months_optically_insufficient": int(table["Optica_insuficiente_aun_sin_perdidas"].sum()),
        "max_storage_fraction_pct": float(
            100.0 * np.max(np.abs(table["Qstorage_W"]) / np.maximum(table["Qsolar_absorbida_W"], 1e-12))
        ),
        "max_balance_closure_W": float(np.max(np.abs(table["Cierre_balance_W"]))),
        "loss_factor_crosses_one": bool(
            len(valid_loss_factor) > 0 and np.min(valid_loss_factor) < 1.0 < np.max(valid_loss_factor)
        ),
        "opt_factor_crosses_one": bool(np.min(opt_factor) < 1.0 < np.max(opt_factor)),
    }

    # Diagnóstico textual deliberadamente conservador: no "identifica" física,
    # sólo decide qué hipótesis merece la siguiente prueba.
    messages: list[str] = []
    if metrics["months_optically_insufficient"] > 0:
        messages.append(
            "En uno o más meses la potencia solar absorbida del modelo sería insuficiente incluso con pérdidas térmicas nulas; "
            "el bloque óptico/entrada solar debe revisarse antes que las pérdidas."
        )
    if metrics["opt_factor_cv_pct"] <= 5.0:
        messages.append(
            "El factor óptico requerido es casi constante entre meses; una corrección óptica global podría ser plausible."
        )
    elif metrics["opt_factor_crosses_one"]:
        messages.append(
            "El factor óptico requerido cambia de >1 a <1 según el mes; una sola constante óptica no puede corregir la forma estacional."
        )
    if metrics["loss_factor_cv_pct"] <= 10.0:
        messages.append(
            "El factor de pérdidas requerido es relativamente estable; revisar un coeficiente global de pérdidas podría ser razonable."
        )
    elif metrics["loss_factor_crosses_one"]:
        messages.append(
            "El factor de pérdidas requerido cruza 1: en algunos meses el modelo pierde demasiado y en otros demasiado poco. "
            "Eso apunta a una dependencia térmica/ambiental incorrecta, no a un único multiplicador de pérdidas."
        )
    if metrics["max_storage_fraction_pct"] < 0.5:
        messages.append(
            "El almacenamiento al final de cada corrida es despreciable frente a la potencia absorbida; la lectura cuasiestacionaria de esta auditoría es válida."
        )

    return {
        "city": city_name,
        "case": expected_case,
        "table": table,
        "metrics": metrics,
        "parameter_source": parameter_source,
        "diagnosis": messages,
        "note": (
            "Prueba diagnóstica sin optimización. F_opt responde: 'si mantengo las pérdidas actuales, ¿cuánto tendría que escalar la potencia solar absorbida?'. "
            "F_loss responde: 'si mantengo la potencia solar absorbida actual, ¿cuánto tendría que escalar las pérdidas?'. "
            "Los factores no son parámetros físicos identificados; sólo permiten decidir qué bloque merece la siguiente prueba."
        ),
    }
