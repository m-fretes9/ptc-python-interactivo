"""Prueba diagnóstica independiente de la Ec. (10) de Rea Quille.

Este módulo está separado de ``validations.py`` a propósito. De ese modo la
función usada por ``app.py`` no depende de que una actualización parcial de
``validations.py`` haya sido aplicada correctamente en un despliegue Streamlit.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from fluid_properties import FluidPropertyEvaluator
from presets import (
    MONTH_ABBR_ES,
    REA_ALVORADA_MONTHLY,
    REA_FOZ_MONTHLY,
    build_rea_monthly_preset,
)
from ptc_model import PTCSimulator


def _relative_error(sim: float, ref: float) -> float:
    if not np.isfinite(sim) or not np.isfinite(ref):
        return float("nan")
    return 100.0 * abs(sim - ref) / max(abs(ref), np.finfo(float).eps)


def _set_effective_optical_efficiency(cfg: dict[str, Any], target: float) -> None:
    optics = cfg["optics"]
    absorber = cfg["materials"]["absorber"]
    rest = (
        float(optics["intercept_factor"])
        * float(absorber["alpha"])
        * float(optics.get("dirt_factor", 1.0))
        * float(optics.get("shade_factor", 1.0))
    )
    if bool(cfg["model"].get("has_glass", False)):
        rest *= float(cfg["materials"]["glass"]["tau"])
    if rest <= 0.0:
        raise ValueError("No se puede imponer eta_opt,ef porque el producto óptico restante es nulo.")
    rho = float(target) / rest
    if not (0.0 < rho <= 1.0 + 1e-12):
        raise ValueError(
            f"eta_opt,ef={target:.5f} exigiría reflectividad={rho:.5f}, fuera de (0,1]."
        )
    optics["reflectivity"] = float(np.clip(rho, 1e-8, 1.0))


def _apply_calibrated_parameters(
    cfg: dict[str, Any],
    db: dict[str, Any],
    parameter_ids: Sequence[str],
    values: Sequence[float],
) -> None:
    """Aplica el subconjunto de parámetros que puede producir el inverso V14.x."""
    for pid, raw_value in zip(parameter_ids, values):
        value = float(raw_value)
        if pid == "eta_opt_eff":
            _set_effective_optical_efficiency(cfg, value)
        elif pid == "eps_abs":
            cfg["materials"]["absorber"]["eps"] = value
        elif pid == "eps_glass":
            cfg["materials"]["glass"]["eps"] = value
        elif pid == "wind_m_s":
            cfg["environment"]["wind_m_s"] = value
        elif pid == "dew_point_C":
            cfg["environment"]["dew_point_C"] = value
        elif pid == "sky_delta_K":
            cfg["environment"]["sky_delta_K"] = value
        elif pid == "support_loss_fraction":
            cfg["model"]["support_loss_fraction"] = value
            cfg["model"]["include_supports"] = bool(value > 1e-12)
        elif pid.startswith("fluid_"):
            prop = pid.split("_", 1)[1]
            fluid_name = str(cfg["operation"]["fluid"])
            db[fluid_name]["multipliers"][prop] = value
        else:
            raise ValueError(f"Parámetro calibrado no soportado por la prueba Ec. (10): {pid}")


def validate_rea_eq10_from_tout(
    city: str,
    fluid_database: Mapping[str, Mapping[str, Any]],
    calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Ejecuta la prueba simple de Rea Quille reconstruyendo eta desde Tout.

    No calibra nada. Para cada mes usa Tin, Tamb, DNI y mdot publicados, ejecuta
    el PTC y reconstruye:

        eta = mdot * Cp * (Tout - Tin) / (Aa * DNI)

    La eficiencia publicada se recalcula también desde Tout_ref para comprobar
    la consistencia de las Tablas 10/11 con la Ec. (10).
    """
    city_key = city.strip().lower()
    if city_key.startswith("foz"):
        data = REA_FOZ_MONTHLY
        city_name = "Foz do Iguaçu"
        expected_case = "rea_foz"
    elif city_key.startswith("alvorada"):
        data = REA_ALVORADA_MONTHLY
        city_name = "Alvorada do Norte"
        expected_case = "rea_alvorada"
    else:
        raise ValueError(f"Ciudad no soportada: {city}")

    calibration_table = None
    parameter_source = "Parámetros nominales del preset Rea Quille"
    if calibration is not None:
        if not isinstance(calibration, Mapping):
            raise ValueError("La calibración suministrada no es válida.")
        if str(calibration.get("case", "")) != expected_case:
            raise ValueError(
                f"La última calibración corresponde a {calibration.get('case', 'otro caso')}, no a {expected_case}."
            )
        calibration_table = calibration.get("parameter_table")
        if not isinstance(calibration_table, pd.DataFrame) or calibration_table.empty:
            raise ValueError("La calibración no contiene parámetros identificados aplicables.")
        parameter_source = "Última calibración guardada para esta ciudad"

    rows: list[dict[str, Any]] = []
    water = FluidPropertyEvaluator("Agua", fluid_database)

    for i in range(12):
        cfg, _ = build_rea_monthly_preset(city_name, i + 1)
        db = deepcopy(fluid_database)

        if calibration_table is not None:
            required = {"ID", "Identificado"}
            if not required.issubset(calibration_table.columns):
                raise ValueError("La tabla de calibración no contiene ID e Identificado.")
            parameter_ids = [str(value) for value in calibration_table["ID"].tolist()]
            values = [float(value) for value in calibration_table["Identificado"].tolist()]
            _apply_calibrated_parameters(cfg, db, parameter_ids, values)

        result = PTCSimulator(cfg, db).simulate()
        k = len(result.t_s) - 1

        tin_c = float(data["Tin_C"][i])
        tout_ref_c = float(data["Tout_ref_C"][i])
        tout_python_c = float(result.Tout_C[k])
        dni = float(data["DNI_W_m2"][i])
        mdot = float(data["mdot_kg_s"][i])
        eta_ref_published = float(data["eta_ref_pct"][i])
        area = float(cfg["geometry"]["W"]) * float(cfg["geometry"]["L"])

        cp_ref = float(water(0.5 * (tin_c + tout_ref_c) + 273.15).Cp)
        cp_python = float(water(0.5 * (tin_c + tout_python_c) + 273.15).Cp)
        q_ref_eq10 = mdot * cp_ref * (tout_ref_c - tin_c)
        q_python_eq10 = mdot * cp_python * (tout_python_c - tin_c)
        eta_ref_eq10 = 100.0 * q_ref_eq10 / (area * dni) if dni > 0 else float("nan")
        eta_python_eq10 = 100.0 * q_python_eq10 / (area * dni) if dni > 0 else float("nan")
        eta_python_native = float(result.scalar_diag["eta_pct"][k])
        eta_python_dni_basis = float(
            result.scalar_diag.get("eta_dni_basis_pct", result.scalar_diag["eta_pct"])[k]
        )

        rows.append(
            {
                "Mes": MONTH_ABBR_ES[i],
                "Tin_ref_C": tin_c,
                "Tout_ref_C": tout_ref_c,
                "Tout_Python_C": tout_python_c,
                "Tamb_ref_C": float(data["Tamb_C"][i]),
                "DNI_ref_W_m2": dni,
                "mdot_ref_kg_s": mdot,
                "Cp_ref_J_kgK": cp_ref,
                "Cp_Python_J_kgK": cp_python,
                "Eta_ref_publicada_pct": eta_ref_published,
                "Eta_ref_recalculada_Eq10_pct": eta_ref_eq10,
                "Eta_Python_Eq10_desde_Tout_pct": eta_python_eq10,
                "Eta_Python_interna_pct": eta_python_native,
                "Eta_Python_DNI_basis_pct": eta_python_dni_basis,
                "Delta_ref_publicada_vs_Eq10_pp": eta_ref_eq10 - eta_ref_published,
                "Delta_Python_interna_vs_Eq10_pp": eta_python_native - eta_python_eq10,
                "Err_Tout_pct": _relative_error(tout_python_c, tout_ref_c),
                "Err_Eta_Eq10_pct": _relative_error(eta_python_eq10, eta_ref_published),
            }
        )

    table = pd.DataFrame(rows)
    eta_delta = table["Eta_Python_Eq10_desde_Tout_pct"] - table["Eta_ref_publicada_pct"]
    tout_delta = table["Tout_Python_C"] - table["Tout_ref_C"]
    metrics = {
        "eta_mae_pp": float(np.mean(np.abs(eta_delta))),
        "eta_rmse_pp": float(np.sqrt(np.mean(np.square(eta_delta)))),
        "eta_mape_pct": float(np.mean(table["Err_Eta_Eq10_pct"])),
        "eta_bias_pp": float(np.mean(eta_delta)),
        "tout_mae_C": float(np.mean(np.abs(tout_delta))),
        "tout_rmse_C": float(np.sqrt(np.mean(np.square(tout_delta)))),
        "tout_mape_pct": float(np.mean(table["Err_Tout_pct"])),
        "max_reference_eq10_mismatch_pp": float(
            np.max(np.abs(table["Delta_ref_publicada_vs_Eq10_pp"]))
        ),
        "max_python_native_eq10_mismatch_pp": float(
            np.max(np.abs(table["Delta_Python_interna_vs_Eq10_pp"]))
        ),
    }

    return {
        "city": city_name,
        "case": expected_case,
        "table": table,
        "metrics": metrics,
        "parameter_source": parameter_source,
        "note": (
            "Prueba diagnóstica sin optimización: el simulador sólo aporta Tout. La eficiencia Python se reconstruye después "
            "con mdot·Cp·(Tout−Tin)/(Aa·DNI), exactamente sobre la base de la Ec. (10) usada por Rea Quille. "
            "La columna de eficiencia interna se conserva únicamente para comprobar si una diferencia de definición del KPI "
            "era responsable de la forma de la curva."
        ),
    }
