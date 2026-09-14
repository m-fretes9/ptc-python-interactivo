"""Casos de comparación y validación contra las fuentes documentales.

La regla central de este módulo es no ocultar lagunas de la fuente. Los presets
pueden contener hipótesis necesarias para ejecutar nuestro modelo; esas hipótesis
se reportan en ``preset_meta`` y las validaciones no las convierten en datos
experimentales.
"""

from __future__ import annotations

from copy import deepcopy
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from fluid_properties import FluidPropertyEvaluator
from presets import (
    MONTH_ABBR_ES,
    REA_ALVORADA_MONTHLY,
    REA_FOZ_MONTHLY,
    REA_PROTOTYPE_HOURS,
    build_bhambare_sukhatme_preset,
    build_rea_monthly_preset,
)
from ptc_model import PTCSimulator


def _relative_error(sim: float, ref: float) -> float:
    if not np.isfinite(sim) or not np.isfinite(ref):
        return float("nan")
    return 100.0 * abs(sim - ref) / max(abs(ref), np.finfo(float).eps)


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.nanargmin(np.abs(np.asarray(values, dtype=float) - float(target))))


def validate_bhambare(
    base_config: Mapping[str, Any] | None,
    fluid_database: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Reproduce el caso de validación Bhambare/Sukhatme con el preset completo."""
    # El preset define la configuración documental completa. Se usa la base de
    # propiedades recibida para que el usuario pueda estudiar sensibilidad.
    cfg, _ = build_bhambare_sukhatme_preset()
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
            "Referencia_Sukhatme": reference,
            "Modelo_Bhambare": article_model,
            "Modelo_Python": simulation,
            "Error_vs_Sukhatme_pct": error_pct,
        }
    )
    return {
        "table": table,
        "result": result,
        "config": cfg,
        "note": (
            "Q_util y eta de Sukhatme se derivan de Tout, mdot y Cp(T), porque no están tabulados directamente. "
            "El preset interpreta Ib=705 W/m² como irradiancia efectiva constante sobre la apertura para obtener un estado comparable."
        ),
    }



def compare_bhambare_solvers(
    fluid_database: Mapping[str, Mapping[str, Any]],
    methods: Sequence[str] = ("RK45", "Radau", "BDF"),
) -> dict[str, Any]:
    """Compara RK45, Radau y BDF sobre exactamente el mismo caso Bhambare/Sukhatme.

    El objetivo es aislar el efecto del integrador temporal: geometría, óptica,
    propiedades, condiciones iniciales, tolerancias y paso máximo permanecen
    idénticos. También se reporta el residuo térmico final para distinguir entre
    coincidencia de la variable observada y verdadero acercamiento al estado
    estacionario.
    """
    allowed = {"RK45", "Radau", "BDF"}
    requested = [str(method).strip() for method in methods]
    invalid = [method for method in requested if method not in allowed]
    if invalid:
        raise ValueError(f"Solvers no soportados: {invalid}. Use RK45, Radau o BDF.")
    if not requested:
        raise ValueError("Debe indicarse al menos un solver para la comparación.")

    base_cfg, _ = build_bhambare_sukhatme_preset()
    ref = base_cfg["preset_meta"]["reference"]

    Tref_glass_K = float(ref["Tglass_book_K"])
    Tref_abs_K = float(ref["Tabs_book_K"])
    Tref_out_C = float(ref["Tout_book_C"])
    Qref_loss_W = float(ref["Qloss_book_W"])

    properties = FluidPropertyEvaluator("ParathermNF", fluid_database)
    prop_mean_ref = properties(0.5 * (float(ref["Tin_C"]) + Tref_out_C) + 273.15)
    Qref_useful_W = (
        float(ref["mdot_kg_s"])
        * prop_mean_ref.Cp
        * (Tref_out_C - float(ref["Tin_C"]))
    )
    eta_ref_pct = 100.0 * Qref_useful_W / (
        float(ref["beam_W_m2"])
        * float(base_cfg["geometry"]["W"])
        * float(base_cfg["geometry"]["L"])
    )

    sukhatme = {
        "Tvid_K": Tref_glass_K,
        "Tabs_K": Tref_abs_K,
        "Tout_C": Tref_out_C,
        "Qloss_W": Qref_loss_W,
        "Qutil_W": Qref_useful_W,
        "eta_pct": eta_ref_pct,
    }
    bhambare = {
        "Tvid_K": float(ref["Tglass_article_K"]),
        "Tabs_K": float(ref["Tabs_article_K"]),
        "Tout_C": float(ref["Tout_article_C"]),
        "Qloss_W": float(ref["Qloss_article_W"]),
        "Qutil_W": float("nan"),
        "eta_pct": float("nan"),
    }

    results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []

    for method in requested:
        cfg = deepcopy(base_cfg)
        cfg["solver"]["method"] = method
        started = perf_counter()
        result = PTCSimulator(cfg, fluid_database).simulate()
        elapsed_s = perf_counter() - started
        results[method] = result
        k = len(result.t_s) - 1

        final_rates = np.concatenate(
            [
                np.asarray(result.node_diag["dTf_dt_K_s"][k], dtype=float),
                np.asarray(result.node_diag["dTabs_dt_K_s"][k], dtype=float),
                np.asarray(result.node_diag["dTglass_dt_K_s"][k], dtype=float),
            ]
        )
        max_residual = float(np.nanmax(np.abs(final_rates)))

        row = {
            "Solver": method,
            "Tvid_K": float(result.Tglass_mean_C[k] + 273.15),
            "Tabs_K": float(result.Tabs_mean_C[k] + 273.15),
            "Tout_C": float(result.Tout_C[k]),
            "Qloss_W": float(result.scalar_diag["Qloss_W"][k]),
            "Qutil_W": float(result.scalar_diag["Quseful_W"][k]),
            "eta_pct": float(result.scalar_diag["eta_pct"][k]),
        }
        for key in ("Tvid_K", "Tabs_K", "Tout_C", "Qloss_W", "Qutil_W", "eta_pct"):
            row[f"Err_{key}_vs_Sukhatme_pct"] = _relative_error(row[key], sukhatme[key])
        rows.append(row)
        performance_rows.append(
            {
                "Solver": method,
                "Tiempo_CPU_s": elapsed_s,
                "nfev": int(result.nfev),
                "njev": int(result.njev),
                "nlu": int(result.nlu),
                "max_abs_dTdt_final_K_s": max_residual,
                "Mensaje": result.solver_message,
            }
        )

    solver_table = pd.DataFrame(rows)
    performance_table = pd.DataFrame(performance_rows)

    # Matriz compacta para contrastar directamente las referencias y los tres solvers.
    magnitude_specs = [
        ("T_vidrio_K", "Tvid_K"),
        ("T_absorbedor_K", "Tabs_K"),
        ("T_salida_C", "Tout_C"),
        ("Q_perdidas_W", "Qloss_W"),
        ("Q_util_derivado_W", "Qutil_W"),
        ("eta_derivada_pct", "eta_pct"),
    ]
    comparison_data: dict[str, Any] = {
        "Magnitud": [label for label, _ in magnitude_specs],
        "Referencia_Sukhatme": [sukhatme[key] for _, key in magnitude_specs],
        "Modelo_Bhambare": [bhambare[key] for _, key in magnitude_specs],
    }
    for method in requested:
        solver_row = solver_table.loc[solver_table["Solver"] == method].iloc[0]
        comparison_data[method] = [float(solver_row[key]) for _, key in magnitude_specs]
    comparison_table = pd.DataFrame(comparison_data)

    spread_specs = {
        "Tout_span_C": "Tout_C",
        "Tabs_span_K": "Tabs_K",
        "Tvid_span_K": "Tvid_K",
        "Qloss_span_W": "Qloss_W",
        "Qutil_span_W": "Qutil_W",
        "eta_span_pp": "eta_pct",
    }
    spreads: dict[str, float] = {}
    for name, column in spread_specs.items():
        values = solver_table[column].to_numpy(dtype=float)
        spreads[name] = float(np.nanmax(values) - np.nanmin(values))
    spreads["max_solver_relative_spread_pct"] = float(
        max(
            100.0
            * (np.nanmax(solver_table[column]) - np.nanmin(solver_table[column]))
            / max(abs(float(np.nanmean(solver_table[column]))), np.finfo(float).eps)
            for column in ("Tout_C", "Tabs_K", "Tvid_K", "Qloss_W", "Qutil_W", "eta_pct")
        )
    )

    return {
        "solver_table": solver_table,
        "performance_table": performance_table,
        "comparison_table": comparison_table,
        "results": results,
        "spreads": spreads,
        "config": base_cfg,
        "note": (
            "Los tres métodos resuelven exactamente el mismo caso Bhambare/Sukhatme con las mismas tolerancias y el mismo paso máximo. "
            "Por lo tanto, si sus estados finales coinciden y la discrepancia con la referencia permanece, esa diferencia no puede atribuirse al solver por sí sola. "
            "El residuo max|dT/dt| final se incluye para comprobar cuánto se acercó cada integración al estado estacionario."
        ),
    }


def validate_rea_quille_city_monthly(
    city: str,
    fluid_database: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Ejecuta los 12 presets mensuales de Rea Quille para una ciudad."""
    city_key = city.strip().lower()
    if city_key.startswith("foz"):
        data = REA_FOZ_MONTHLY
        city_name = "Foz do Iguaçu"
    elif city_key.startswith("alvorada"):
        data = REA_ALVORADA_MONTHLY
        city_name = "Alvorada do Norte"
    else:
        raise ValueError(f"Ciudad no soportada: {city}")

    n = 12
    Tout_sim_C = np.full(n, np.nan)
    Quseful_sim_W = np.full(n, np.nan)
    Qloss_sim_W = np.full(n, np.nan)
    eta_sim_pct = np.full(n, np.nan)
    Quseful_ref_W = np.full(n, np.nan)
    water = FluidPropertyEvaluator("Agua", fluid_database)

    for i in range(n):
        cfg, _ = build_rea_monthly_preset(city_name, i + 1)
        result = PTCSimulator(cfg, fluid_database).simulate()
        k = len(result.t_s) - 1
        Tout_sim_C[i] = result.Tout_C[k]
        Quseful_sim_W[i] = result.scalar_diag["Quseful_W"][k]
        Qloss_sim_W[i] = result.scalar_diag["Qloss_W"][k]
        eta_sim_pct[i] = result.scalar_diag["eta_pct"][k]
        prop_ref = water(0.5 * (data["Tin_C"][i] + data["Tout_ref_C"][i]) + 273.15)
        Quseful_ref_W[i] = (
            data["mdot_kg_s"][i]
            * prop_ref.Cp
            * (data["Tout_ref_C"][i] - data["Tin_C"][i])
        )

    Tin_C = np.asarray(data["Tin_C"], dtype=float)
    Tout_ref_C = np.asarray(data["Tout_ref_C"], dtype=float)
    Tamb_C = np.asarray(data["Tamb_C"], dtype=float)
    DNI = np.asarray(data["DNI_W_m2"], dtype=float)
    mdot = np.asarray(data["mdot_kg_s"], dtype=float)
    eta_ref_pct = np.asarray(data["eta_ref_pct"], dtype=float)

    err_Tout_pct = 100.0 * np.abs(Tout_sim_C - Tout_ref_C) / np.maximum(np.abs(Tout_ref_C), np.finfo(float).eps)
    err_Quseful_pct = 100.0 * np.abs(Quseful_sim_W - Quseful_ref_W) / np.maximum(np.abs(Quseful_ref_W), np.finfo(float).eps)
    err_eta_pct = 100.0 * np.abs(eta_sim_pct - eta_ref_pct) / np.maximum(np.abs(eta_ref_pct), np.finfo(float).eps)

    table = pd.DataFrame(
        {
            "Mes": MONTH_ABBR_ES,
            "Tin_C": Tin_C,
            "Tout_ref_C": Tout_ref_C,
            "Tout_Python_C": Tout_sim_C,
            "Err_Tout_pct": err_Tout_pct,
            "Tamb_C": Tamb_C,
            "DNI_ref_W_m2": DNI,
            "mdot_ref_kg_s": mdot,
            "Qutil_ref_derivado_W": Quseful_ref_W,
            "Qutil_Python_W": Quseful_sim_W,
            "Qloss_Python_W": Qloss_sim_W,
            "Eta_ref_pct": eta_ref_pct,
            "Eta_Python_pct": eta_sim_pct,
            "Diferencia_eta_pp": eta_sim_pct - eta_ref_pct,
            "Err_Qutil_pct": err_Quseful_pct,
            "Err_Eta_pct": err_eta_pct,
        }
    )
    metrics = {
        "MAPE_Tout_pct": float(np.mean(err_Tout_pct)),
        "MAPE_Qutil_pct": float(np.mean(err_Quseful_pct)),
        "MAPE_eta_pct": float(np.mean(err_eta_pct)),
        "Bias_eta_pp": float(np.mean(eta_sim_pct - eta_ref_pct)),
    }
    return {
        "table": table,
        "metrics": metrics,
        "city": city_name,
        "note": (
            "Rea Quille publica promedios mensuales, no días ni series horarias de entrada para estas Tablas. "
            "Cada fila se ejecuta como un estado cuasiestacionario con Tin, Tamb, irradiación y mdot exactamente tabulados. "
            "Viento, pérdidas de cielo y parámetros ópticos no publicados permanecen como hipótesis explícitas del preset; por eso esta comparación diagnostica también esas lagunas."
        ),
    }


def validate_tcc_monthly(
    base_config: Mapping[str, Any] | None,
    fluid_database: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Alias histórico: validación mensual de Foz do Iguaçu (Tabela 10)."""
    return validate_rea_quille_city_monthly("Foz do Iguaçu", fluid_database)


def prototype_tcc_table() -> dict[str, Any]:
    hours = np.asarray(REA_PROTOTYPE_HOURS["hours"], dtype=int)
    eta_exp = np.asarray(REA_PROTOTYPE_HOURS["eta_exp_pct"], dtype=float)
    eta_trnsys = np.asarray(REA_PROTOTYPE_HOURS["eta_trnsys_pct"], dtype=float)
    difference_pp = np.abs(eta_trnsys - eta_exp)
    error_relative_pct = 100.0 * difference_pp / np.maximum(np.abs(eta_exp), np.finfo(float).eps)
    rmse = float(np.sqrt(np.mean((eta_trnsys - eta_exp) ** 2)))
    mape = float(np.mean(error_relative_pct))
    table = pd.DataFrame(
        {
            "Hora": [f"{hour:02d}:00" for hour in hours],
            "mdot_kg_s": np.asarray(REA_PROTOTYPE_HOURS["mdot_kg_s"], dtype=float),
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
        "eta_exp_mean_pct": float(np.mean(eta_exp)),
        "eta_trnsys_mean_pct": float(np.mean(eta_trnsys)),
        "note": (
            "La Tabela 8 no publica Tin ni Tout por hora. Por ello estos vectores son referencia experimental/TRNSYS, "
            "pero no bastan para una validación Python estricta del balance m*Cp*(Tout-Tin)."
        ),
    }


def validate_active_preset(
    config: Mapping[str, Any],
    fluid_database: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compara la configuración activa con la referencia asociada a su preset."""
    cfg = deepcopy(config)
    meta = cfg.get("preset_meta") or {}
    family = str(meta.get("family", ""))
    reference = meta.get("reference") or {}
    if not reference:
        raise ValueError("El caso activo no contiene una referencia documental asociada.")

    # Bhambare tiene una tabla multivariable específica.
    if family == "Bhambare / Sukhatme":
        return {"kind": "bhambare", **validate_bhambare(cfg, fluid_database)}

    result = PTCSimulator(cfg, fluid_database).simulate()
    k = len(result.t_s) - 1

    if family == "Rea Quille / Fiamonzini prototipo":
        hours = np.asarray(REA_PROTOTYPE_HOURS["hours"], dtype=float)
        eta_exp = np.asarray(REA_PROTOTYPE_HOURS["eta_exp_pct"], dtype=float)
        eta_trnsys = np.asarray(REA_PROTOTYPE_HOURS["eta_trnsys_pct"], dtype=float)
        eta_python = np.full_like(hours, np.nan, dtype=float)
        for i, hour in enumerate(hours):
            idx = _nearest_index(result.LAT_h, hour)
            eta_python[i] = float(result.scalar_diag["eta_pct"][idx])
        table = pd.DataFrame(
            {
                "Hora": [f"{int(h):02d}:00" for h in hours],
                "Eta_experimental_pct": eta_exp,
                "Eta_TRNSYS_pct": eta_trnsys,
                "Eta_Python_con_Tin_asumida_pct": eta_python,
                "Python_minus_exp_pp": eta_python - eta_exp,
            }
        )
        return {
            "kind": "prototype",
            "table": table,
            "result": result,
            "metrics": {
                "Eta_exp_media_pct": float(np.mean(eta_exp)),
                "Eta_TRNSYS_media_pct": float(np.mean(eta_trnsys)),
                "Eta_Python_media_pct": float(np.nanmean(eta_python)),
                "Bias_Python_vs_exp_pp": float(np.nanmean(eta_python - eta_exp)),
            },
            "note": (
                "COMPARACIÓN EXPLORATORIA, NO VALIDACIÓN ESTRICTA: Rea Quille no publica Tin/Tout horarios en la Tabela 8. "
                f"El preset activo usa Tin={cfg['operation']['Tin_K'] - 273.15:.2f} °C como hipótesis editable. "
                "Los valores Python se muestrean a las horas de la tabla para cuantificar la discrepancia sin ocultar esa limitación."
            ),
        }

    if family in {"Rea Quille mensual", "Rea Quille anual"}:
        fluid = str(cfg["operation"]["fluid"])
        evaluator = FluidPropertyEvaluator(fluid, fluid_database)
        tin_ref = float(reference["Tin_C"])
        tout_ref = float(reference["Tout_C"])
        mdot_ref = float(reference["mdot_kg_s"])
        prop_ref = evaluator(0.5 * (tin_ref + tout_ref) + 273.15)
        q_ref = mdot_ref * prop_ref.Cp * (tout_ref - tin_ref)

        values = [
            ("T_salida_C", tout_ref, float(result.Tout_C[k])),
            ("Eta_termica_pct", float(reference["eta_pct"]), float(result.scalar_diag["eta_pct"][k])),
            ("Q_util_derivado_W", q_ref, float(result.scalar_diag["Quseful_W"][k])),
        ]
        table = pd.DataFrame(
            {
                "Magnitud": [row[0] for row in values],
                "Referencia_Rea_Quille": [row[1] for row in values],
                "Modelo_Python": [row[2] for row in values],
                "Diferencia": [row[2] - row[1] for row in values],
                "Error_rel_pct": [_relative_error(row[2], row[1]) for row in values],
            }
        )
        return {
            "kind": "rea_monthly",
            "table": table,
            "result": result,
            "note": (
                "La fila de Rea Quille se compara con un estado cuasiestacionario que usa exactamente Tin, Tamb, irradiación y mdot tabulados. "
                "La comparación no es estrictamente equivalente al TRNSYS anual porque la fuente no publica la serie meteorológica ni todos los parámetros ópticos y de pérdidas del Type1358."
            ),
        }

    raise ValueError(f"Tipo de preset no soportado por la validación activa: {family}")
