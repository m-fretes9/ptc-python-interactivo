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
from scipy.optimize import least_squares

from fluid_properties import FluidPropertyEvaluator
from presets import (
    MONTH_ABBR_ES,
    REA_ALVORADA_MONTHLY,
    REA_FOZ_MONTHLY,
    REA_PROTOTYPE_HOURS,
    REA_PROTOTYPE_USER_EXPORT_2026_09_14,
    build_bhambare_sukhatme_preset,
    build_rea_monthly_preset,
    build_rea_prototype_preset,
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



IDENTIFIED_PARAMETER_TEMPLATES_2026_09_14: dict[str, dict[str, Any]] = {
    "bhambare": {
        "label": "Bhambare/Sukhatme · modelo inverso 14/09/2026",
        "eta_opt_eff": 0.7671242147795794,
        "eps_abs": 0.989999998086766,
        "eps_glass": 0.9899999947177927,
        "source": "ptc_modelo_inverso_bhambare.xlsx",
        "near_bounds": ["eta_opt_eff", "eps_abs", "eps_glass"],
    },
    "rea_foz": {
        "label": "Rea Quille · Foz · modelo inverso 14/09/2026",
        "eta_opt_eff": 0.5988709459872553,
        "wind_m_s": 0.000453171255123984,
        "source": "ptc_modelo_inverso_rea_foz.xlsx",
    },
    "rea_alvorada": {
        "label": "Rea Quille · Alvorada · modelo inverso 14/09/2026",
        "eta_opt_eff": 0.582784559681394,
        "wind_m_s": 0.01046720113608728,
        "source": "ptc_modelo_inverso_rea_alvorada.xlsx",
    },
    "rea_prototype": {
        "label": "Rea Quille/Fiamonzini · Tabela 8 · modelo inverso 14/09/2026",
        "eta_opt_eff": 0.5583394106143328,
        "wind_m_s": 6.32658029702888,
        "source": "ptc_modelo_inverso_rea_prototype_tab_8.xlsx",
    },
}

# Salidas exactas guardadas del XLSX entregado por el usuario. Se usan como
# benchmark de regresión, no como una segunda fuente experimental.
BHAMBARE_USER_INVERSE_2026_09_14: dict[str, dict[str, float]] = {
    "initial": {
        "Qloss_W": 693.8508069287897,
        "Tabs_K": 450.4787451823364,
        "Tout_C": 156.0717331741304,
        "Tvid_K": 328.1041422446137,
    },
    "identified": {
        "Qloss_W": 819.1450608851163,
        "Tabs_K": 455.1953510611756,
        "Tout_C": 157.125172030547,
        "Tvid_K": 331.730769053495,
    },
}


def identified_parameter_template(case: str) -> dict[str, Any]:
    key = str(case).strip().lower()
    if key not in IDENTIFIED_PARAMETER_TEMPLATES_2026_09_14:
        raise ValueError(f"No hay template identificado guardado para: {case}")
    return deepcopy(IDENTIFIED_PARAMETER_TEMPLATES_2026_09_14[key])


def apply_identified_parameter_template(cfg: dict[str, Any], case: str) -> dict[str, Any]:
    """Aplica al config los parámetros identificados guardados del 14/09/2026.

    Los valores son templates de calibración reproducibles, no propiedades universales
    del colector. En particular, los vientos casi nulos de los casos mensuales deben
    interpretarse como señal de compensación/identificabilidad, no como mediciones.
    """
    spec = identified_parameter_template(case)
    if "eta_opt_eff" in spec:
        _set_effective_optical_efficiency(cfg, float(spec["eta_opt_eff"]))
    if "wind_m_s" in spec:
        cfg["environment"]["wind_m_s"] = float(spec["wind_m_s"])
    if "eps_abs" in spec:
        cfg["materials"]["absorber"]["eps"] = float(spec["eps_abs"])
    if "eps_glass" in spec:
        if not bool(cfg["model"].get("has_glass", False)):
            raise ValueError("El template identificado requiere una cubierta de vidrio.")
        cfg["materials"]["glass"]["eps"] = float(spec["eps_glass"])
    cfg.setdefault("preset_meta", {})["identified_template"] = spec["label"]
    cfg["preset_meta"]["identified_template_source"] = spec["source"]
    return spec


def bhambare_user_inverse_template() -> dict[str, Any]:
    """Benchmark reproducible del XLSX de identificación Bhambare entregado el 14/09/2026."""
    spec = identified_parameter_template("bhambare")
    parameter_table = pd.DataFrame(
        [
            {
                "ID": "eta_opt_eff",
                "Parametro": "η óptica efectiva (ρ·γ·τ·α…)",
                "Nominal": 0.6520562499999999,
                "Identificado": float(spec["eta_opt_eff"]),
                "Cerca_del_limite": True,
            },
            {
                "ID": "eps_abs",
                "Parametro": "Emisividad del absorbedor εabs",
                "Nominal": 0.95,
                "Identificado": float(spec["eps_abs"]),
                "Cerca_del_limite": True,
            },
            {
                "ID": "eps_glass",
                "Parametro": "Emisividad del vidrio εvid",
                "Nominal": 0.88,
                "Identificado": float(spec["eps_glass"]),
                "Cerca_del_limite": True,
            },
        ]
    )
    comparison_table = pd.DataFrame(
        [
            {
                "Magnitud": key,
                "Modelo_inicial_XLSX": float(BHAMBARE_USER_INVERSE_2026_09_14["initial"][key]),
                "Modelo_identificado_XLSX": float(BHAMBARE_USER_INVERSE_2026_09_14["identified"][key]),
            }
            for key in ("Tout_C", "Tabs_K", "Tvid_K", "Qloss_W")
        ]
    )
    return {
        "parameter_table": parameter_table,
        "comparison_table": comparison_table,
        "note": (
            "Template incorporado desde ptc_modelo_inverso_bhambare.xlsx. "
            "Los tres parámetros identificados quedaron muy próximos a sus límites superiores; "
            "por eso deben interpretarse como una calibración diagnóstica y no como propiedades universales."
        ),
    }


def _bhambare_target_values(target_key: str) -> tuple[str, dict[str, float]]:
    """Devuelve el vector objetivo para la validación multivariable Bhambare."""
    key = str(target_key).strip().lower()
    cfg, _ = build_bhambare_sukhatme_preset()
    ref = cfg["preset_meta"]["reference"]
    targets: dict[str, tuple[str, dict[str, float]]] = {
        "sukhatme": (
            "Sukhatme & Nayak · referencia",
            {
                "Tout_C": float(ref["Tout_book_C"]),
                "Tabs_K": float(ref["Tabs_book_K"]),
                "Tvid_K": float(ref["Tglass_book_K"]),
                "Qloss_W": float(ref["Qloss_book_W"]),
            },
        ),
        "bhambare": (
            "Bhambare · modelo publicado",
            {
                "Tout_C": float(ref["Tout_article_C"]),
                "Tabs_K": float(ref["Tabs_article_K"]),
                "Tvid_K": float(ref["Tglass_article_K"]),
                "Qloss_W": float(ref["Qloss_article_W"]),
            },
        ),
        "xlsx_initial": (
            "XLSX 14/09 · modelo inicial",
            deepcopy(BHAMBARE_USER_INVERSE_2026_09_14["initial"]),
        ),
        "xlsx_identified": (
            "XLSX 14/09 · modelo identificado",
            deepcopy(BHAMBARE_USER_INVERSE_2026_09_14["identified"]),
        ),
    }
    if key not in targets:
        raise ValueError(f"Objetivo Bhambare desconocido: {target_key}")
    return targets[key]


def validate_bhambare_mode(
    fluid_database: Mapping[str, Mapping[str, Any]],
    *,
    target_key: str = "sukhatme",
    parameter_template: str = "nominal",
) -> dict[str, Any]:
    """Valida Bhambare con métricas multivariables comparables a la UI de Rea.

    Como las magnitudes tienen unidades distintas, MAE/RMSE globales se calculan
    sobre residuos relativos normalizados. La tabla conserva además el error
    absoluto en las unidades naturales de cada variable.
    """
    cfg, _ = build_bhambare_sukhatme_preset()
    # El XLSX del modelo inverso fue generado con BDF; usar el mismo integrador
    # hace que esos targets sirvan como test de regresión reproducible.
    if str(target_key).strip().lower().startswith("xlsx_"):
        cfg["solver"]["method"] = "BDF"
    parameter_mode = str(parameter_template).strip().lower()
    applied_template = None
    if parameter_mode == "identified":
        applied_template = apply_identified_parameter_template(cfg, "bhambare")
    elif parameter_mode != "nominal":
        raise ValueError(f"Template de parámetros Bhambare desconocido: {parameter_template}")

    result = PTCSimulator(cfg, fluid_database).simulate()
    k = len(result.t_s) - 1
    sim = {
        "Tout_C": float(result.Tout_C[k]),
        "Tabs_K": float(result.Tabs_mean_C[k] + 273.15),
        "Tvid_K": float(result.Tglass_mean_C[k] + 273.15),
        "Qloss_W": float(result.scalar_diag["Qloss_W"][k]),
    }
    target_label, target = _bhambare_target_values(target_key)

    labels = {
        "Tout_C": "T salida",
        "Tabs_K": "T absorbedor",
        "Tvid_K": "T vidrio",
        "Qloss_W": "Q pérdidas",
    }
    units = {"Tout_C": "°C", "Tabs_K": "K", "Tvid_K": "K", "Qloss_W": "W"}
    rows: list[dict[str, Any]] = []
    signed_relative: list[float] = []
    for key in ("Tout_C", "Tabs_K", "Tvid_K", "Qloss_W"):
        ref_value = float(target[key])
        sim_value = float(sim[key])
        denom = max(abs(ref_value), np.finfo(float).eps)
        rel = (sim_value - ref_value) / denom
        signed_relative.append(rel)
        rows.append(
            {
                "Magnitud": labels[key],
                "Unidad": units[key],
                "Objetivo": ref_value,
                "Modelo_Python": sim_value,
                "Diferencia": sim_value - ref_value,
                "Error_abs": abs(sim_value - ref_value),
                "Error_rel_pct": 100.0 * abs(rel),
                "Bias_rel_pct": 100.0 * rel,
            }
        )
    residuals = np.asarray(signed_relative, dtype=float)
    abs_pct = 100.0 * np.abs(residuals)
    metrics = {
        "MAPE_multivariable_pct": float(np.mean(abs_pct)),
        "RMSRE_pct": float(100.0 * np.sqrt(np.mean(residuals**2))),
        "Bias_rel_medio_pct": float(100.0 * np.mean(residuals)),
        "Error_max_pct": float(np.max(abs_pct)),
    }

    near_bounds = []
    if applied_template is not None:
        near_bounds = list(applied_template.get("near_bounds", []))
    note = (
        "Las cuatro salidas tienen unidades diferentes, por lo que las métricas globales se calculan sobre errores relativos normalizados. "
        "La tabla muestra también el error absoluto de cada magnitud en su unidad natural."
    )
    if applied_template is not None:
        note += (
            " Se aplicó el template identificado del XLSX del 14/09/2026: "
            f"ηopt,ef={applied_template['eta_opt_eff']:.10f}, "
            f"εabs={applied_template['eps_abs']:.10f}, "
            f"εvid={applied_template['eps_glass']:.10f}."
        )
        if near_bounds:
            note += " Los parámetros identificados están próximos al límite superior del espacio de búsqueda."

    return {
        "target_key": str(target_key),
        "target_label": target_label,
        "parameter_template": parameter_mode,
        "applied_parameters": applied_template,
        "table": pd.DataFrame(rows),
        "result": result,
        "config": cfg,
        "metrics": metrics,
        "note": note,
    }


def prototype_user_export_template() -> dict[str, Any]:
    """Curva benchmark exacta del CSV entregado por el usuario el 14/09/2026."""
    hours = np.asarray(REA_PROTOTYPE_USER_EXPORT_2026_09_14["hours"], dtype=int)
    eta_ref = np.asarray(REA_PROTOTYPE_USER_EXPORT_2026_09_14["eta_reference_pct"], dtype=float)
    eta_initial = np.asarray(REA_PROTOTYPE_USER_EXPORT_2026_09_14["eta_model_initial_pct"], dtype=float)
    eta_identified = np.asarray(REA_PROTOTYPE_USER_EXPORT_2026_09_14["eta_model_identified_pct"], dtype=float)
    table = pd.DataFrame(
        {
            "Hora": [f"{int(h):02d}:00" for h in hours],
            "Eta_referencia_pct": eta_ref,
            "Eta_modelo_inicial_CSV_pct": eta_initial,
            "Eta_modelo_identificado_CSV_pct": eta_identified,
        }
    )
    return {
        "table": table,
        "note": (
            "Template de salida incorporado desde 2026-09-14T12-26_export.csv. "
            "Los parámetros que generaron la curva identificada fueron recuperados del XLSX del modelo inverso: "
            "ηopt,ef=0.5583394106 y viento=6.326580297 m/s. El modo identificado puede ahora reproducirse directamente."
        ),
    }


def _prototype_target(target_key: str) -> tuple[str, np.ndarray]:
    key = str(target_key).strip().lower()
    if key == "experimental":
        return "Tabela 8 · experimental", np.asarray(REA_PROTOTYPE_HOURS["eta_exp_pct"], dtype=float)
    if key == "trnsys":
        return "Tabela 8 · TRNSYS", np.asarray(REA_PROTOTYPE_HOURS["eta_trnsys_pct"], dtype=float)
    if key == "csv_initial":
        return "CSV 14/09 · modelo inicial", np.asarray(REA_PROTOTYPE_USER_EXPORT_2026_09_14["eta_model_initial_pct"], dtype=float)
    if key == "csv_identified":
        return "CSV 14/09 · modelo identificado", np.asarray(REA_PROTOTYPE_USER_EXPORT_2026_09_14["eta_model_identified_pct"], dtype=float)
    raise ValueError(f"Template de referencia del prototipo desconocido: {target_key}")


def validate_rea_prototype_mode(
    validation_mode: str,
    fluid_database: Mapping[str, Mapping[str, Any]],
    *,
    target_key: str = "experimental",
    dni_source: str = "nominal",
    parameter_template: str = "auto",
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Ejecuta uno de los dos modos del prototipo y lo compara con una curva objetivo.

    - ``trnsys_published``: DNI=905 W/m², θ=0 e IAM=1, conforme a la
      idealización declarada en la conclusión del TCC.
    - ``physical_fixed_ns``: colector fijo, incidencia/IAM horario explícito.
      El DNI puede ser nominal o de cielo claro; no se presenta como dato medido.
    """
    if config is None:
        cfg, _ = build_rea_prototype_preset(validation_mode, dni_source=dni_source)
    else:
        cfg = deepcopy(config)

    template_key = str(parameter_template).strip().lower()
    if template_key not in {"auto", "nominal", "identified"}:
        raise ValueError(f"Template de parámetros desconocido: {parameter_template}")
    use_identified = template_key == "identified" or (template_key == "auto" and str(target_key).lower() == "csv_identified")
    applied_template = None
    if use_identified:
        applied_template = apply_identified_parameter_template(cfg, "rea_prototype")

    result = PTCSimulator(cfg, fluid_database).simulate()
    hours = np.asarray(REA_PROTOTYPE_HOURS["hours"], dtype=float)
    eta_exp = np.asarray(REA_PROTOTYPE_HOURS["eta_exp_pct"], dtype=float)
    eta_trnsys = np.asarray(REA_PROTOTYPE_HOURS["eta_trnsys_pct"], dtype=float)
    target_label, eta_target = _prototype_target(target_key)
    # Para Rea/Fiamonzini se usa la misma base de la Ec. (10):
    # eta = m Cp (Tout-Tin) / (Aa * Ib), con Ib=DNI. En el modo fijo físico
    # esto NO divide por cos(theta), por lo que las pérdidas geométricas del
    # colector sin tracking aparecen realmente como caída de eficiencia.
    eta_key = "eta_dni_basis_pct"
    eta_python = np.asarray(
        [float(result.scalar_diag[eta_key][_nearest_index(result.LAT_h, hour)]) for hour in hours],
        dtype=float,
    )
    eta_projected = np.asarray(
        [float(result.scalar_diag["eta_pct"][_nearest_index(result.LAT_h, hour)]) for hour in hours],
        dtype=float,
    )
    dni_python = np.asarray(
        [float(result.scalar_diag["DNI_W_m2"][_nearest_index(result.LAT_h, hour)]) for hour in hours],
        dtype=float,
    )
    theta_python = np.asarray(
        [float(result.scalar_diag["theta_deg"][_nearest_index(result.LAT_h, hour)]) for hour in hours],
        dtype=float,
    )
    iam_python = np.asarray(
        [float(result.scalar_diag["IAM"][_nearest_index(result.LAT_h, hour)]) for hour in hours],
        dtype=float,
    )
    endloss_python = np.asarray(
        [float(result.scalar_diag["endLoss"][_nearest_index(result.LAT_h, hour)]) for hour in hours],
        dtype=float,
    )
    g_aperture_python = np.asarray(
        [float(result.scalar_diag["G_aperture_W_m2"][_nearest_index(result.LAT_h, hour)]) for hour in hours],
        dtype=float,
    )
    solar_time_python = np.asarray(
        [float(result.scalar_diag["solar_time_h"][_nearest_index(result.LAT_h, hour)]) for hour in hours],
        dtype=float,
    )
    delta = eta_python - eta_target
    mae = float(np.mean(np.abs(delta)))
    rmse = float(np.sqrt(np.mean(np.square(delta))))
    mape = float(100.0 * np.mean(np.abs(delta) / np.maximum(np.abs(eta_target), np.finfo(float).eps)))
    table = pd.DataFrame(
        {
            "Hora": [f"{int(h):02d}:00" for h in hours],
            "Eta_experimental_pct": eta_exp,
            "Eta_TRNSYS_pct": eta_trnsys,
            "Eta_objetivo_pct": eta_target,
            "Eta_Python_Ec10_pct": eta_python,
            "Eta_Python_sobre_proyectada_pct": eta_projected,
            "Diferencia_objetivo_pp": delta,
            "DNI_Python_W_m2": dni_python,
            "G_apertura_Python_W_m2": g_aperture_python,
            "Hora_solar_aparente_h": solar_time_python,
            "theta_Python_deg": theta_python,
            "IAM_Python": iam_python,
            "EndLoss_Python": endloss_python,
        }
    )
    # preset_meta es opcional en configuraciones antiguas/de sesión; si existe
    # explícitamente como None, ``dict.get(..., {})`` devolvería None.
    meta = cfg.get("preset_meta") or {}
    mode_label = str(meta.get("prototype_validation_mode_label", validation_mode))
    if str(validation_mode).lower() == "trnsys_published":
        note = (
            "Reproducción de la idealización declarada por Rea Quille: DNI nominal 905 W/m², IAM=1 y "
            "sin tracking. Tin horario no está publicado, por lo que Tin=25 °C sigue siendo una hipótesis."
        )
    else:
        note = (
            "Exploración física sin tracking: el colector permanece fijo y se calcula la geometría solar horaria. "
            "La eficiencia comparada usa la Ec. (10), eta=mCpΔT/(Aa·DNI), por lo que cos(theta), IAM y EndLoss "
            "reducen realmente la eficiencia fuera del mediodía. El TCC no publica una serie DNI medida; la fuente "
            "DNI elegida aquí es una hipótesis/modelo explícito."
        )
    if applied_template is not None:
        note += (
            f" Parámetros identificados aplicados: ηopt,ef={applied_template['eta_opt_eff']:.10f}, "
            f"viento={applied_template['wind_m_s']:.6f} m/s ({applied_template['source']})."
        )
    else:
        note += " Parámetros físicos/ópticos nominales del preset."

    return {
        "kind": "prototype_mode",
        "mode": str(validation_mode),
        "mode_label": mode_label,
        "target_key": str(target_key),
        "target_label": target_label,
        "dni_source": str(dni_source),
        "parameter_template": "identified" if applied_template is not None else "nominal",
        "applied_parameters": applied_template,
        "table": table,
        "result": result,
        "metrics": {
            "Eta_target_mean_pct": float(np.mean(eta_target)),
            "Eta_python_mean_pct": float(np.mean(eta_python)),
            "MAE_pp": mae,
            "RMSE_pp": rmse,
            "MAPE_pct": mape,
        },
        "note": note,
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



def _bhambare_reference_bundle(
    fluid_database: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, float]]:
    """Configuración y referencia estricta usada por los diagnósticos de sensibilidad."""
    cfg, _ = build_bhambare_sukhatme_preset()
    ref_meta = cfg["preset_meta"]["reference"]
    properties = FluidPropertyEvaluator("ParathermNF", fluid_database)
    tout_ref = float(ref_meta["Tout_book_C"])
    tin_ref = float(ref_meta["Tin_C"])
    prop_mean = properties(0.5 * (tin_ref + tout_ref) + 273.15)
    qutil_ref = float(ref_meta["mdot_kg_s"]) * prop_mean.Cp * (tout_ref - tin_ref)
    eta_ref = 100.0 * qutil_ref / (
        float(ref_meta["beam_W_m2"])
        * float(cfg["geometry"]["W"])
        * float(cfg["geometry"]["L"])
    )
    reference = {
        "Tout_C": tout_ref,
        "Tabs_K": float(ref_meta["Tabs_book_K"]),
        "Tvid_K": float(ref_meta["Tglass_book_K"]),
        "Qloss_W": float(ref_meta["Qloss_book_W"]),
        "Qutil_W": qutil_ref,
        "eta_pct": eta_ref,
    }
    return cfg, reference


def _final_metrics(result: Any) -> dict[str, float]:
    k = len(result.t_s) - 1
    rates = np.concatenate(
        [
            np.asarray(result.node_diag["dTf_dt_K_s"][k], dtype=float),
            np.asarray(result.node_diag["dTabs_dt_K_s"][k], dtype=float),
            np.asarray(result.node_diag["dTglass_dt_K_s"][k], dtype=float),
        ]
    )
    return {
        "Tout_C": float(result.Tout_C[k]),
        "Tabs_K": float(result.Tabs_mean_C[k] + 273.15),
        "Tvid_K": float(result.Tglass_mean_C[k] + 273.15),
        "Qloss_W": float(result.scalar_diag["Qloss_W"][k]),
        "Qutil_W": float(result.scalar_diag["Quseful_W"][k]),
        "eta_pct": float(result.scalar_diag["eta_pct"][k]),
        "max_abs_dTdt_K_s": float(np.nanmax(np.abs(rates))),
        "nfev": float(result.nfev),
    }


def _reference_error_score(metrics: Mapping[str, float], reference: Mapping[str, float]) -> float:
    """RMS de errores relativos de las cuatro magnitudes directamente tabuladas por Sukhatme."""
    keys = ("Tout_C", "Tabs_K", "Tvid_K", "Qloss_W")
    errors = [
        (float(metrics[key]) - float(reference[key]))
        / max(abs(float(reference[key])), np.finfo(float).eps)
        for key in keys
    ]
    return 100.0 * float(np.sqrt(np.mean(np.square(errors))))


def _run_case_fast(
    cfg: Mapping[str, Any],
    fluid_database: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, float], float]:
    started = perf_counter()
    result = PTCSimulator(cfg, fluid_database).simulate()
    elapsed = perf_counter() - started
    metrics = _final_metrics(result)
    metrics["cpu_s"] = float(elapsed)
    return metrics, elapsed


def analyze_bhambare_numerical_convergence(
    fluid_database: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Diagnóstico de independencia de malla, paso, tolerancias y duración.

    El análisis mantiene la física del caso Bhambare/Sukhatme fija. Se usa BDF
    para que la prueba espacial/temporal no quede dominada por el costo de RK45;
    la sensibilidad al integrador ya se evalúa por separado.
    """
    base_cfg, reference = _bhambare_reference_bundle(fluid_database)
    base_cfg = deepcopy(base_cfg)
    base_cfg["solver"]["method"] = "BDF"
    base_cfg["operation"]["output_step_s"] = 600.0

    def execute(group: str, value: float | int, cfg: dict[str, Any]) -> dict[str, Any]:
        metrics, _ = _run_case_fast(cfg, fluid_database)
        return {
            "Grupo": group,
            "Valor": value,
            **metrics,
            "Score_vs_Sukhatme_pct": _reference_error_score(metrics, reference),
        }

    rows: list[dict[str, Any]] = []

    # 1) Independencia espacial. Un paso temporal moderado acelera la prueba sin
    # alterar la solución final; la sensibilidad a max_step se prueba después.
    for nseg in (1, 2, 4, 6, 8, 12, 16, 24, 32, 48):
        cfg = deepcopy(base_cfg)
        cfg["geometry"]["Nseg"] = int(nseg)
        cfg["solver"]["max_step_s"] = 120.0
        rows.append(execute("Nodos", nseg, cfg))

    # 2) Sensibilidad al paso máximo del integrador con N=12.
    for max_step in (600.0, 300.0, 120.0, 60.0, 20.0, 10.0):
        cfg = deepcopy(base_cfg)
        cfg["geometry"]["Nseg"] = 12
        cfg["solver"]["max_step_s"] = float(max_step)
        rows.append(execute("max_step_s", max_step, cfg))

    # 3) Sensibilidad a tolerancias. atol se mantiene una década por debajo de rtol.
    for rtol in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        cfg = deepcopy(base_cfg)
        cfg["geometry"]["Nseg"] = 12
        cfg["solver"]["max_step_s"] = 120.0
        cfg["solver"]["rtol"] = float(rtol)
        cfg["solver"]["atol"] = float(rtol) / 10.0
        rows.append(execute("rtol", rtol, cfg))

    # 4) Tiempo de calentamiento desde las condiciones iniciales hasta 12:30 LAT.
    end_s = float(base_cfg["operation"]["t_end_s"])
    for hours in (0.5, 1.0, 2.0, 4.0, 8.0):
        cfg = deepcopy(base_cfg)
        cfg["geometry"]["Nseg"] = 12
        cfg["solver"]["max_step_s"] = 120.0
        cfg["operation"]["t_start_s"] = end_s - float(hours) * 3600.0
        rows.append(execute("Duracion_h", hours, cfg))

    table = pd.DataFrame(rows)

    # Para la malla, la solución N=48 se toma como referencia interna de convergencia.
    mesh = table.loc[table["Grupo"] == "Nodos"].copy()
    mesh_ref = mesh.loc[mesh["Valor"].astype(float).idxmax()]
    for key in ("Tout_C", "Tabs_K", "Tvid_K", "Qloss_W", "eta_pct"):
        denom = max(abs(float(mesh_ref[key])), np.finfo(float).eps)
        mesh[f"Delta_{key}_vs_N48_pct"] = 100.0 * np.abs(mesh[key].astype(float) - float(mesh_ref[key])) / denom
    mesh["Max_delta_vs_N48_pct"] = mesh[
        [f"Delta_{key}_vs_N48_pct" for key in ("Tout_C", "Tabs_K", "Tvid_K", "Qloss_W", "eta_pct")]
    ].max(axis=1)

    # Primer N que queda por debajo de 0.5 % y 0.1 % en todas las magnitudes.
    def first_n(tol: float) -> int | None:
        ok = mesh.loc[mesh["Max_delta_vs_N48_pct"] <= tol]
        return int(ok.iloc[0]["Valor"]) if not ok.empty else None

    summary = {
        "N_0p5pct": first_n(0.5),
        "N_0p1pct": first_n(0.1),
        "mesh_span_Qloss_W": float(mesh["Qloss_W"].max() - mesh["Qloss_W"].min()),
        "mesh_span_Tout_C": float(mesh["Tout_C"].max() - mesh["Tout_C"].min()),
        "reference": reference,
    }
    return {
        "table": table,
        "mesh_table": mesh,
        "summary": summary,
        "reference": reference,
        "note": (
            "Esta prueba separa convergencia numérica de discrepancia física. Si N, max_step, rtol y duración alcanzan una meseta mientras el error frente a Sukhatme permanece, el problema no es de discretización/integración sino de hipótesis, propiedades o submodelos físicos."
        ),
    }


def analyze_bhambare_physical_sensitivity(
    fluid_database: Mapping[str, Mapping[str, Any]],
    perturbation: float = 0.10,
) -> dict[str, Any]:
    """Sensibilidad local OAT de los principales parámetros físicos del caso.

    Cada parámetro se perturba ±perturbation manteniendo todos los demás fijos.
    Se reporta la elasticidad normalizada y cuánto mejora/empeora el ajuste a las
    cuatro magnitudes tabuladas por Sukhatme.
    """
    p = float(perturbation)
    if not (0.0 < p < 0.5):
        raise ValueError("perturbation debe estar entre 0 y 0.5")

    base_cfg, reference = _bhambare_reference_bundle(fluid_database)
    base_cfg = deepcopy(base_cfg)
    base_cfg["solver"]["method"] = "BDF"
    base_cfg["solver"]["max_step_s"] = 300.0
    base_cfg["operation"]["output_step_s"] = 600.0
    base_metrics, _ = _run_case_fast(base_cfg, fluid_database)
    base_score = _reference_error_score(base_metrics, reference)

    # label, category, getter(cfg,db), setter(cfg,db,value), bounds
    specs: list[tuple[str, str, Any, Any, tuple[float | None, float | None]]] = []

    def cfg_spec(label: str, category: str, path: tuple[str, ...], bounds=(None, None)) -> None:
        def getter(cfg, db):
            obj = cfg
            for key in path:
                obj = obj[key]
            return float(obj)
        def setter(cfg, db, value):
            obj = cfg
            for key in path[:-1]:
                obj = obj[key]
            obj[path[-1]] = float(value)
        specs.append((label, category, getter, setter, bounds))

    cfg_spec("Reflectividad del espejo", "Óptica", ("optics", "reflectivity"), (0.0, 1.0))
    cfg_spec("Factor de interceptación", "Óptica", ("optics", "intercept_factor"), (0.0, 1.0))
    cfg_spec("Transmitancia del vidrio", "Óptica", ("materials", "glass", "tau"), (0.0, 1.0))
    cfg_spec("Absortancia del absorbedor", "Óptica", ("materials", "absorber", "alpha"), (0.0, 1.0))
    cfg_spec("Emisividad del absorbedor", "Pérdidas", ("materials", "absorber", "eps"), (0.01, 1.0))
    cfg_spec("Emisividad del vidrio", "Pérdidas", ("materials", "glass", "eps"), (0.01, 1.0))
    cfg_spec("Velocidad del viento", "Ambiente", ("environment", "wind_m_s"), (0.0, None))
    cfg_spec("Delta T cielo", "Ambiente", ("environment", "sky_delta_K"), (0.0, None))
    cfg_spec("Caudal másico", "Entrada publicada", ("operation", "mdot"), (1e-8, None))
    cfg_spec("DNI", "Entrada publicada", ("solar", "DNI_constant_W_m2"), (0.0, None))

    # Multiplicadores de propiedades del fluido: son especialmente importantes
    # porque Bhambare no publica tablas completas de rho, mu y k de Paratherm NF.
    def fluid_mult_spec(label: str, key: str) -> None:
        def getter(cfg, db):
            return float(db["ParathermNF"]["multipliers"][key])
        def setter(cfg, db, value):
            db["ParathermNF"]["multipliers"][key] = float(value)
        specs.append((label, "Propiedades HTF", getter, setter, (0.05, None)))

    fluid_mult_spec("Multiplicador Cp HTF", "Cp")
    fluid_mult_spec("Multiplicador μ HTF", "mu")
    fluid_mult_spec("Multiplicador k HTF", "k")
    fluid_mult_spec("Multiplicador ρ HTF", "rho")

    rows: list[dict[str, Any]] = []
    for label, category, getter, setter, bounds in specs:
        nominal = float(getter(base_cfg, fluid_database))
        values: dict[str, dict[str, float]] = {}
        for direction, factor in (("-", 1.0 - p), ("+", 1.0 + p)):
            cfg = deepcopy(base_cfg)
            db = deepcopy(fluid_database)
            value = nominal * factor
            lo, hi = bounds
            if lo is not None:
                value = max(float(lo), value)
            if hi is not None:
                value = min(float(hi), value)
            setter(cfg, db, value)
            metrics, _ = _run_case_fast(cfg, db)
            values[direction] = {**metrics, "parameter_value": float(value)}

        minus = values["-"]
        plus = values["+"]
        score_minus = _reference_error_score(minus, reference)
        score_plus = _reference_error_score(plus, reference)
        best_score = min(score_minus, score_plus)
        best_direction = "-" if score_minus <= score_plus else "+"

        # Elasticidades centrales: (dy/y)/(dp/p). Para los parámetros limitados
        # por 1.0 se usa el delta real aplicado.
        dp_rel = (plus["parameter_value"] - minus["parameter_value"]) / max(abs(nominal), np.finfo(float).eps)
        row: dict[str, Any] = {
            "Parametro": label,
            "Categoria": category,
            "Nominal": nominal,
            "Valor_menos": minus["parameter_value"],
            "Valor_mas": plus["parameter_value"],
            "Score_base_pct": base_score,
            "Score_menos_pct": score_minus,
            "Score_mas_pct": score_plus,
            "Mejor_direccion": best_direction,
            "Mejor_score_pct": best_score,
            "Mejora_score_pp": base_score - best_score,
        }
        for key in ("Tout_C", "Tabs_K", "Tvid_K", "Qloss_W", "Qutil_W", "eta_pct"):
            y0 = float(base_metrics[key])
            dy_rel = (float(plus[key]) - float(minus[key])) / max(abs(y0), np.finfo(float).eps)
            row[f"S_{key}"] = dy_rel / dp_rel if abs(dp_rel) > np.finfo(float).eps else float("nan")
            row[f"Delta_{key}_menos"] = float(minus[key]) - y0
            row[f"Delta_{key}_mas"] = float(plus[key]) - y0
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("Mejora_score_pp", ascending=False).reset_index(drop=True)
    return {
        "table": table,
        "baseline": base_metrics,
        "reference": reference,
        "baseline_score_pct": base_score,
        "perturbation_pct": 100.0 * p,
        "note": (
            "Sensibilidad OAT local: cada parámetro se cambia ±10 % manteniendo el resto fijo. La columna Mejora_score_pp indica cuánto reduce el RMS de error relativo frente a Tout, Tabs, Tvid y Qloss de Sukhatme. Las entradas publicadas (DNI, mdot) se muestran como control de sensibilidad, no como parámetros que deban ajustarse arbitrariamente."
        ),
    }

# -----------------------------------------------------------------------------
# Identificación paramétrica multiparámetro / modelo inverso
# -----------------------------------------------------------------------------


def _effective_optical_efficiency(cfg: Mapping[str, Any]) -> float:
    optics = cfg["optics"]
    absorber = cfg["materials"]["absorber"]
    value = (
        float(optics["reflectivity"])
        * float(optics["intercept_factor"])
        * float(absorber["alpha"])
        * float(optics.get("dirt_factor", 1.0))
        * float(optics.get("shade_factor", 1.0))
    )
    if bool(cfg["model"].get("has_glass", False)):
        value *= float(cfg["materials"]["glass"]["tau"])
    return float(value)


def _set_effective_optical_efficiency(cfg: dict[str, Any], target: float) -> None:
    """Ajusta rho para imponer eta_opt,ef manteniendo gamma/tau/alpha fijos.

    Esto evita intentar identificar simultáneamente factores ópticos fuertemente
    correlacionados. El parámetro identificado es el producto efectivo; la
    reflectividad equivalente solo es el mecanismo usado por el modelo para
    imponer ese producto.
    """
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


def _inverse_parameter_registry(
    representative_cfg: Mapping[str, Any],
    fluid_database: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Parámetros candidatos con límites físicos explícitos."""
    cfg = deepcopy(representative_cfg)
    eta_nom = _effective_optical_efficiency(cfg)
    optics = cfg["optics"]
    absorber = cfg["materials"]["absorber"]
    optical_rest = (
        float(optics["intercept_factor"])
        * float(absorber["alpha"])
        * float(optics.get("dirt_factor", 1.0))
        * float(optics.get("shade_factor", 1.0))
    )
    if bool(cfg["model"].get("has_glass", False)):
        optical_rest *= float(cfg["materials"]["glass"]["tau"])
    eta_upper = min(0.999999 * optical_rest, 0.999999)
    eta_lower = max(0.05, min(eta_nom * 0.45, eta_upper * 0.75))

    fluid_name = str(cfg["operation"]["fluid"])
    db_entry = fluid_database.get(fluid_name, {})
    multipliers = db_entry.get("multipliers", {}) if isinstance(db_entry, Mapping) else {}

    registry: dict[str, dict[str, Any]] = {
        "eta_opt_eff": {
            "label": "η óptica efectiva (ρ·γ·τ·α…)",
            "category": "Óptica",
            "nominal": eta_nom,
            "bounds": (eta_lower, eta_upper),
            "status": "Producto efectivo; evita separar factores ópticos no identificables individualmente.",
        },
        "eps_abs": {
            "label": "Emisividad del absorbedor εabs",
            "category": "Radiación",
            "nominal": float(cfg["materials"]["absorber"]["eps"]),
            "bounds": (0.05, 0.99),
            "status": "Ajustable solo si su valor documental es incierto o se estudia sensibilidad.",
        },
        "wind_m_s": {
            "label": "Velocidad de viento",
            "category": "Ambiente",
            "nominal": float(cfg["environment"]["wind_m_s"]),
            "bounds": (0.0, 12.0),
            "status": "En Rea Quille no está tabulada en las tablas mensuales; en Bhambare está publicada.",
        },
        "support_loss_fraction": {
            "label": "Fracción de pérdida por soportes",
            "category": "Pérdidas",
            "nominal": float(cfg["model"].get("support_loss_fraction", 0.015))
            if bool(cfg["model"].get("include_supports", False))
            else 0.0,
            "bounds": (0.0, 0.08),
            "status": "Parámetro diagnóstico de pérdida adicional; no debe usarse como factor de ajuste arbitrario.",
        },
    }
    if bool(cfg["model"].get("has_glass", False)):
        registry["eps_glass"] = {
            "label": "Emisividad del vidrio εvid",
            "category": "Radiación",
            "nominal": float(cfg["materials"]["glass"]["eps"]),
            "bounds": (0.50, 0.99),
            "status": "Cubierta de vidrio; relevante al intercambio absorbedor–vidrio y a pérdidas exteriores.",
        }
    if str(cfg["environment"].get("sky_model", "")) == "rea_quille":
        registry["dew_point_C"] = {
            "label": "Punto de rocío Tdp",
            "category": "Cielo",
            "nominal": float(cfg["environment"].get("dew_point_C", 15.0)),
            "bounds": (-5.0, 30.0),
            "status": "No publicado en las tablas mensuales de Rea Quille; hipótesis del preset.",
        }
    else:
        registry["sky_delta_K"] = {
            "label": "ΔT cielo = Tamb − Tsky",
            "category": "Cielo",
            "nominal": float(cfg["environment"].get("sky_delta_K", 6.0)),
            "bounds": (0.0, 25.0),
            "status": "Parámetro del modelo simplificado de cielo.",
        }

    for key, label, bounds in (
        ("Cp", "Multiplicador Cp del HTF", (0.80, 1.20)),
        ("mu", "Multiplicador μ del HTF", (0.60, 1.40)),
        ("k", "Multiplicador k del HTF", (0.75, 1.25)),
    ):
        if key in multipliers:
            registry[f"fluid_{key}"] = {
                "label": label,
                "category": "Propiedades HTF",
                "nominal": float(multipliers[key]),
                "bounds": bounds,
                "status": "Útil si las propiedades del fluido no están completamente documentadas.",
            }
    return registry


def inverse_parameter_options(
    case: str,
    fluid_database: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Expone al UI los parámetros disponibles para cada familia de validación."""
    key = str(case).strip().lower()
    if key == "bhambare":
        cfg, _ = build_bhambare_sukhatme_preset()
    elif key == "rea_foz":
        cfg, _ = build_rea_monthly_preset("Foz do Iguaçu", 1)
    elif key == "rea_alvorada":
        cfg, _ = build_rea_monthly_preset("Alvorada do Norte", 1)
    elif key == "rea_prototype":
        cfg, _ = build_rea_prototype_preset()
    else:
        raise ValueError(f"Caso inverso desconocido: {case}")
    return _inverse_parameter_registry(cfg, fluid_database)


def _apply_inverse_parameters(
    cfg: dict[str, Any],
    db: dict[str, Any],
    parameter_ids: Sequence[str],
    values: Sequence[float],
) -> None:
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
            raise ValueError(f"Parámetro inverso no soportado: {pid}")


def _prepare_inverse_cfg(cfg: dict[str, Any], *, fast: bool = False) -> None:
    """Configura la integración del inverso.

    Durante la búsqueda se usa N=6 y max_step=600 s; la propia sensibilidad
    espacial mostró error <0.5 % a N=6. La revalidación final vuelve a la malla
    documental del preset (normalmente N=12).
    """
    cfg["solver"]["method"] = "BDF"
    cfg["solver"]["rtol"] = min(float(cfg["solver"].get("rtol", 1e-6)), 1e-6)
    cfg["solver"]["atol"] = min(float(cfg["solver"].get("atol", 1e-7)), 1e-7)
    if fast:
        cfg["geometry"]["Nseg"] = min(int(cfg["geometry"].get("Nseg", 12)), 6)
        cfg["solver"]["max_step_s"] = 600.0
        cfg["operation"]["output_step_s"] = 600.0
    else:
        cfg["solver"]["max_step_s"] = max(float(cfg["solver"].get("max_step_s", 20.0)), 120.0)
        cfg["operation"]["output_step_s"] = max(float(cfg["operation"].get("output_step_s", 60.0)), 300.0)


def _bhambare_inverse_evaluation(
    x: Sequence[float],
    parameter_ids: Sequence[str],
    fluid_database: Mapping[str, Mapping[str, Any]],
    *, fast: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    cfg, reference = _bhambare_reference_bundle(fluid_database)
    cfg = deepcopy(cfg)
    db = deepcopy(fluid_database)
    _prepare_inverse_cfg(cfg, fast=fast)
    _apply_inverse_parameters(cfg, db, parameter_ids, x)
    result = PTCSimulator(cfg, db).simulate()
    metrics = _final_metrics(result)
    residuals = np.asarray(
        [
            (metrics["Tout_C"] - reference["Tout_C"]) / max(abs(reference["Tout_C"]), 1.0),
            (metrics["Tabs_K"] - reference["Tabs_K"]) / max(abs(reference["Tabs_K"]), 1.0),
            (metrics["Tvid_K"] - reference["Tvid_K"]) / max(abs(reference["Tvid_K"]), 1.0),
            (metrics["Qloss_W"] - reference["Qloss_W"]) / max(abs(reference["Qloss_W"]), 1.0),
        ],
        dtype=float,
    )
    rows = [
        {
            "Conjunto": "calibración",
            "Caso": "Bhambare / Sukhatme",
            "Magnitud": name,
            "Referencia": float(reference[key]),
            "Modelo": float(metrics[key]),
            "Error_rel_pct": _relative_error(metrics[key], reference[key]),
            "Residual_obj": (float(metrics[key]) - float(reference[key])) / max(abs(float(reference[key])), 1.0),
        }
        for name, key in (
            ("Tout_C", "Tout_C"),
            ("Tabs_K", "Tabs_K"),
            ("Tvid_K", "Tvid_K"),
            ("Qloss_W", "Qloss_W"),
        )
    ]
    return residuals, rows


def _rea_month_indices(strategy: str) -> tuple[list[int], list[int]]:
    mode = str(strategy).strip().lower()
    if mode == "all":
        return list(range(12)), []
    if mode == "alternating":
        # Cuatro puntos estacionales para identificar; ocho meses quedan totalmente fuera del ajuste.
        train = [0, 3, 6, 9]  # Ene, Abr, Jul, Oct
        return train, [i for i in range(12) if i not in train]
    raise ValueError(f"Estrategia mensual desconocida: {strategy}")


def _rea_monthly_inverse_evaluation(
    city: str,
    x: Sequence[float],
    parameter_ids: Sequence[str],
    fluid_database: Mapping[str, Mapping[str, Any]],
    strategy: str,
    return_all: bool = False,
    fast: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    city_key = city.strip().lower()
    if city_key.startswith("foz"):
        data = REA_FOZ_MONTHLY
        city_name = "Foz do Iguaçu"
    else:
        data = REA_ALVORADA_MONTHLY
        city_name = "Alvorada do Norte"
    train_idx, validation_idx = _rea_month_indices(strategy)
    indices = list(range(12)) if return_all else train_idx
    residuals: list[float] = []
    rows: list[dict[str, Any]] = []
    for i in indices:
        cfg, _ = build_rea_monthly_preset(city_name, i + 1)
        db = deepcopy(fluid_database)
        _prepare_inverse_cfg(cfg, fast=fast)
        _apply_inverse_parameters(cfg, db, parameter_ids, x)
        result = PTCSimulator(cfg, db).simulate()
        k = len(result.t_s) - 1
        tout = float(result.Tout_C[k])
        eta = float(result.scalar_diag["eta_pct"][k])
        tout_ref = float(data["Tout_ref_C"][i])
        tin_ref = float(data["Tin_C"][i])
        eta_ref = float(data["eta_ref_pct"][i])
        dT_ref = tout_ref - tin_ref
        dT_sim = tout - tin_ref
        if not return_all or i in train_idx:
            residuals.extend(
                [
                    (dT_sim - dT_ref) / max(abs(dT_ref), 1.0),
                    (eta - eta_ref) / max(abs(eta_ref), 1.0),
                ]
            )
        rows.extend(
            [
                {
                    "Conjunto": "calibración" if i in train_idx else "validación",
                    "Caso": MONTH_ABBR_ES[i],
                    "Magnitud": "Tout_C",
                    "Referencia": tout_ref,
                    "Modelo": tout,
                    "Error_rel_pct": _relative_error(tout, tout_ref),
                    "Residual_obj": (dT_sim - dT_ref) / max(abs(dT_ref), 1.0),
                },
                {
                    "Conjunto": "calibración" if i in train_idx else "validación",
                    "Caso": MONTH_ABBR_ES[i],
                    "Magnitud": "Eta_pct",
                    "Referencia": eta_ref,
                    "Modelo": eta,
                    "Error_rel_pct": _relative_error(eta, eta_ref),
                    "Residual_obj": (eta - eta_ref) / max(abs(eta_ref), 1.0),
                },
            ]
        )
    return np.asarray(residuals, dtype=float), rows


def _rea_prototype_inverse_evaluation(
    x: Sequence[float],
    parameter_ids: Sequence[str],
    fluid_database: Mapping[str, Mapping[str, Any]],
    *, fast: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    cfg, _ = build_rea_prototype_preset()
    db = deepcopy(fluid_database)
    _prepare_inverse_cfg(cfg, fast=fast)
    _apply_inverse_parameters(cfg, db, parameter_ids, x)
    result = PTCSimulator(cfg, db).simulate()
    hours = np.asarray(REA_PROTOTYPE_HOURS["hours"], dtype=float)
    eta_ref = np.asarray(REA_PROTOTYPE_HOURS["eta_exp_pct"], dtype=float)
    eta_model = np.asarray(
        [float(result.scalar_diag["eta_pct"][_nearest_index(result.LAT_h, hour)]) for hour in hours],
        dtype=float,
    )
    residuals = (eta_model - eta_ref) / np.maximum(np.abs(eta_ref), 1.0)
    rows = [
        {
            "Conjunto": "calibración exploratoria",
            "Caso": f"{int(hour):02d}:00",
            "Magnitud": "Eta_pct",
            "Referencia": float(ref),
            "Modelo": float(sim),
            "Error_rel_pct": _relative_error(sim, ref),
            "Residual_obj": (float(sim) - float(ref)) / max(abs(float(ref)), 1.0),
        }
        for hour, ref, sim in zip(hours, eta_ref, eta_model)
    ]
    return residuals, rows


def _score_from_residuals(residuals: Sequence[float]) -> float:
    arr = np.asarray(residuals, dtype=float)
    return 100.0 * float(np.sqrt(np.mean(np.square(arr)))) if arr.size else float("nan")


def calibrate_inverse_model(
    case: str,
    fluid_database: Mapping[str, Mapping[str, Any]],
    parameter_ids: Sequence[str],
    *,
    monthly_strategy: str = "alternating",
    max_nfev: int = 28,
) -> dict[str, Any]:
    """Identificación multiparámetro acotada y revalidación posterior.

    ``case``: bhambare | rea_foz | rea_alvorada | rea_prototype.
    Para Rea mensual, ``alternating`` reserva seis meses como hold-out; ``all``
    usa los doce puntos para calibración y no constituye validación independiente.
    """
    case_key = str(case).strip().lower()
    if not parameter_ids:
        raise ValueError("Seleccione al menos un parámetro para identificar.")

    if case_key == "bhambare":
        representative_cfg, _ = build_bhambare_sukhatme_preset()
    elif case_key == "rea_foz":
        representative_cfg, _ = build_rea_monthly_preset("Foz do Iguaçu", 1)
    elif case_key == "rea_alvorada":
        representative_cfg, _ = build_rea_monthly_preset("Alvorada do Norte", 1)
    elif case_key == "rea_prototype":
        representative_cfg, _ = build_rea_prototype_preset()
    else:
        raise ValueError(f"Caso inverso desconocido: {case}")

    registry = _inverse_parameter_registry(representative_cfg, fluid_database)
    unknown = [pid for pid in parameter_ids if pid not in registry]
    if unknown:
        raise ValueError(f"Parámetros no disponibles para este caso: {unknown}")
    if case_key == "bhambare" and len(parameter_ids) > 4:
        raise ValueError("Bhambare aporta cuatro magnitudes independientes; use como máximo cuatro parámetros simultáneos.")

    x0 = np.asarray([float(registry[pid]["nominal"]) for pid in parameter_ids], dtype=float)
    lower = np.asarray([float(registry[pid]["bounds"][0]) for pid in parameter_ids], dtype=float)
    upper = np.asarray([float(registry[pid]["bounds"][1]) for pid in parameter_ids], dtype=float)
    x0 = np.minimum(np.maximum(x0, lower + 1e-10), upper - 1e-10)

    cache: dict[tuple[float, ...], tuple[np.ndarray, list[dict[str, Any]]]] = {}

    def evaluate(x: Sequence[float], *, full: bool = False, fast: bool = False):
        key = tuple(np.round(np.asarray(x, dtype=float), 10)) + ((1.0,) if full else (0.0,), (1.0,) if fast else (0.0,))
        if key in cache:
            return cache[key]
        if case_key == "bhambare":
            out = _bhambare_inverse_evaluation(x, parameter_ids, fluid_database, fast=fast)
        elif case_key == "rea_foz":
            out = _rea_monthly_inverse_evaluation(
                "Foz do Iguaçu", x, parameter_ids, fluid_database, monthly_strategy, return_all=full, fast=fast
            )
        elif case_key == "rea_alvorada":
            out = _rea_monthly_inverse_evaluation(
                "Alvorada do Norte", x, parameter_ids, fluid_database, monthly_strategy, return_all=full, fast=fast
            )
        else:
            out = _rea_prototype_inverse_evaluation(x, parameter_ids, fluid_database, fast=fast)
        cache[key] = out
        return out

    started = perf_counter()
    result = least_squares(
        lambda x: evaluate(x, full=False, fast=True)[0],
        x0=x0,
        bounds=(lower, upper),
        method="trf",
        jac="2-point",
        x_scale="jac",
        loss="linear",
        max_nfev=int(max(6, max_nfev)),
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-7,
    )
    elapsed_s = perf_counter() - started
    xopt = np.asarray(result.x, dtype=float)
    # Revalidación final a resolución completa (N del preset).
    residual0, rows0 = evaluate(x0, full=(case_key in {"rea_foz", "rea_alvorada"}), fast=False)
    residual_opt, rows_opt = evaluate(xopt, full=(case_key in {"rea_foz", "rea_alvorada"}), fast=False)
    score0 = _score_from_residuals(residual0)
    score_opt = _score_from_residuals(residual_opt)

    # Identificabilidad local por valores singulares del Jacobiano escalado.
    jac = np.asarray(result.jac, dtype=float)
    singular_values = np.linalg.svd(jac, compute_uv=False) if jac.size else np.asarray([], dtype=float)
    if singular_values.size and singular_values[-1] > np.finfo(float).eps:
        condition_number = float(singular_values[0] / singular_values[-1])
    elif singular_values.size:
        condition_number = float("inf")
    else:
        condition_number = float("nan")
    jac_rank = int(np.linalg.matrix_rank(jac)) if jac.size else 0

    param_rows: list[dict[str, Any]] = []
    for pid, start, value, lo, hi in zip(parameter_ids, x0, xopt, lower, upper):
        span = hi - lo
        boundary_distance = min(value - lo, hi - value) / span if span > 0 else 0.0
        param_rows.append(
            {
                "ID": pid,
                "Parametro": registry[pid]["label"],
                "Categoria": registry[pid]["category"],
                "Nominal": float(start),
                "Identificado": float(value),
                "Cambio_pct": 100.0 * (float(value) - float(start)) / max(abs(float(start)), 1e-12),
                "Limite_inf": float(lo),
                "Limite_sup": float(hi),
                "Cerca_del_limite": bool(boundary_distance < 0.03),
                "Estado_fuente": registry[pid]["status"],
            }
        )
    parameter_table = pd.DataFrame(param_rows)

    predictions_before = pd.DataFrame(rows0)
    predictions_after = pd.DataFrame(rows_opt)

    def subset_score(table: pd.DataFrame, subset: str) -> float:
        part = table.loc[table["Conjunto"] == subset]
        if part.empty:
            return float("nan")
        if "Residual_obj" in part.columns:
            return _score_from_residuals(part["Residual_obj"].to_numpy(float))
        rel = (part["Modelo"].to_numpy(float) - part["Referencia"].to_numpy(float)) / np.maximum(
            np.abs(part["Referencia"].to_numpy(float)), 1.0
        )
        return _score_from_residuals(rel)

    # Las tablas mensuales incluyen hold-out; estos scores son descriptivos sobre
    # magnitudes reportadas. El score de optimización usa ΔT y eta para evitar que
    # Tout absoluto oculte el error térmico.
    validation_summary: dict[str, float] = {}
    if case_key in {"rea_foz", "rea_alvorada"} and str(monthly_strategy).lower() == "alternating":
        validation_summary = {
            "holdout_score_before_pct": subset_score(predictions_before, "validación"),
            "holdout_score_after_pct": subset_score(predictions_after, "validación"),
        }

    near_bound = bool(parameter_table["Cerca_del_limite"].any()) if not parameter_table.empty else False
    physically_admissible = bool(np.all(xopt >= lower) and np.all(xopt <= upper))
    identifiable = bool(jac_rank == len(parameter_ids) and np.isfinite(condition_number) and condition_number < 1e6)

    notes: list[str] = []
    if case_key == "bhambare":
        notes.append(
            "Bhambare/Sukhatme usa el mismo caso para identificación y comprobación; por tanto, el ajuste es calibración y no validación independiente."
        )
    elif case_key in {"rea_foz", "rea_alvorada"}:
        if str(monthly_strategy).lower() == "alternating":
            notes.append(
                "Enero, abril, julio y octubre se usan para identificar; los otros ocho meses quedan como hold-out y no participan del ajuste."
            )
        else:
            notes.append(
                "Se usaron los 12 meses para calibrar; el resultado mide ajuste global, no capacidad predictiva fuera de muestra."
            )
    else:
        notes.append(
            "Tabela 8 no publica Tin/Tout horarios. Esta identificación es exploratoria porque mantiene Tin=25 °C como hipótesis del preset."
        )
    if near_bound:
        notes.append(
            "Al menos un parámetro quedó a menos del 3 % de un límite físico impuesto; esto sugiere revisar la estructura del modelo o ampliar evidencia documental antes de aceptar el ajuste."
        )
    if not identifiable:
        notes.append(
            "El Jacobiano indica identificación débil o correlación fuerte entre parámetros; no interprete los valores individuales como únicos aunque el ajuste mejore."
        )

    return {
        "case": case_key,
        "parameter_table": parameter_table,
        "predictions_before": predictions_before,
        "predictions_after": predictions_after,
        "score_before_pct": float(score0),
        "score_after_pct": float(score_opt),
        "improvement_pp": float(score0 - score_opt),
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "cpu_s": float(elapsed_s),
        "condition_number": condition_number,
        "jacobian_rank": jac_rank,
        "n_parameters": len(parameter_ids),
        "physically_admissible": physically_admissible,
        "locally_identifiable": identifiable,
        "validation_summary": validation_summary,
        "notes": notes,
    }
