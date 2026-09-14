"""Interfaz Streamlit para el modelo nodal del colector PTC."""

from __future__ import annotations

import io
import json
from copy import deepcopy
from typing import Any, Mapping

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


from defaults import default_config, default_fluid_database
from interactive_visuals import ptc_optical_component_html, thermal_circuit_component_html
from presets import MONTH_NAMES_ES, PRESET_FAMILY_LABELS, build_preset, preset_summary_rows
from fluid_properties import FluidPropertyEvaluator, property_curve
from ptc_model import PTCSimulator, SimulationResult, effective_sky_temperature
from technical_report import build_technical_report, result_summary
from validations import (
    analyze_bhambare_numerical_convergence,
    analyze_bhambare_physical_sensitivity,
    compare_bhambare_solvers,
    calibrate_inverse_model,
    inverse_parameter_options,
    identified_parameter_template,
    apply_identified_parameter_template,
    apply_calibrated_parameters,
    bhambare_user_inverse_template,
    validate_bhambare_mode,
    prototype_tcc_table,
    prototype_user_export_template,
    validate_rea_prototype_mode,
    validate_active_preset,
    validate_bhambare,
    validate_rea_quille_city_monthly,
    validate_tcc_monthly,
)
from visualizations import (
    axial_profiles,
    bhambare_solver_comparison_figure,
    comparative_overview,
    daily_irradiance_histogram,
    dynamic_overview,
    node_balance,
    numerical_convergence_figure,
    property_figure,
    sensitivity_tornado_figure,
    validation_bhambare_figure,
    validation_tcc_figure,
)


st.set_page_config(
    page_title="PTC nodal en Python",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_state() -> None:
    if "config" not in st.session_state or "fluid_database" not in st.session_state:
        initial_cfg, initial_db = build_preset("base")
        st.session_state.config = initial_cfg
        st.session_state.fluid_database = initial_db
    if "results" not in st.session_state:
        st.session_state.results = {}
    if "result_signature" not in st.session_state:
        st.session_state.result_signature = None
    if "validations" not in st.session_state:
        st.session_state.validations = {}
    if "loaded_package_name" not in st.session_state:
        st.session_state.loaded_package_name = None
    if "ui_revision" not in st.session_state:
        st.session_state.ui_revision = 0
    if "preset_family_selector" not in st.session_state:
        st.session_state.preset_family_selector = "base"
    if "preset_month_selector" not in st.session_state:
        st.session_state.preset_month_selector = 1
    if "preset_use_annual" not in st.session_state:
        st.session_state.preset_use_annual = False
    if "preset_prototype_mode_selector" not in st.session_state:
        st.session_state.preset_prototype_mode_selector = "trnsys_published"
    if "preset_prototype_dni_source" not in st.session_state:
        st.session_state.preset_prototype_dni_source = "temporal_clear_sky_905"
    if "ray_seed" not in st.session_state:
        st.session_state.ray_seed = 0

    # Streamlit no permite modificar el estado de un widget después de que
    # ese widget ya fue instanciado en el mismo rerun. Las acciones que
    # necesitan cambiar selectores del sidebar dejan aquí una actualización
    # pendiente, que se consume al inicio del rerun siguiente, antes de
    # crear los widgets.
    pending_widget_state = st.session_state.pop("_pending_widget_state", None)
    if isinstance(pending_widget_state, dict):
        for key, value in pending_widget_state.items():
            st.session_state[key] = value


def integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Integra por la regla trapezoidal con NumPy 2.x / Python 3.14."""
    return float(np.trapezoid(y, x))


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def project_signature() -> str:
    payload = {
        "config": json_ready(st.session_state.config),
        "fluid_database": json_ready(st.session_state.fluid_database),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def reset_project() -> None:
    base_cfg, base_db = build_preset("base")
    st.session_state.config = base_cfg
    st.session_state.fluid_database = base_db
    st.session_state.results = {}
    st.session_state.validations = {}
    st.session_state.result_signature = None
    st.session_state.loaded_package_name = None
    st.session_state.ui_revision += 1


def apply_reference_preset(
    family: str,
    month: int | None = None,
    annual: bool = False,
    *,
    variant: str | None = None,
    dni_source: str = "nominal",
) -> None:
    new_cfg, new_fluid_db = build_preset(
        family, month=month, annual=annual, variant=variant, dni_source=dni_source
    )
    st.session_state.config = new_cfg
    st.session_state.fluid_database = new_fluid_db
    st.session_state.results = {}
    st.session_state.validations = {}
    st.session_state.result_signature = None
    st.session_state.loaded_package_name = None
    st.session_state.ui_revision += 1


def widget_key(name: str) -> str:
    return f"{name}_{int(st.session_state.ui_revision)}"


def ensure_constant_properties(fluid_key: str) -> None:
    spec = st.session_state.fluid_database[fluid_key]
    if "constants" in spec:
        return
    if fluid_key == "ParathermNF":
        evaluator = FluidPropertyEvaluator(fluid_key, st.session_state.fluid_database)
        prop = evaluator(100.0 + 273.15)
    elif fluid_key == "Agua":
        evaluator = FluidPropertyEvaluator(fluid_key, st.session_state.fluid_database)
        prop = evaluator(50.0 + 273.15)
    else:
        return
    spec["constants"] = {
        "rho_kg_m3": prop.rho,
        "mu_Pa_s": prop.mu,
        "Cp_J_kgK": prop.Cp,
        "k_W_mK": prop.k,
    }


def load_project_package(uploaded_file: Any) -> None:
    if uploaded_file is None:
        return
    if uploaded_file.name == st.session_state.loaded_package_name:
        return
    payload = json.loads(uploaded_file.getvalue().decode("utf-8"))
    if "config" not in payload or "fluid_database" not in payload:
        raise ValueError("El JSON debe contener 'config' y 'fluid_database'.")
    st.session_state.config = payload["config"]
    st.session_state.fluid_database = payload["fluid_database"]
    st.session_state.results = {}
    st.session_state.validations = {}
    st.session_state.result_signature = None
    st.session_state.loaded_package_name = uploaded_file.name
    st.session_state.ui_revision += 1


def parse_mass_flows(text: str) -> list[float]:
    values = []
    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value <= 0.0:
            raise ValueError("Todos los caudales deben ser positivos.")
        values.append(value)
    if not values:
        raise ValueError("Ingrese al menos un caudal.")
    return values


def active_result_selector(location: str) -> tuple[str | None, SimulationResult | None]:
    results: dict[str, SimulationResult] = st.session_state.results
    if not results:
        return None, None
    labels = list(results.keys())
    label = st.selectbox("Escenario mostrado", labels, key=f"scenario_{location}")
    return label, results[label]


def _format_lat_hour(hour: float | None) -> str:
    if hour is None or not np.isfinite(hour):
        return "—"
    total_minutes = int(round(float(hour) * 60.0))
    total_minutes = max(0, min(total_minutes, 24 * 60))
    if total_minutes == 24 * 60:
        return "24:00"
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"


def daily_solar_diagnostics(config: dict[str, Any], fluid_database: dict[str, Any]) -> dict[str, Any]:
    """Evalúa el modelo solar seleccionado durante un día LAT completo.

    Se usan 1440 intervalos de un minuto, evaluados en su punto medio. La
    tarjeta de horas de sol representa las horas para las que el propio modelo
    entrega DNI > 1 W/m². No es una medición meteorológica externa.
    """
    simulator = PTCSimulator(deepcopy(config), deepcopy(fluid_database))
    minute_index = np.arange(24 * 60, dtype=int)
    lat_h = (minute_index + 0.5) / 60.0
    solar_rows = [simulator.solar_model(float(hour * 3600.0)) for hour in lat_h]

    dni = np.asarray([row["DNI_W_m2"] for row in solar_rows], dtype=float)
    cos_theta = np.asarray([row["cosTheta"] for row in solar_rows], dtype=float)
    beam_aperture = dni * np.clip(cos_theta, 0.0, 1.0)
    sun_mask = np.isfinite(dni) & (dni > 1.0)

    sun_minutes = int(np.count_nonzero(sun_mask))
    sun_hours = sun_minutes / 60.0
    dni_daily_kWh_m2 = float(np.nansum(np.where(np.isfinite(dni), dni, 0.0))) / 60.0 / 1000.0
    beam_daily_kWh_m2 = float(np.nansum(np.where(np.isfinite(beam_aperture), beam_aperture, 0.0))) / 60.0 / 1000.0

    if sun_minutes:
        lit = np.flatnonzero(sun_mask)
        sunrise_h = float(lit[0]) / 60.0
        sunset_h = float(lit[-1] + 1) / 60.0
    else:
        sunrise_h = None
        sunset_h = None

    hourly = pd.DataFrame(
        {
            "hour": np.arange(24, dtype=int),
            "hour_label": [f"{h:02d}–{h + 1:02d}" for h in range(24)],
            "DNI_mean_W_m2": np.nanmean(dni.reshape(24, 60), axis=1),
            "beam_on_aperture_mean_W_m2": np.nanmean(beam_aperture.reshape(24, 60), axis=1),
        }
    )

    return {
        "sun_hours": sun_hours,
        "sunrise_h": sunrise_h,
        "sunset_h": sunset_h,
        "DNI_max_W_m2": float(np.nanmax(dni)) if np.any(np.isfinite(dni)) else np.nan,
        "DNI_daily_kWh_m2": dni_daily_kWh_m2,
        "beam_daily_kWh_m2": beam_daily_kWh_m2,
        "hourly": hourly,
    }


def representative_time_index(result: SimulationResult) -> tuple[int, str]:
    """Selecciona un instante útil para KPIs y análisis nodal.

    Con DNI constante no se debe usar argmax(DNI), porque todos los puntos son
    iguales y NumPy devuelve t0. En t0 el modelo todavía está en su condición
    inicial, por lo que Q hacia el HTF es cero aunque después exista transporte
    continuo de energía. Para irradiación constante se usa el estado final;
    para irradiación variable se usa el último punto del máximo de DNI.
    """
    dni_series = np.asarray(result.scalar_diag["DNI_W_m2"], dtype=float)
    finite_idx = np.flatnonzero(np.isfinite(dni_series))
    if finite_idx.size == 0:
        return len(result.t_s) - 1, "Estado final"

    dni_valid = dni_series[finite_idx]
    dni_scale = max(float(np.nanmax(np.abs(dni_valid))), 1.0)
    dni_is_constant = float(np.nanmax(dni_valid) - np.nanmin(dni_valid)) <= 1e-8 * dni_scale
    if dni_is_constant:
        return int(finite_idx[-1]), "Estado final"

    max_dni = float(np.nanmax(dni_valid))
    peak_candidates = finite_idx[
        np.isclose(dni_series[finite_idx], max_dni, rtol=1e-10, atol=1e-9)
    ]
    if peak_candidates.size:
        return int(peak_candidates[-1]), "DNI máximo"
    return int(finite_idx[np.nanargmax(dni_valid)]), "DNI máximo"


def render_axial_node_selector(
    result: SimulationResult,
    time_index: int,
    state_key: str,
    scenario_label: str,
) -> int:
    """Renderiza el selector axial con botones Streamlit estilizados como volúmenes.

    Se usan botones nativos para que el clic actualice de forma fiable el estado
    de Streamlit. La apariencia se resuelve con CSS: bloques grises contiguos,
    flecha de transporte axial y DeltaH dentro de cada volumen.
    """
    n = result.n_segments
    k = int(np.clip(time_index, 0, len(result.t_s) - 1))
    selected = int(np.clip(st.session_state[state_key], 1, n))
    d_h = -np.asarray(result.node_diag["Qadvection_W"][k, :], dtype=float)

    safe_label = "".join(ch if ch.isalnum() else "_" for ch in str(scenario_label))[-48:]
    shell_key = f"node_selector_shell_{safe_label}_{k}"

    st.markdown(
        f"""
        <style>
        .st-key-{shell_key} [data-testid="stHorizontalBlock"] {{
            gap: 4px !important;
        }}
        .st-key-{shell_key} button {{
            min-height: 92px !important;
            height: 92px !important;
            padding: 8px 3px !important;
            border-radius: 5px !important;
            border: 1px solid #cbd5e1 !important;
            background: #e5e7eb !important;
            color: #111827 !important;
            box-shadow: none !important;
            transition: background .14s ease, border-color .14s ease, transform .10s ease !important;
        }}
        .st-key-{shell_key} button:hover {{
            background: #d7dce2 !important;
            border-color: #94a3b8 !important;
            transform: translateY(-1px);
        }}
        .st-key-{shell_key} button p {{
            white-space: pre-line !important;
            text-align: center !important;
            line-height: 1.13 !important;
            font-size: 12px !important;
            font-weight: 650 !important;
            margin: 0 !important;
        }}
        """,
        unsafe_allow_html=True,
    )

    # Regla específica para el nodo activo. Se inyecta por key, sin depender
    # de los colores de tema de Streamlit.
    active_button_key = f"node_tile_{safe_label}_{k}_{selected}"
    st.markdown(
        f"""
        <style>
        .st-key-{active_button_key} button {{
            background: #59616d !important;
            color: white !important;
            border-color: #374151 !important;
        }}
        .st-key-{active_button_key} button:hover {{
            background: #4b5563 !important;
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key=shell_key):
        cols = st.columns(n, gap="small")
        for i, col in enumerate(cols):
            q = float(d_h[i])
            arrow = "→" if q >= 0.0 else "←"
            label = f"{i + 1}\n{arrow}\nΔH {abs(q):.1f} W"
            button_key = f"node_tile_{safe_label}_{k}_{i + 1}"
            if col.button(label, key=button_key, use_container_width=True):
                st.session_state[state_key] = i + 1
                st.rerun()

    return int(st.session_state[state_key])


def solar_input_is_temporally_variable(result: SimulationResult) -> bool:
    """True si la potencia solar absorbida cambia de forma apreciable en el tiempo.

    Se evalúa la fuente térmica realmente aplicada al receptor, no solamente DNI.
    Así un DNI constante con geometría/IAM variables sigue considerándose una
    entrada solar temporal.
    """
    q_abs = np.asarray(result.scalar_diag.get("QsolarAbs_W", []), dtype=float)
    q_glass = np.asarray(result.scalar_diag.get("QsolarGlass_W", []), dtype=float)
    if q_abs.size == 0:
        return False
    q = q_abs + (q_glass if q_glass.size == q_abs.size else 0.0)
    finite = q[np.isfinite(q)]
    if finite.size < 2:
        return False
    scale = max(float(np.nanmax(np.abs(finite))), 1.0)
    return float(np.nanmax(finite) - np.nanmin(finite)) > 1e-5 * scale


def calibration_error_summary(calibration: Mapping[str, Any] | None) -> dict[str, float]:
    """Indicadores rápidos del hold-out de la última calibración/validación."""
    if not isinstance(calibration, Mapping):
        return {}
    table = calibration.get("predictions_after")
    if not isinstance(table, pd.DataFrame) or table.empty or "Conjunto" not in table.columns:
        return {}
    hold = table.loc[table["Conjunto"] == "validación"].copy()
    if hold.empty:
        return {}

    out: dict[str, float] = {}
    for magnitude, prefix in (("Eta_pct", "eta"), ("Tout_C", "tout")):
        part = hold.loc[hold["Magnitud"] == magnitude]
        if part.empty:
            continue
        ref = part["Referencia"].to_numpy(float)
        model = part["Modelo"].to_numpy(float)
        err = model - ref
        out[f"{prefix}_mae"] = float(np.mean(np.abs(err)))
        out[f"{prefix}_rmse"] = float(np.sqrt(np.mean(err**2)))
        out[f"{prefix}_bias"] = float(np.mean(err))
        valid = np.abs(ref) > 1e-12
        out[f"{prefix}_mape_pct"] = float(np.mean(np.abs(err[valid] / ref[valid])) * 100.0) if np.any(valid) else float("nan")
        out[f"{prefix}_max_rel_pct"] = float(np.max(np.abs(err[valid] / ref[valid])) * 100.0) if np.any(valid) else float("nan")

    validation_summary = calibration.get("validation_summary", {}) or {}
    if "holdout_score_after_pct" in validation_summary:
        out["holdout_score_pct"] = float(validation_summary["holdout_score_after_pct"])
    if "holdout_score_before_pct" in validation_summary:
        out["holdout_score_before_pct"] = float(validation_summary["holdout_score_before_pct"])
    return out


def validation_comparison_table(calibration: Mapping[str, Any] | None) -> pd.DataFrame:
    if not isinstance(calibration, Mapping):
        return pd.DataFrame()
    before = calibration.get("predictions_before")
    after = calibration.get("predictions_after")
    if not isinstance(before, pd.DataFrame) or not isinstance(after, pd.DataFrame):
        return pd.DataFrame()
    before = before.copy().rename(columns={
        "Modelo": "Modelo_inicial",
        "Error_rel_pct": "Error_inicial_pct",
        "Residual_obj": "Residual_inicial",
    })
    after = after.copy().rename(columns={
        "Modelo": "Modelo_calibrado",
        "Error_rel_pct": "Error_calibrado_pct",
        "Residual_obj": "Residual_calibrado",
    })
    keys = ["Conjunto", "Caso", "Magnitud", "Referencia"]
    return before.merge(after, on=keys, how="outer")


initialize_state()
cfg = st.session_state.config
fluid_db = st.session_state.fluid_database

header_left, header_right = st.columns([5.2, 1.35], vertical_alignment="center")
with header_left:
    st.title("Modelo nodal de colector cilindro-parabólico")
with header_right:
    run_clicked = st.button(
        "▶ Ejecutar simulación",
        type="primary",
        use_container_width=True,
        key="run_simulation_top",
        help="Ejecuta la configuración actual. El botón permanece arriba para evitar bajar por la página.",
    )

with st.sidebar:
    st.header("Configuración")

    st.subheader("Preset documental")
    preset_family = st.selectbox(
        "Caso de referencia",
        list(PRESET_FAMILY_LABELS.keys()),
        format_func=lambda key: PRESET_FAMILY_LABELS[key],
        key="preset_family_selector",
    )
    preset_month = None
    preset_annual = False
    if preset_family in {"rea_foz_monthly", "rea_alvorada_monthly"}:
        preset_scope = st.radio(
            "Referencia temporal",
            ["Mes", "Promedio anual"],
            horizontal=True,
            key="preset_scope_selector",
        )
        preset_annual = preset_scope == "Promedio anual"
        if not preset_annual:
            preset_month = st.selectbox(
                "Mes de Rea Quille",
                list(range(1, 13)),
                index=max(0, int(st.session_state.get("preset_month_selector", 1)) - 1),
                format_func=lambda value: MONTH_NAMES_ES[value - 1],
                key="preset_month_selector",
            )

    prototype_variant = None
    prototype_dni_source = "nominal"
    if preset_family == "rea_prototype":
        prototype_mode_labels = {
            "trnsys_published": "TRNSYS publicado · DNI 905 / IAM=1",
            "physical_fixed_ns": "Físico corregido · fijo N-S / IAM variable",
        }
        prototype_variant = st.radio(
            "Modo del prototipo",
            list(prototype_mode_labels.keys()),
            format_func=lambda key: prototype_mode_labels[key],
            key="preset_prototype_mode_selector",
        )
        if prototype_variant == "physical_fixed_ns":
            dni_labels = {
                "temporal_clear_sky_905": "DNI temporal recomendado · cielo claro, 905 W/m² al mediodía solar",
                "nominal": "DNI nominal 905 W/m² (aislar efecto geométrico)",
                "clear_sky": "DNI variable por cielo claro A·exp(-B/cos z)",
            }
            prototype_dni_source = st.selectbox(
                "Fuente DNI del modo físico",
                list(dni_labels.keys()),
                format_func=lambda key: dni_labels[key],
                key="preset_prototype_dni_source",
            )
            st.caption("El TCC no publica una serie DNI horaria medida. El perfil temporal recomendado reproduce la geometría solar y normaliza el cielo claro a 905 W/m² al mediodía; sigue siendo un modelo, no una medición.")

    if st.button("Aplicar preset completo", type="primary", use_container_width=True):
        apply_reference_preset(
            preset_family,
            month=preset_month,
            annual=preset_annual,
            variant=prototype_variant,
            dni_source=prototype_dni_source,
        )
        st.rerun()

    active_meta = cfg.get("preset_meta", {})
    if active_meta:
        st.caption(f"Activo: {active_meta.get('family', '—')} · {active_meta.get('date_label', '—')}")
        with st.expander("Ver parámetros fijados y supuestos", expanded=False):
            st.dataframe(
                pd.DataFrame(preset_summary_rows(cfg), columns=["Campo", "Valor"]),
                use_container_width=True,
                hide_index=True,
            )
            assumptions = active_meta.get("assumptions", [])
            if assumptions:
                st.warning("La fuente no informa todos los parámetros requeridos por este modelo. Estos valores quedan explícitamente marcados como supuestos:")
                for item in assumptions:
                    st.markdown(f"- {item}")

    st.divider()
    uploaded = st.file_uploader("Cargar proyecto JSON", type=["json"])
    if uploaded is not None:
        try:
            load_project_package(uploaded)
            st.success("Proyecto cargado.")
            cfg = st.session_state.config
            fluid_db = st.session_state.fluid_database
        except Exception as exc:
            st.error(str(exc))

    if st.button("Restablecer valores de referencia", use_container_width=True):
        reset_project()
        st.rerun()

    with st.expander("Geometría", expanded=True):
        g = cfg["geometry"]
        g["L"] = st.number_input("Longitud L (m)", min_value=0.01, value=float(g["L"]), step=0.1)
        g["W"] = st.number_input("Abertura W (m)", min_value=0.01, value=float(g["W"]), step=0.05)
        g["f"] = st.number_input("Distancia focal f (m)", min_value=0.001, value=float(g["f"]), step=0.01)
        g["D2"] = st.number_input("D2 interno absorbedor (m)", min_value=0.001, value=float(g["D2"]), format="%.5f")
        g["D3"] = st.number_input("D3 externo absorbedor (m)", min_value=0.001, value=float(g["D3"]), format="%.5f")
        g["D4"] = st.number_input("D4 interno vidrio (m)", min_value=0.001, value=float(g["D4"]), format="%.5f")
        g["D5"] = st.number_input("D5 externo vidrio (m)", min_value=0.001, value=float(g["D5"]), format="%.5f")
        g["Nseg"] = st.number_input("Número de nodos axiales", min_value=1, max_value=100, value=int(g["Nseg"]), step=1)

    with st.expander("Operación y ambiente", expanded=True):
        operation = cfg["operation"]
        environment = cfg["environment"]
        operation["fluid"] = st.selectbox(
            "Fluido activo",
            list(fluid_db.keys()),
            index=list(fluid_db.keys()).index(operation["fluid"]),
        )
        operation["mdot"] = st.number_input("Caudal másico (kg/s)", min_value=1e-5, value=float(operation["mdot"]), format="%.6f")
        Tin_C = st.number_input("Temperatura de entrada (°C)", value=float(operation["Tin_K"] - 273.15), step=1.0)
        operation["Tin_K"] = Tin_C + 273.15
        Tamb_C = st.number_input("Temperatura ambiente (°C)", value=float(environment["Tamb_K"] - 273.15), step=1.0)
        environment["Tamb_K"] = Tamb_C + 273.15

        sky_labels = {
            "rea_quille": "Rea Quille / Martin-Berdahl (Ecs. 12-14)",
            "delta_constante": "Legado: Tsky = Tamb - ΔT",
        }
        sky_modes = list(sky_labels.keys())
        current_sky = str(environment.get("sky_model", "rea_quille"))
        if current_sky not in sky_modes:
            current_sky = "rea_quille"
        environment["sky_model"] = st.selectbox(
            "Modelo de temperatura efectiva del cielo",
            sky_modes,
            index=sky_modes.index(current_sky),
            format_func=lambda x: sky_labels[x],
        )
        if environment["sky_model"] == "rea_quille":
            environment["dew_point_C"] = st.number_input(
                "Temperatura de punto de rocío Tdp (°C)",
                value=float(environment.get("dew_point_C", 20.0)),
                step=0.5,
                help="Entrada del Type15-3 en Rea Quille. Las tablas mensuales del TCC no publican Tdp; en los presets se marca como hipótesis cuando no está disponible.",
            )
            environment["cloud_adjustment"] = st.checkbox(
                "Aplicar corrección por nubosidad (Ec. 13)",
                value=bool(environment.get("cloud_adjustment", False)),
            )
            if environment["cloud_adjustment"]:
                environment["cloud_factor"] = st.number_input(
                    "Factor de nubosidad f_nuvem",
                    min_value=0.0, max_value=1.0,
                    value=float(environment.get("cloud_factor", 0.0)), step=0.05,
                )
                environment["cloud_emissivity"] = st.number_input(
                    "Emisividad de nube ε_nuvem",
                    min_value=0.0, max_value=1.5,
                    value=float(environment.get("cloud_emissivity", 1.0)), step=0.05,
                )
                formula_labels = {
                    "rea_quille_impresa": "Ec. (13) impresa: ε = ε0 + (1 + ε0)·f·εnube",
                    "variante_fisica": "Sensibilidad: ε = ε0 + (1 - ε0)·f·εnube",
                }
                formula_keys = list(formula_labels.keys())
                current_formula = str(environment.get("cloud_formula", "rea_quille_impresa"))
                if current_formula not in formula_keys:
                    current_formula = "rea_quille_impresa"
                environment["cloud_formula"] = st.selectbox(
                    "Convención para la Ec. (13)", formula_keys,
                    index=formula_keys.index(current_formula),
                    format_func=lambda x: formula_labels[x],
                )
                if environment["cloud_formula"] == "rea_quille_impresa":
                    st.warning(
                        "El TCC imprime (1 + ε0) en la Ec. (13), pero el texto también afirma que f_nuvem=0 representa cielo totalmente nublado. Ambas afirmaciones no son mutuamente consistentes. La app conserva la ecuación impresa sin corregirla silenciosamente."
                    )
        else:
            environment["sky_delta_K"] = st.number_input(
                "Tamb - Tsky (K)", min_value=0.0,
                value=float(environment.get("sky_delta_K", 6.0)), step=0.5
            )

        environment["wind_m_s"] = st.number_input("Viento (m/s)", min_value=0.0, value=float(environment["wind_m_s"]), step=0.1)
        environment["pressure_Pa"] = st.number_input("Presión ambiente (Pa)", min_value=1000.0, value=float(environment["pressure_Pa"]), step=100.0)
        t_start_h = st.number_input("Hora inicial LAT", value=float(operation["t_start_s"] / 3600.0), step=0.5)
        t_end_h = st.number_input("Hora final LAT", value=float(operation["t_end_s"] / 3600.0), step=0.5)
        operation["t_start_s"] = t_start_h * 3600.0
        operation["t_end_s"] = t_end_h * 3600.0
        operation["output_step_s"] = st.number_input("Paso de salida (s)", min_value=1.0, value=float(operation["output_step_s"]), step=10.0)

    with st.expander("Irradiación y óptica", expanded=True):
        solar = cfg["solar"]
        optics = cfg["optics"]
        mode_labels = {
            "Parishwad": "Parishwad / cielo claro · tracking N-S",
            "constante": "DNI y ángulo constantes",
            "fijo_horizontal": "Fijo horizontal / geometría solar explícita",
            "perfil": "Perfil horario editable",
        }
        solar_modes = list(mode_labels.keys())
        current_mode = solar["mode"] if solar["mode"] in solar_modes else "Parishwad"
        selected_mode = st.selectbox(
            "Modelo de irradiación",
            solar_modes,
            index=solar_modes.index(current_mode),
            format_func=lambda key: mode_labels[key],
        )
        solar["mode"] = selected_mode
        if selected_mode == "Parishwad":
            solar["A"] = st.number_input("Constante A (W/m²)", min_value=0.0, value=float(solar["A"]), step=1.0)
            solar["B"] = st.number_input("Constante B", min_value=0.0, value=float(solar["B"]), step=0.001, format="%.4f")
            solar["day_of_year"] = st.number_input("Día del año", min_value=1, max_value=366, value=int(solar["day_of_year"]), step=1)
            solar["latitude_deg"] = st.number_input("Latitud (°)", min_value=-90.0, max_value=90.0, value=float(solar["latitude_deg"]), step=0.1)
        elif selected_mode == "constante":
            solar["DNI_constant_W_m2"] = st.number_input("DNI constante (W/m²)", min_value=0.0, value=float(solar["DNI_constant_W_m2"]), step=10.0)
            solar["angle_constant_deg"] = st.number_input("Ángulo de incidencia (°)", min_value=0.0, max_value=90.0, value=float(solar["angle_constant_deg"]), step=1.0)
        elif selected_mode == "fijo_horizontal":
            solar["day_of_year"] = st.number_input("Día del año", min_value=1, max_value=366, value=int(solar["day_of_year"]), step=1, key="fixed_doy")
            solar["latitude_deg"] = st.number_input("Latitud (°)", min_value=-90.0, max_value=90.0, value=float(solar["latitude_deg"]), step=0.1, key="fixed_lat")
            source_labels = {"temporal_clear_sky_905": "Temporal cielo claro · 905 W/m² al mediodía", "nominal": "DNI nominal constante", "clear_sky": "Cielo claro A·exp(-B/cos z)"}
            source_keys = list(source_labels.keys())
            current_source = str(solar.get("fixed_dni_source", "nominal"))
            if current_source not in source_keys:
                current_source = "nominal"
            solar["fixed_dni_source"] = st.selectbox(
                "Fuente DNI", source_keys, index=source_keys.index(current_source), format_func=lambda key: source_labels[key], key="fixed_dni_source_ui"
            )
            if solar["fixed_dni_source"] == "nominal":
                solar["DNI_constant_W_m2"] = st.number_input("DNI nominal (W/m²)", min_value=0.0, value=float(solar["DNI_constant_W_m2"]), step=10.0, key="fixed_dni_nominal")
            elif solar["fixed_dni_source"] == "temporal_clear_sky_905":
                solar["DNI_noon_W_m2"] = st.number_input("DNI al mediodía solar (W/m²)", min_value=0.0, value=float(solar.get("DNI_noon_W_m2", 905.0)), step=10.0, key="fixed_dni_noon")
                solar["B"] = st.number_input("Coeficiente atmosférico B", min_value=0.0, value=float(solar["B"]), step=0.001, format="%.4f", key="fixed_B_norm")
                solar["utc_offset_h"] = st.number_input("UTC local (h)", min_value=-12.0, max_value=14.0, value=float(solar.get("utc_offset_h", -3.0)), step=1.0, key="fixed_utc")
                solar["clock_time_correction"] = st.checkbox("Convertir hora civil → hora solar aparente", value=bool(solar.get("clock_time_correction", True)), key="fixed_clock_corr")
            else:
                solar["A"] = st.number_input("Constante A (W/m²)", min_value=0.0, value=float(solar["A"]), step=1.0, key="fixed_A")
                solar["B"] = st.number_input("Constante B", min_value=0.0, value=float(solar["B"]), step=0.001, format="%.4f", key="fixed_B")
            st.caption("Sin tracking: el colector permanece fijo. θ, irradiancia proyectada sobre la apertura, IAM y EndLoss cambian con la hora.")
        else:
            st.info("Edite el perfil completo en la pestaña Propiedades e irradiación.")
        optics["reflectivity"] = st.number_input("Reflectividad", min_value=0.0, max_value=1.0, value=float(optics["reflectivity"]), step=0.01)
        optics["intercept_factor"] = st.number_input("Factor de interceptación", min_value=0.0, max_value=1.0, value=float(optics["intercept_factor"]), step=0.01)
        optics["dirt_factor"] = st.number_input("Factor de suciedad", min_value=0.0, max_value=1.0, value=float(optics["dirt_factor"]), step=0.01)
        optics["shade_factor"] = st.number_input("Factor de sombra", min_value=0.0, max_value=1.0, value=float(optics["shade_factor"]), step=0.01)

    with st.expander("Materiales", expanded=False):
        absorber = cfg["materials"]["absorber"]
        glass = cfg["materials"]["glass"]
        st.markdown("**Absorbedor**")
        absorber["rho"] = st.number_input("rho absorbedor (kg/m³)", min_value=1.0, value=float(absorber["rho"]), key="abs_rho")
        absorber["Cp"] = st.number_input("Cp absorbedor (J/kg K)", min_value=1.0, value=float(absorber["Cp"]), key="abs_cp")
        absorber["k"] = st.number_input("k absorbedor (W/m K)", min_value=0.001, value=float(absorber["k"]), key="abs_k")
        absorber["eps"] = st.number_input("Emisividad absorbedor", min_value=0.001, max_value=1.0, value=float(absorber["eps"]), key="abs_eps")
        absorber["alpha"] = st.number_input("Absortancia absorbedor", min_value=0.0, max_value=1.0, value=float(absorber["alpha"]), key="abs_alpha")
        st.markdown("**Vidrio**")
        glass["rho"] = st.number_input("rho vidrio (kg/m³)", min_value=1.0, value=float(glass["rho"]), key="glass_rho")
        glass["Cp"] = st.number_input("Cp vidrio (J/kg K)", min_value=1.0, value=float(glass["Cp"]), key="glass_cp")
        glass["k"] = st.number_input("k vidrio (W/m K)", min_value=0.001, value=float(glass["k"]), key="glass_k")
        glass["eps"] = st.number_input("Emisividad vidrio", min_value=0.001, max_value=1.0, value=float(glass["eps"]), key="glass_eps")
        glass["tau"] = st.number_input("Transmitancia vidrio", min_value=0.0, max_value=1.0, value=float(glass["tau"]), key="glass_tau")
        glass["alpha"] = st.number_input("Absortancia vidrio", min_value=0.0, max_value=1.0, value=float(glass["alpha"]), key="glass_alpha")

    with st.expander("Modelo y solver", expanded=False):
        model = cfg["model"]
        solver = cfg["solver"]
        model["has_glass"] = st.checkbox("Receptor con cubierta de vidrio", value=bool(model["has_glass"]))
        annulus_options = ["vacio_ideal", "vacio_efectivo", "aire"]
        model["annulus"] = st.selectbox("Modelo del anular", annulus_options, index=annulus_options.index(model["annulus"]))
        if model["annulus"] == "vacio_efectivo":
            model["annulus_h_effective_W_m2K"] = st.number_input(
                "h anular efectivo (W/m² K)", min_value=0.0, value=float(model["annulus_h_effective_W_m2K"]), step=0.01
            )
        correlation_options = ["automatica", "dittusboelter", "dittusboelter_forzado", "gnielinski"]
        model["internal_correlation"] = st.selectbox(
            "Correlación interna",
            correlation_options,
            index=correlation_options.index(model["internal_correlation"]),
        )
        if model["internal_correlation"] != "dittusboelter_forzado":
            re_cols = st.columns(2)
            model["Re_laminar_max"] = re_cols[0].number_input(
                "Fin de régimen laminar (Re)",
                min_value=1.0,
                value=float(model.get("Re_laminar_max", 2300.0)),
                step=100.0,
            )
            model["Re_turbulent_min"] = re_cols[1].number_input(
                "Inicio de régimen turbulento (Re)",
                min_value=float(model["Re_laminar_max"]) + 1.0,
                value=max(float(model.get("Re_turbulent_min", 4000.0)), float(model["Re_laminar_max"]) + 1.0),
                step=100.0,
            )
            st.caption(
                "Entre ambos Reynolds se interpola suavemente Nu para evitar saltos no físicos "
                "al cruzar de laminar a transición/turbulento."
            )
        model["include_supports"] = st.checkbox("Incluir pérdidas en soportes", value=bool(model["include_supports"]))
        if model["include_supports"]:
            model["support_loss_fraction"] = st.number_input(
                "Fracción de pérdidas en soportes", min_value=0.0, max_value=1.0, value=float(model["support_loss_fraction"]), step=0.005
            )
        solver_methods = ["RK45", "Radau", "BDF"]
        current_method = str(solver.get("method", "RK45"))
        if current_method not in solver_methods:
            current_method = "RK45"
        solver["method"] = st.selectbox(
            "Método de integración temporal",
            solver_methods,
            index=solver_methods.index(current_method),
            help=(
                "RK45: Runge-Kutta explícito 5(4). Radau: Runge-Kutta implícito de orden 5. "
                "BDF: fórmula de diferenciación hacia atrás para sistemas potencialmente stiff."
            ),
        )
        solver["rtol"] = st.number_input("rtol", min_value=1e-12, max_value=1e-2, value=float(solver["rtol"]), format="%.1e")
        solver["atol"] = st.number_input("atol", min_value=1e-12, max_value=1e-2, value=float(solver["atol"]), format="%.1e")
        solver["max_step_s"] = st.number_input(
            "Paso máximo del integrador (s)", min_value=0.1, value=float(solver["max_step_s"]), step=5.0
        )
        solver["use_jac_sparsity"] = st.checkbox(
            "Usar patrón disperso de Jacobiano",
            value=bool(solver["use_jac_sparsity"]),
            disabled=solver["method"] == "RK45",
            help="Solo se utiliza con BDF y Radau; RK45 es un método explícito.",
        )

    st.divider()
    sweep = st.checkbox("Barrido comparativo de caudales", value=False)
    sweep_text = st.text_input("Caudales (kg/s), separados por coma", value="0.015, 0.045, 0.090", disabled=not sweep)

if run_clicked:
    try:
        current_cfg = deepcopy(st.session_state.config)
        print(build_technical_report(current_cfg))
        results: dict[str, SimulationResult] = {}
        if sweep:
            mass_flows = parse_mass_flows(sweep_text)
        else:
            mass_flows = [float(current_cfg["operation"]["mdot"])]
        progress = st.progress(0.0, text="Preparando simulación")
        for index, mass_flow in enumerate(mass_flows):
            case = deepcopy(current_cfg)
            case["operation"]["mdot"] = mass_flow
            case["operation"]["name"] = f"{case['operation']['fluid']}, mdot = {mass_flow:.6g} kg/s"
            progress.progress(index / max(len(mass_flows), 1), text=f"Resolviendo {case['operation']['name']}")
            result = PTCSimulator(case, st.session_state.fluid_database).simulate()
            results[case["operation"]["name"]] = result
            print("\n" + result_summary(result))
        progress.progress(1.0, text="Simulación finalizada")
        st.session_state.results = results
        st.session_state.result_signature = project_signature()
        st.success("Simulación completada.")
    except Exception as exc:
        st.exception(exc)

if st.session_state.results and st.session_state.result_signature != project_signature():
    st.warning("Los parámetros visibles cambiaron después de la última simulación. Ejecute nuevamente para actualizar los resultados.")

main_section = st.radio(
    "Sección principal",
    ["Simulación", "Propiedades", "Validación", "Sensibilidad"],
    horizontal=True,
    label_visibility="collapsed",
    key="main_section_v14",
)
st.caption({
    "Simulación": "Resultados, análisis nodal y exportación del caso activo.",
    "Propiedades": "Irradiación, cielo y propiedades termofísicas del HTF.",
    "Validación": "Calibración con muestra y prueba fuera de muestra con parámetros congelados.",
    "Sensibilidad": "Convergencia numérica y efecto de perturbaciones paramétricas sobre las salidas.",
}[main_section])

if main_section == "Simulación":
    sim_tab_results, sim_tab_nodes, sim_tab_report = st.tabs(["Resultados", "Nodo por nodo", "Reporte y exportación"])
    with sim_tab_results:
        label, result = active_result_selector("simulation")
        if result is None:
            st.info("Configure el caso y pulse Ejecutar simulación.")
        else:
            k_ref, _ = representative_time_index(result)

            cols = st.columns(6)
            cols[0].metric("DNI", f"{result.scalar_diag['DNI_W_m2'][k_ref]:.1f} W/m²")
            cols[1].metric("T salida", f"{result.Tout_C[k_ref]:.2f} °C")
            cols[2].metric("T absorbedor", f"{result.Tabs_mean_C[k_ref]:.2f} °C")
            cols[3].metric("Q útil", f"{result.scalar_diag['Quseful_W'][k_ref]:.1f} W")
            cols[4].metric("Q pérdidas", f"{result.scalar_diag['Qloss_W'][k_ref]:.1f} W")
            cols[5].metric("η térmica HTF", f"{result.scalar_diag['eta_pct'][k_ref]:.2f} %")
            qinc = np.asarray(result.scalar_diag["Qincident_W"], dtype=float)
            quse = np.asarray(result.scalar_diag["Quseful_W"], dtype=float)
            einc = integrate_trapezoid(qinc, result.t_s) if len(result.t_s) > 1 else float("nan")
            euse = integrate_trapezoid(quse, result.t_s) if len(result.t_s) > 1 else float("nan")
            eta_period = 100.0 * euse / einc if np.isfinite(einc) and einc > 0.0 else float("nan")
            re_out = float(result.node_diag["Re_internal"][k_ref, -1])
            re_lam = float(result.config["model"].get("Re_laminar_max", 2300.0))
            re_turb = float(result.config["model"].get("Re_turbulent_min", 4000.0))
            regime = "laminar" if re_out <= re_lam else ("transición" if re_out < re_turb else "turbulento")
            solar_varies = solar_input_is_temporally_variable(result)
            diag_cols = st.columns(4)
            diag_cols[0].metric("η óptica al absorbedor", f"{result.scalar_diag['eta_optical_abs_pct'][k_ref]:.2f} %")
            if solar_varies:
                diag_cols[1].metric("η integrada del período", f"{eta_period:.2f} %")
            else:
                diag_cols[1].metric("Modo de resultado", "Estado representativo")
            diag_cols[2].metric("Re salida", f"{re_out:.0f}")
            diag_cols[3].metric("Régimen salida", regime)
            if solar_varies:
                if len(st.session_state.results) > 1:
                    st.plotly_chart(comparative_overview(st.session_state.results), use_container_width=True)
                else:
                    st.plotly_chart(dynamic_overview(result), use_container_width=True)
            else:
                st.info(
                    "La entrada solar aplicada al receptor es constante. Se omite la respuesta transitoria: "
                    "mostrar el calentamiento desde una condición inicial arbitraria no aporta a la comparación estacionaria. "
                    "Se muestran directamente el estado representativo y los perfiles axiales."
                )
                if len(st.session_state.results) > 1:
                    rows = []
                    for scenario_name, scenario_result in st.session_state.results.items():
                        kk, _ = representative_time_index(scenario_result)
                        rows.append({
                            "Escenario": scenario_name,
                            "Tout_C": float(scenario_result.Tout_C[kk]),
                            "Tabs_C": float(scenario_result.Tabs_mean_C[kk]),
                            "Tvid_C": float(scenario_result.Tglass_mean_C[kk]),
                            "Qutil_W": float(scenario_result.scalar_diag["Quseful_W"][kk]),
                            "Qloss_W": float(scenario_result.scalar_diag["Qloss_W"][kk]),
                            "Eta_pct": float(scenario_result.scalar_diag["eta_pct"][kk]),
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.plotly_chart(axial_profiles(result, k_ref), use_container_width=True)
            with st.expander("Resumen del solver y del escenario"):
                st.code(result_summary(result), language="text")
            scalar_csv = result.scalar_dataframe().to_csv(index=False).encode("utf-8")
            st.download_button(
                "Descargar serie temporal CSV",
                data=scalar_csv,
                file_name="resultado_ptc_temporal.csv",
                mime="text/csv",
            )


    with sim_tab_nodes:
        label, result = active_result_selector("nodes")
        if result is None:
            st.info("Primero ejecute una simulación.")
        else:
            node_default_index, _ = representative_time_index(result)
            time_index = st.slider(
                "Instante de análisis",
                min_value=0,
                max_value=len(result.t_s) - 1,
                value=node_default_index,
                format="índice %d",
                key=f"node_time_index_{label}",
            )
            state_key = f"selected_node_{label}"
            if state_key not in st.session_state:
                st.session_state[state_key] = min(1, result.n_segments)
            else:
                st.session_state[state_key] = int(np.clip(st.session_state[state_key], 1, result.n_segments))

            st.subheader("Selector axial interactivo")
            node_number = render_axial_node_selector(result, time_index, state_key, label)
            node_index = node_number - 1
            snapshot = result.node_snapshot(time_index, node_index)
            st.caption(f"LAT = {snapshot['LAT_h']:.3f} h; nodo {node_number}; x = {(node_index + 0.5) * result.config['geometry']['L'] / result.n_segments:.3f} m")

            q_to_htf = float(snapshot["Qfluid_W"])
            q_htf_rise = -float(snapshot["Qadvection_W"])
            q_fluid_storage = q_to_htf - q_htf_rise
            q_abs_to_glass = float(snapshot["Qrad_abs_glass_W"] + snapshot["Qconv_annulus_W"])
            q_external_loss = float(snapshot["Qconv_external_W"] + snapshot["Qrad_sky_W"] + snapshot["Qsupports_W"])
            flow_cols = st.columns(6)
            flow_cols[0].metric("Q solar / nodo", f"{snapshot['Qsolar_abs_node_W']:.2f} W")
            flow_cols[1].metric("Absorbedor → HTF", f"{q_to_htf:.2f} W")
            flow_cols[2].metric("ΔH axial del HTF", f"{q_htf_rise:.2f} W")
            flow_cols[3].metric("Absorbedor → vidrio", f"{q_abs_to_glass:.2f} W")
            flow_cols[4].metric("Pérdida exterior", f"{q_external_loss:.2f} W")
            flow_cols[5].metric("Acumulación HTF", f"{q_fluid_storage:.3e} W")
            st.caption(
                "Balance del HTF por nodo: C_f·dTf/dt = Q_absorbedor→HTF - ΔH_axial. "
                "En equilibrio térmico dTf/dt ≈ 0, pero ambos términos permanecen finitos y casi iguales."
            )

            st.subheader("Sección transversal interactiva del PTC")
            components.html(
                ptc_optical_component_html(result.config, snapshot, node_index),
                height=735,
                scrolling=False,
            )

            st.subheader("Circuito térmico")
            components.html(
                thermal_circuit_component_html(result.config, snapshot),
                height=690,
                scrolling=False,
            )

            st.plotly_chart(axial_profiles(result, time_index), use_container_width=True)
            left, right = st.columns([1.1, 0.9])
            with left:
                st.plotly_chart(node_balance(snapshot, bool(result.config["model"]["has_glass"])), use_container_width=True)
            with right:
                st.subheader("Estado y derivadas del nodo")
                node_values = pd.DataFrame(
                    {
                        "Magnitud": [
                            "Tf",
                            "Tabs",
                            "Tvidrio",
                            "dTf/dt",
                            "dTabs/dt",
                            "dTvid/dt",
                            "Re",
                            "Pr",
                            "Nu",
                            "Peso transición",
                            "h interno",
                            "rho",
                            "mu",
                            "Cp",
                            "k",
                        ],
                        "Valor": [
                            snapshot["Tf_C"],
                            snapshot["Tabs_C"],
                            snapshot["Tglass_C"],
                            snapshot["dTf_dt_K_s"],
                            snapshot["dTabs_dt_K_s"],
                            snapshot["dTglass_dt_K_s"],
                            snapshot["Re_internal"],
                            snapshot["Pr_internal"],
                            snapshot["Nu_internal"],
                            snapshot["transition_weight"],
                            snapshot["h_internal_W_m2K"],
                            snapshot["rho_kg_m3"],
                            snapshot["mu_Pa_s"],
                            snapshot["Cp_J_kgK"],
                            snapshot["k_W_mK"],
                        ],
                        "Unidad": [
                            "°C",
                            "°C",
                            "°C",
                            "K/s",
                            "K/s",
                            "K/s",
                            "-",
                            "-",
                            "-",
                            "-",
                            "W/(m² K)",
                            "kg/m³",
                            "Pa·s",
                            "J/(kg K)",
                            "W/(m K)",
                        ],
                    }
                )
                st.dataframe(node_values, use_container_width=True, hide_index=True)
            node_table = result.node_dataframe(time_index)
            st.subheader("Todos los nodos en el instante seleccionado")
            st.dataframe(node_table, use_container_width=True, hide_index=True)
            st.download_button(
                "Descargar nodos del instante CSV",
                data=node_table.to_csv(index=False).encode("utf-8"),
                file_name=f"nodos_LAT_{snapshot['LAT_h']:.3f}.csv",
                mime="text/csv",
            )


    with sim_tab_report:
        report_text = build_technical_report(cfg)
        validation_result = st.session_state.validations.get("model_validation")
        validation_metrics = calibration_error_summary(validation_result)
        comparison = validation_comparison_table(validation_result)

        st.subheader("Reporte y exportación")
        if st.session_state.results:
            _, report_result = active_result_selector("report")
            if report_result is not None:
                kr, _ = representative_time_index(report_result)
                quick = st.columns(6)
                quick[0].metric("Tout", f"{report_result.Tout_C[kr]:.2f} °C")
                quick[1].metric("η HTF", f"{report_result.scalar_diag['eta_pct'][kr]:.2f} %")
                quick[2].metric("Q útil", f"{report_result.scalar_diag['Quseful_W'][kr]:.1f} W")
                quick[3].metric("Q pérdidas", f"{report_result.scalar_diag['Qloss_W'][kr]:.1f} W")
                quick[4].metric("Tabs", f"{report_result.Tabs_mean_C[kr]:.2f} °C")
                quick[5].metric("Tvid", f"{report_result.Tglass_mean_C[kr]:.2f} °C")

        st.markdown("#### Indicadores de error · última validación fuera de muestra")
        if validation_metrics:
            em = st.columns(6)
            em[0].metric("Score relativo", f"{validation_metrics.get('holdout_score_pct', float('nan')):.2f} %")
            em[1].metric("RMSE η", f"{validation_metrics.get('eta_rmse', float('nan')):.2f} pp")
            em[2].metric("MAE η", f"{validation_metrics.get('eta_mae', float('nan')):.2f} pp")
            em[3].metric("MAPE η", f"{validation_metrics.get('eta_mape_pct', float('nan')):.2f} %")
            em[4].metric("RMSE Tout", f"{validation_metrics.get('tout_rmse', float('nan')):.2f} °C")
            em[5].metric("Máx. error rel. η", f"{validation_metrics.get('eta_max_rel_pct', float('nan')):.2f} %")
        else:
            st.info("Ejecute la sección Validación para incorporar RMSE, MAE, MAPE y errores relativos al reporte.")

        report_lines = [report_text]
        if validation_metrics:
            report_lines.extend([
                "",
                "=== VALIDACION FUERA DE MUESTRA ===",
                f"Caso: {validation_result.get('case', '—')}",
                f"Score relativo hold-out: {validation_metrics.get('holdout_score_pct', float('nan')):.4f} %",
                f"RMSE eta: {validation_metrics.get('eta_rmse', float('nan')):.4f} pp",
                f"MAE eta: {validation_metrics.get('eta_mae', float('nan')):.4f} pp",
                f"MAPE eta: {validation_metrics.get('eta_mape_pct', float('nan')):.4f} %",
                f"Bias eta: {validation_metrics.get('eta_bias', float('nan')):.4f} pp",
                f"RMSE Tout: {validation_metrics.get('tout_rmse', float('nan')):.4f} °C",
                f"MAPE Tout: {validation_metrics.get('tout_mape_pct', float('nan')):.4f} %",
            ])
        report_text_with_validation = "\n".join(report_lines)
        with st.expander("Ver reporte técnico", expanded=False):
            st.code(report_text_with_validation, language="text")

        project_payload = {"config": json_ready(cfg), "fluid_database": json_ready(fluid_db)}
        project_json = json.dumps(project_payload, indent=2, ensure_ascii=False).encode("utf-8")
        export_cols = st.columns(3)
        export_cols[0].download_button(
            "Descargar reporte TXT",
            data=report_text_with_validation.encode("utf-8"),
            file_name="reporte_tecnico_ptc.txt",
            mime="text/plain",
            use_container_width=True,
        )
        export_cols[1].download_button(
            "Guardar proyecto JSON",
            data=project_json,
            file_name="proyecto_ptc.json",
            mime="application/json",
            use_container_width=True,
        )
        if st.session_state.results:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                for index, (scenario, scenario_result) in enumerate(st.session_state.results.items(), start=1):
                    scenario_result.scalar_dataframe().to_excel(writer, sheet_name=f"Escenario_{index}", index=False)
                    scenario_result.node_dataframe(len(scenario_result.t_s) - 1).to_excel(writer, sheet_name=f"Nodos_final_{index}", index=False)
                if validation_metrics:
                    pd.DataFrame([validation_metrics]).to_excel(writer, sheet_name="Errores_validacion", index=False)
                    if not comparison.empty:
                        comparison.to_excel(writer, sheet_name="Detalle_validacion", index=False)
                    if isinstance(validation_result, dict) and isinstance(validation_result.get("parameter_table"), pd.DataFrame):
                        validation_result["parameter_table"].to_excel(writer, sheet_name="Parametros_calibrados", index=False)
            export_cols[2].download_button(
                "Exportar resultados XLSX",
                data=buffer.getvalue(),
                file_name="resultados_ptc.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            export_cols[2].info("Ejecute una simulación para habilitar XLSX.")

elif main_section == "Propiedades":
    st.subheader("Irradiación diaria del modelo")
    st.caption(
        "Diagnóstico de 00:00 a 24:00 LAT calculado directamente con el modelo de irradiación seleccionado. "
        "Las horas de sol son las horas en que el modelo entrega DNI > 1 W/m²."
    )
    try:
        solar_day = daily_solar_diagnostics(cfg, fluid_db)
        solar_cols = st.columns(4)
        solar_cols[0].metric("Horas de sol", f"{solar_day['sun_hours']:.2f} h")
        solar_cols[1].metric("DNI máximo", f"{solar_day['DNI_max_W_m2']:.1f} W/m²")
        solar_cols[2].metric("DNI diario", f"{solar_day['DNI_daily_kWh_m2']:.2f} kWh/m²")
        if solar_day["sunrise_h"] is None:
            solar_window = "Sin irradiación"
        else:
            solar_window = f"{_format_lat_hour(solar_day['sunrise_h'])} – {_format_lat_hour(solar_day['sunset_h'])}"
        solar_cols[3].metric("Ventana solar LAT", solar_window)
        st.plotly_chart(daily_irradiance_histogram(solar_day["hourly"]), use_container_width=True)
        st.caption(
            f"Energía diaria proyectada sobre la apertura antes de pérdidas ópticas: "
            f"{solar_day['beam_daily_kWh_m2']:.2f} kWh/m². "
            "La barra es el DNI medio de cada hora y la línea representa DNI·cos(theta)."
        )
        if str(cfg["solar"]["mode"]).lower() == "constante":
            st.info(
                "En modo DNI y ángulo constantes no existe amanecer/ocaso dentro de la formulación: "
                "el valor constante se aplica a cualquier hora evaluada. Por eso, si DNI > 1 W/m², "
                "el diagnóstico diario muestra 24 h de sol. Esto permite identificar claramente la "
                "diferencia entre una hipótesis constante y un modelo solar horario."
            )
    except Exception as exc:
        st.error(f"No fue posible construir el diagnóstico diario de irradiación: {exc}")

    st.divider()
    st.subheader("Temperatura efectiva del cielo")
    try:
        t0 = float(cfg["operation"]["t_start_s"])
        t1 = float(cfg["operation"]["t_end_s"])
        tm = 0.5 * (t0 + t1)
        sky_start = effective_sky_temperature(t0, cfg["environment"])
        sky_mid = effective_sky_temperature(tm, cfg["environment"])
        sky_end = effective_sky_temperature(t1, cfg["environment"])
        sky_cols = st.columns(4)
        sky_cols[0].metric("T cielo · inicio", f"{sky_start['Tsky_K'] - 273.15:.2f} °C")
        sky_cols[1].metric("T cielo · medio", f"{sky_mid['Tsky_K'] - 273.15:.2f} °C")
        sky_cols[2].metric("T cielo · final", f"{sky_end['Tsky_K'] - 273.15:.2f} °C")
        sky_cols[3].metric("ε cielo · medio", f"{sky_mid['eps_sky']:.4f}")
        if str(cfg["environment"].get("sky_model", "delta_constante")) == "rea_quille":
            st.caption(
                "Modelo de Rea Quille / Martin-Berdahl: ε0 depende de Tdp, hora y presión; "
                "Tsky = ε_sky^0.25·Tamb. La corrección por nubosidad solo se aplica si está activada en Operación y ambiente."
            )
        else:
            st.caption("Modo legado: Tsky = Tamb - ΔT constante.")
    except Exception as exc:
        st.error(f"No fue posible calcular la temperatura efectiva del cielo: {exc}")

    st.divider()
    st.subheader("Propiedades del fluido")
    fluid_key = st.selectbox("Fluido a editar", list(fluid_db.keys()), key="fluid_editor")
    spec = fluid_db[fluid_key]
    available_modes = ["original", "table", "constant"] if fluid_key in {"ParathermNF", "Agua"} else ["table", "constant"]
    if spec.get("mode") not in available_modes:
        spec["mode"] = available_modes[0]
    spec["mode"] = st.selectbox(
        "Modelo de propiedades",
        available_modes,
        index=available_modes.index(spec["mode"]),
        format_func=lambda value: {
            "original": "Correlación original del modelo",
            "table": "Tabla completa editable + PCHIP",
            "constant": "Propiedades constantes",
        }[value],
    )
    if spec["mode"] == "table" and not spec.get("table"):
        ensure_constant_properties(fluid_key)
        constants = spec["constants"]
        spec["table"] = {
            "T_C": [20.0, 100.0],
            "rho_kg_m3": [constants["rho_kg_m3"], constants["rho_kg_m3"]],
            "mu_Pa_s": [constants["mu_Pa_s"], constants["mu_Pa_s"]],
            "Cp_J_kgK": [constants["Cp_J_kgK"], constants["Cp_J_kgK"]],
            "k_W_mK": [constants["k_W_mK"], constants["k_W_mK"]],
        }
    st.caption(
        "Los multiplicadores se aplican después de la correlación o interpolación. "
        "Permiten calibrar cada propiedad sin eliminar su dependencia térmica."
    )
    multiplier_cols = st.columns(4)
    for col, key, label in zip(multiplier_cols, ["rho", "mu", "Cp", "k"], ["rho", "mu", "Cp", "k"], strict=True):
        spec["multipliers"][key] = col.number_input(
            f"Factor {label}", min_value=0.001, max_value=1000.0, value=float(spec["multipliers"][key]), step=0.01, key=f"mult_{fluid_key}_{key}"
        )
    if fluid_key == "ParathermNF" and spec["mode"] in {"original", "table"}:
        cp_cols = st.columns(2)
        spec["cp_a"] = cp_cols[0].number_input("Coeficiente a de Cp = a*T_K + b", value=float(spec.get("cp_a", 3.6161)), format="%.6f")
        spec["cp_b"] = cp_cols[1].number_input("Coeficiente b de Cp = a*T_K + b", value=float(spec.get("cp_b", 814.37)), format="%.6f")
    if spec["mode"] == "constant":
        ensure_constant_properties(fluid_key)
        constants = spec["constants"]
        constant_cols = st.columns(4)
        constants["rho_kg_m3"] = constant_cols[0].number_input("rho (kg/m³)", min_value=0.001, value=float(constants["rho_kg_m3"]), key=f"const_rho_{fluid_key}")
        constants["mu_Pa_s"] = constant_cols[1].number_input("mu (Pa·s)", min_value=1e-9, value=float(constants["mu_Pa_s"]), format="%.8f", key=f"const_mu_{fluid_key}")
        constants["Cp_J_kgK"] = constant_cols[2].number_input("Cp (J/kg K)", min_value=0.001, value=float(constants["Cp_J_kgK"]), key=f"const_cp_{fluid_key}")
        constants["k_W_mK"] = constant_cols[3].number_input("k (W/m K)", min_value=1e-6, value=float(constants["k_W_mK"]), format="%.6f", key=f"const_k_{fluid_key}")
    else:
        table = pd.DataFrame(spec.get("table", {}))
        edited_table = st.data_editor(
            table,
            num_rows="dynamic",
            use_container_width=True,
            key=f"property_table_{fluid_key}",
        )
        spec["table"] = {column: edited_table[column].tolist() for column in edited_table.columns}
        if fluid_key == "Agua" and spec["mode"] == "original":
            st.info("En modo original, la tabla de agua es solo una vista de referencia. Para usar los valores editados, seleccione Tabla completa editable + PCHIP.")
    try:
        curve = property_curve(fluid_key, fluid_db, -10.0, 320.0, 250)
        st.plotly_chart(property_figure(curve, spec.get("display_name", fluid_key)), use_container_width=True)
    except Exception as exc:
        st.error(f"No fue posible evaluar las propiedades: {exc}")

    st.divider()
    st.subheader("Perfil horario de irradiación")
    profile_df = pd.DataFrame(cfg["solar"]["profile"])
    profile_edited = st.data_editor(profile_df, num_rows="dynamic", use_container_width=True, key="solar_profile_editor")
    cfg["solar"]["profile"] = {column: profile_edited[column].tolist() for column in profile_edited.columns}
    st.caption("Este perfil se usa cuando el modo de irradiación es Perfil horario editable.")


elif main_section == "Validación":
    st.subheader("Validación del modelo")
    st.write(
        "La validación se separa de la calibración. Primero se identifican parámetros usando solamente una muestra documental; "
        "después esos parámetros quedan congelados y se prueban contra datos que no participaron del ajuste."
    )

    workflow_cols = st.columns([1.45, 1.0, 1.0])
    validation_case_labels = {
        "Rea Quille · Foz do Iguaçu · 12 meses": "rea_foz",
        "Rea Quille · Alvorada do Norte · 12 meses": "rea_alvorada",
    }
    validation_case_label = workflow_cols[0].selectbox(
        "Conjunto para calibración + validación",
        list(validation_case_labels.keys()),
        key="validation_case_selector_v14",
    )
    validation_case = validation_case_labels[validation_case_label]
    workflow_cols[1].metric("Muestra de calibración", "4 meses")
    workflow_cols[2].metric("Hold-out", "8 meses")
    st.caption(
        "Calibración: enero, abril, julio y octubre. Validación fuera de muestra: los ocho meses restantes. "
        "Los meses de hold-out nunca entran en la función objetivo del optimizador."
    )

    registry = inverse_parameter_options(validation_case, fluid_db)
    label_to_id = {spec["label"]: pid for pid, spec in registry.items()}
    default_ids = ["eta_opt_eff", "wind_m_s"]
    default_labels = [registry[pid]["label"] for pid in default_ids if pid in registry]
    selected_labels = st.multiselect(
        "Parámetros a calibrar",
        list(label_to_id.keys()),
        default=default_labels,
        key=f"validation_parameters_{validation_case}",
        help="Seleccione solamente parámetros inciertos. Los valores publicados deberían permanecer fijos.",
    )
    selected_ids = [label_to_id[label] for label in selected_labels]

    if selected_ids:
        preview = pd.DataFrame([
            {
                "Parámetro": registry[pid]["label"],
                "Nominal": registry[pid]["nominal"],
                "Límite inferior": registry[pid]["bounds"][0],
                "Límite superior": registry[pid]["bounds"][1],
                "Estado": registry[pid]["status"],
            }
            for pid in selected_ids
        ])
        with st.expander("Parámetros que entrarán en la calibración", expanded=False):
            st.dataframe(preview, use_container_width=True, hide_index=True)

    controls = st.columns([1.0, 1.35])
    max_nfev = controls[0].slider(
        "Máx. evaluaciones",
        min_value=8,
        max_value=60,
        value=18,
        step=2,
        key=f"validation_nfev_{validation_case}",
    )
    run_validation = controls[1].button(
        "Calibrar con 4 meses y validar en 8 no usados",
        type="primary",
        use_container_width=True,
        disabled=not selected_ids,
        key="run_calibration_holdout_v14",
    )
    if run_validation:
        try:
            with st.spinner("Calibrando la muestra y reejecutando automáticamente los ocho meses de hold-out..."):
                st.session_state.validations["model_validation"] = calibrate_inverse_model(
                    validation_case,
                    fluid_db,
                    selected_ids,
                    monthly_strategy="alternating",
                    max_nfev=max_nfev,
                )
        except Exception as exc:
            st.exception(exc)

    validation_result = st.session_state.validations.get("model_validation")
    if isinstance(validation_result, dict) and validation_result.get("case") == validation_case:
        st.divider()
        st.markdown("#### Resultado: calibración → validación fuera de muestra")
        summary = calibration_error_summary(validation_result)
        top = st.columns(6)
        top[0].metric("Score muestra · antes", f"{validation_result['score_before_pct']:.2f} %")
        top[1].metric("Score muestra · calibrado", f"{validation_result['score_after_pct']:.2f} %")
        top[2].metric("Score hold-out", f"{summary.get('holdout_score_pct', float('nan')):.2f} %")
        top[3].metric("RMSE η hold-out", f"{summary.get('eta_rmse', float('nan')):.2f} pp")
        top[4].metric("MAPE η hold-out", f"{summary.get('eta_mape_pct', float('nan')):.2f} %")
        top[5].metric("RMSE Tout hold-out", f"{summary.get('tout_rmse', float('nan')):.2f} °C")

        if validation_result.get("success") and validation_result.get("physically_admissible"):
            st.success("La calibración terminó dentro de los límites físicos impuestos; los indicadores anteriores corresponden a datos no usados en el ajuste.")
        else:
            st.warning(f"La calibración terminó con advertencias: {validation_result.get('message', '—')}")
        if not validation_result.get("locally_identifiable", True):
            st.warning("El Jacobiano indica identificabilidad débil: varios conjuntos de parámetros pueden producir respuestas parecidas.")

        comparison = validation_comparison_table(validation_result)
        holdout_table = comparison.loc[comparison["Conjunto"] == "validación"].copy() if not comparison.empty else pd.DataFrame()
        if not holdout_table.empty:
            st.markdown("**Datos no usados durante la calibración**")
            st.dataframe(holdout_table, use_container_width=True, hide_index=True)
            eta_hold = holdout_table.loc[holdout_table["Magnitud"] == "Eta_pct"].copy()
            if not eta_hold.empty:
                improved = eta_hold.loc[eta_hold["Error_calibrado_pct"].abs() < eta_hold["Error_inicial_pct"].abs()]
                worsened = eta_hold.loc[eta_hold["Error_calibrado_pct"].abs() > eta_hold["Error_inicial_pct"].abs()]
                g = st.columns(3)
                g[0].metric("Meses mejorados", f"{len(improved)}/{len(eta_hold)}")
                g[1].metric("Meses que empeoran", f"{len(worsened)}/{len(eta_hold)}")
                g[2].metric("Generalización", "Consistente" if len(worsened) == 0 else "Mixta")
                if len(worsened):
                    details = ", ".join(
                        f"{row.Caso}: {abs(row.Error_inicial_pct):.2f}% → {abs(row.Error_calibrado_pct):.2f}%"
                        for row in worsened.itertuples()
                    )
                    st.warning(
                        "El error global puede disminuir aunque algunos meses empeoren, porque el optimizador minimiza una función conjunta. "
                        f"Meses con pérdida de precisión: {details}."
                    )

        st.markdown("**Parámetros calibrados**")
        st.dataframe(validation_result["parameter_table"], use_container_width=True, hide_index=True)
        action = st.columns(2)
        if action[0].button(
            "Usar parámetros calibrados en el simulador",
            type="primary",
            use_container_width=True,
            key="apply_latest_calibration_v14",
        ):
            try:
                apply_calibrated_parameters(st.session_state.config, st.session_state.fluid_database, validation_result)
                st.session_state.results = {}
                st.session_state.result_signature = None
                st.session_state.ui_revision += 1
                st.success("Parámetros calibrados aplicados al proyecto activo.")
                st.rerun()
            except Exception as exc:
                st.exception(exc)
        export_validation = io.BytesIO()
        with pd.ExcelWriter(export_validation, engine="openpyxl") as writer:
            validation_result["parameter_table"].to_excel(writer, sheet_name="Parametros_calibrados", index=False)
            comparison.to_excel(writer, sheet_name="Calibracion_y_holdout", index=False)
            pd.DataFrame([summary]).to_excel(writer, sheet_name="Indicadores_holdout", index=False)
        action[1].download_button(
            "Exportar calibración + validación",
            data=export_validation.getvalue(),
            file_name=f"validacion_holdout_{validation_case}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.divider()
    with st.expander("Benchmarks documentales · diagnóstico adicional", expanded=False):
        st.caption(
            "Estos benchmarks sirven para auditar el modelo contra literatura, pero no sustituyen la validación fuera de muestra. "
            "Bhambare/Sukhatme es un único caso; Fiamonzini Tabela 8 carece de varias entradas horarias experimentales."
        )
        bench_cols = st.columns(2)
        if bench_cols[0].button("Bhambare vs Sukhatme", use_container_width=True, key="compact_bhambare_v14"):
            try:
                st.session_state.validations["compact_bhambare"] = validate_bhambare_mode(
                    fluid_db, target_key="sukhatme", parameter_template="nominal"
                )
            except Exception as exc:
                st.exception(exc)
        if bench_cols[1].button("Rea/Fiamonzini · Tabela 8", use_container_width=True, key="compact_rea_proto_v14"):
            try:
                st.session_state.validations["compact_prototype"] = validate_rea_prototype_mode(
                    "trnsys_published", fluid_db, target_key="experimental", dni_source="nominal", parameter_template="nominal"
                )
            except Exception as exc:
                st.exception(exc)
        if "compact_bhambare" in st.session_state.validations:
            bv = st.session_state.validations["compact_bhambare"]
            bm = bv["metrics"]
            c = st.columns(4)
            c[0].metric("RMSRE", f"{bm['RMSRE_pct']:.2f} %")
            c[1].metric("MAPE multivariable", f"{bm['MAPE_multivariable_pct']:.2f} %")
            c[2].metric("Bias relativo", f"{bm['Bias_rel_medio_pct']:+.2f} %")
            c[3].metric("Error máximo", f"{bm['Error_max_pct']:.2f} %")
            st.dataframe(bv["table"], use_container_width=True, hide_index=True)
        if "compact_prototype" in st.session_state.validations:
            pv = st.session_state.validations["compact_prototype"]
            pm = pv["metrics"]
            c = st.columns(4)
            c[0].metric("MAE η", f"{pm['MAE_pp']:.2f} pp")
            c[1].metric("RMSE η", f"{pm['RMSE_pp']:.2f} pp")
            c[2].metric("MAPE η", f"{pm['MAPE_pct']:.2f} %")
            c[3].metric("η Python media", f"{pm['Eta_python_mean_pct']:.2f} %")
            st.dataframe(pv["table"], use_container_width=True, hide_index=True)

elif main_section == "Sensibilidad":
    st.subheader("Sensibilidad y convergencia")
    st.write(
        "El análisis de sensibilidad cuantifica cuánto cambian los resultados buscados cuando cambia un parámetro. "
        "No calibra el modelo: perturba cada parámetro de forma controlada para identificar cuáles dominan Tout, Tabs, Tvid, Q pérdidas y eficiencia. "
        "La convergencia numérica, por separado, verifica que la respuesta no dependa artificialmente de la malla, del paso temporal o de las tolerancias."
    )

    sens_cols = st.columns(2)
    if sens_cols[0].button(
        "1 · Ejecutar convergencia numérica",
        type="primary",
        use_container_width=True,
        help="Barre nodos, max_step, tolerancias y tiempo de calentamiento.",
    ):
        try:
            with st.spinner("Analizando independencia de malla, paso temporal, tolerancias y estado estacionario..."):
                st.session_state.validations["numerical_sensitivity"] = analyze_bhambare_numerical_convergence(fluid_db)
        except Exception as exc:
            st.exception(exc)

    if sens_cols[1].button(
        "2 · Ejecutar sensibilidad física ±10 %",
        use_container_width=True,
        help="Perturba un parámetro por vez y mide cuánto cambia el ajuste a Sukhatme.",
    ):
        try:
            with st.spinner("Perturbando parámetros físicos uno por uno. Esta prueba puede tardar alrededor de medio minuto..."):
                st.session_state.validations["physical_sensitivity"] = analyze_bhambare_physical_sensitivity(fluid_db, perturbation=0.10)
        except Exception as exc:
            st.exception(exc)

    if "numerical_sensitivity" in st.session_state.validations:
        analysis = st.session_state.validations["numerical_sensitivity"]
        st.divider()
        st.subheader("1 · Convergencia numérica")
        st.caption(analysis["note"])
        summary = analysis["summary"]
        c = st.columns(4)
        c[0].metric("N para <0.5 %", str(summary["N_0p5pct"] or "—"))
        c[1].metric("N para <0.1 %", str(summary["N_0p1pct"] or "—"))
        c[2].metric("Span Tout por malla", f"{summary['mesh_span_Tout_C']:.5f} °C")
        c[3].metric("Span Qloss por malla", f"{summary['mesh_span_Qloss_W']:.2f} W")
        st.plotly_chart(numerical_convergence_figure(analysis["mesh_table"]), use_container_width=True)

        group = st.selectbox(
            "Detalle numérico",
            ["Nodos", "max_step_s", "rtol", "Duracion_h"],
            key="sensitivity_numeric_group",
        )
        detail = analysis["table"].loc[analysis["table"]["Grupo"] == group].copy()
        st.dataframe(detail, use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar convergencia numérica · CSV",
            data=analysis["table"].to_csv(index=False).encode("utf-8"),
            file_name="ptc_convergencia_numerica_bhambare.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if summary["N_0p5pct"] is not None and summary["N_0p5pct"] <= 12:
            st.success(
                "La solución alcanza independencia de malla antes o en N=12 con el criterio de 0.5 %. "
                "Si el error documental persiste, aumentar el número de nodos no es la corrección principal."
            )

    if "physical_sensitivity" in st.session_state.validations:
        analysis = st.session_state.validations["physical_sensitivity"]
        st.divider()
        st.subheader("2 · Sensibilidad física local")
        st.caption(analysis["note"])
        baseline = analysis["baseline"]
        reference = analysis["reference"]
        c = st.columns(5)
        c[0].metric("Score base", f"{analysis['baseline_score_pct']:.2f} %")
        c[1].metric("Tout Python / ref", f"{baseline['Tout_C']:.2f} / {reference['Tout_C']:.2f} °C")
        c[2].metric("Tabs Python / ref", f"{baseline['Tabs_K']:.2f} / {reference['Tabs_K']:.2f} K")
        c[3].metric("Tvid Python / ref", f"{baseline['Tvid_K']:.2f} / {reference['Tvid_K']:.2f} K")
        c[4].metric("Qloss Python / ref", f"{baseline['Qloss_W']:.1f} / {reference['Qloss_W']:.1f} W")

        st.plotly_chart(sensitivity_tornado_figure(analysis["table"]), use_container_width=True)
        display_cols = [
            "Parametro", "Categoria", "Nominal", "Valor_menos", "Valor_mas",
            "Mejor_direccion", "Mejora_score_pp", "Mejor_score_pct",
            "S_Tout_C", "S_Tabs_K", "S_Tvid_K", "S_Qloss_W", "S_eta_pct",
        ]
        st.dataframe(analysis["table"][display_cols], use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar sensibilidad física · CSV",
            data=analysis["table"].to_csv(index=False).encode("utf-8"),
            file_name="ptc_sensibilidad_fisica_bhambare.csv",
            mime="text/csv",
            use_container_width=True,
        )

        top = analysis["table"].iloc[0]
        st.info(
            f"Mayor capacidad local de reducir la discrepancia: {top['Parametro']} "
            f"({top['Mejor_direccion']}10 %), con una reducción del score de "
            f"{top['Mejora_score_pp']:.2f} puntos. Esto identifica sensibilidad, no autoriza calibrar el parámetro fuera de su valor físico/documental."
        )

