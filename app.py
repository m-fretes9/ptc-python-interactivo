"""Interfaz Streamlit para el modelo nodal del colector PTC."""

from __future__ import annotations

import io
import json
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from streamlit_plotly_events import plotly_events

from defaults import default_config, default_fluid_database
from interactive_visuals import ptc_optical_component_html, thermal_circuit_component_html
from presets import MONTH_NAMES_ES, PRESET_FAMILY_LABELS, build_preset, preset_summary_rows
from fluid_properties import FluidPropertyEvaluator, property_curve
from ptc_model import PTCSimulator, SimulationResult, effective_sky_temperature
from technical_report import build_technical_report, result_summary
from validations import (
    prototype_tcc_table,
    validate_active_preset,
    validate_bhambare,
    validate_rea_quille_city_monthly,
    validate_tcc_monthly,
)
from visualizations import (
    axial_profiles,
    comparative_overview,
    daily_irradiance_histogram,
    dynamic_overview,
    node_balance,
    node_flow_selector_figure,
    property_figure,
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
    if "ray_seed" not in st.session_state:
        st.session_state.ray_seed = 0


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


def apply_reference_preset(family: str, month: int | None = None, annual: bool = False) -> None:
    new_cfg, new_fluid_db = build_preset(family, month=month, annual=annual)
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


initialize_state()
cfg = st.session_state.config
fluid_db = st.session_state.fluid_database

st.title("Modelo nodal de colector cilindro-parabólico")
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
    if st.button("Aplicar preset completo", type="primary", use_container_width=True):
        apply_reference_preset(preset_family, month=preset_month, annual=preset_annual)
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
            "Parishwad": "Parishwad / cielo claro",
            "constante": "DNI y ángulo constantes",
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
        solver["rtol"] = st.number_input("rtol", min_value=1e-12, max_value=1e-2, value=float(solver["rtol"]), format="%.1e")
        solver["atol"] = st.number_input("atol", min_value=1e-12, max_value=1e-2, value=float(solver["atol"]), format="%.1e")
        solver["max_step_s"] = st.number_input("Paso máximo BDF (s)", min_value=0.1, value=float(solver["max_step_s"]), step=5.0)
        solver["use_jac_sparsity"] = st.checkbox("Usar patrón disperso de Jacobiano", value=bool(solver["use_jac_sparsity"]))

    st.divider()
    sweep = st.checkbox("Barrido comparativo de caudales", value=False)
    sweep_text = st.text_input("Caudales (kg/s), separados por coma", value="0.015, 0.045, 0.090", disabled=not sweep)
    run_clicked = st.button("Ejecutar simulación", type="primary", use_container_width=True)

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

tab_sim, tab_nodes, tab_props, tab_validation, tab_report = st.tabs(
    ["Simulación", "Nodo por nodo", "Propiedades e irradiación", "Validación", "Reporte y exportación"]
)

with tab_sim:
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
        diag_cols = st.columns(4)
        diag_cols[0].metric("η óptica al absorbedor", f"{result.scalar_diag['eta_optical_abs_pct'][k_ref]:.2f} %")
        diag_cols[1].metric("η integrada del período", f"{eta_period:.2f} %")
        diag_cols[2].metric("Re salida", f"{re_out:.0f}")
        diag_cols[3].metric("Régimen salida", regime)
        result_meta = result.config.get("preset_meta", {})
        source_ref = result_meta.get("reference", {})
        if source_ref:
            with st.expander("Comparación rápida con la referencia del preset", expanded=True):
                ref_cols = st.columns(4)
                ref_cols[0].metric("Fuente", str(result_meta.get("reference_table", "—")))
                if "eta_pct" in source_ref:
                    eta_ref = float(source_ref["eta_pct"])
                    eta_py = float(result.scalar_diag["eta_pct"][k_ref])
                    ref_cols[1].metric("η referencia", f"{eta_ref:.2f} %")
                    ref_cols[2].metric("η Python", f"{eta_py:.2f} %", delta=f"{eta_py - eta_ref:+.2f} pp")
                    if "Tout_C" in source_ref:
                        ref_cols[3].metric("Tout ref / Python", f"{float(source_ref['Tout_C']):.2f} / {result.Tout_C[k_ref]:.2f} °C")
                elif "eta_exp_mean_pct" in source_ref:
                    eta_py = float(result.scalar_diag["eta_pct"][k_ref])
                    ref_cols[1].metric("η exp. media", f"{float(source_ref['eta_exp_mean_pct']):.2f} %")
                    ref_cols[2].metric("η TRNSYS media", f"{float(source_ref['eta_trnsys_mean_pct']):.2f} %")
                    ref_cols[3].metric("η Python", f"{eta_py:.2f} %")
                    st.warning("Tabela 8 no publica Tin/Tout horarios; la comparación Python depende de la Tin asumida en el preset y no es una validación estricta.")
                elif "Tout_book_C" in source_ref:
                    ref_cols[1].metric("Tout Sukhatme", f"{float(source_ref['Tout_book_C']):.2f} °C")
                    ref_cols[2].metric("Tout Bhambare", f"{float(source_ref['Tout_article_C']):.2f} °C")
                    ref_cols[3].metric("Tout Python", f"{result.Tout_C[k_ref]:.2f} °C")

        if len(st.session_state.results) > 1:
            st.plotly_chart(comparative_overview(st.session_state.results), use_container_width=True)
        else:
            st.plotly_chart(dynamic_overview(result), use_container_width=True)
        with st.expander("Resumen del solver y del escenario"):
            st.code(result_summary(result), language="text")
        scalar_csv = result.scalar_dataframe().to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar serie temporal CSV",
            data=scalar_csv,
            file_name="resultado_ptc_temporal.csv",
            mime="text/csv",
        )

with tab_nodes:
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
        selector_fig = node_flow_selector_figure(result, time_index, st.session_state[state_key] - 1)
        clicked = plotly_events(
            selector_fig,
            click_event=True,
            hover_event=False,
            select_event=False,
            key=f"node_selector_plot_{label}_{time_index}_{result.n_segments}",
            override_height=205,
            override_width="100%",
        )
        if clicked:
            candidate = int(clicked[0].get("pointNumber", 0)) + 1
            candidate = int(np.clip(candidate, 1, result.n_segments))
            if candidate != st.session_state[state_key]:
                st.session_state[state_key] = candidate
                st.rerun()

        node_number = int(st.session_state[state_key])
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

with tab_props:
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

with tab_validation:
    st.subheader("Validación documental")
    st.write(
        "Los presets separan los valores publicados de las hipótesis que nuestra formulación necesita y la fuente no informa. "
        "La comparación del preset activo utiliza exactamente la configuración que está viendo/editando en el sidebar."
    )

    meta = cfg.get("preset_meta", {})
    if meta:
        source_cols = st.columns(4)
        source_cols[0].metric("Preset activo", str(meta.get("family", "—")))
        source_cols[1].metric("Ciudad", str(meta.get("city", "—")))
        source_cols[2].metric("Referencia", str(meta.get("reference_table", "—")))
        source_cols[3].metric("Caso", str(meta.get("date_label", "—")))
        if meta.get("assumptions"):
            with st.expander("Supuestos del preset que afectan la comparación", expanded=False):
                for item in meta["assumptions"]:
                    st.markdown(f"- {item}")

    val_cols = st.columns(4)
    if val_cols[0].button("Validar preset activo", type="primary", use_container_width=True):
        try:
            with st.spinner("Ejecutando el preset activo contra su referencia"):
                st.session_state.validations["active_preset"] = validate_active_preset(cfg, fluid_db)
        except Exception as exc:
            st.exception(exc)
    if val_cols[1].button("12 meses · Foz", use_container_width=True):
        try:
            with st.spinner("Ejecutando los 12 presets mensuales de Foz do Iguaçu"):
                st.session_state.validations["rea_foz"] = validate_rea_quille_city_monthly("Foz do Iguaçu", fluid_db)
        except Exception as exc:
            st.exception(exc)
    if val_cols[2].button("12 meses · Alvorada", use_container_width=True):
        try:
            with st.spinner("Ejecutando los 12 presets mensuales de Alvorada do Norte"):
                st.session_state.validations["rea_alvorada"] = validate_rea_quille_city_monthly("Alvorada do Norte", fluid_db)
        except Exception as exc:
            st.exception(exc)
    if val_cols[3].button("Tabla 8 · Experimental/TRNSYS", use_container_width=True):
        st.session_state.validations["prototype_table"] = prototype_tcc_table()

    if "active_preset" in st.session_state.validations:
        validation = st.session_state.validations["active_preset"]
        st.divider()
        st.subheader("Preset activo vs documento")
        st.caption(validation["note"])
        kind = validation.get("kind")
        if kind == "prototype":
            metrics = validation["metrics"]
            mcols = st.columns(4)
            mcols[0].metric("η exp. media", f"{metrics['Eta_exp_media_pct']:.2f} %")
            mcols[1].metric("η TRNSYS media", f"{metrics['Eta_TRNSYS_media_pct']:.2f} %")
            mcols[2].metric("η Python media", f"{metrics['Eta_Python_media_pct']:.2f} %")
            mcols[3].metric("Bias Python-exp", f"{metrics['Bias_Python_vs_exp_pp']:+.2f} pp")
            st.warning("La comparación Python del prototipo es exploratoria porque la Tabela 8 no publica Tin/Tout horarios. El valor de Tin mostrado en el preset es una hipótesis explícita y editable.")
            st.dataframe(validation["table"], use_container_width=True, hide_index=True)
        elif kind == "bhambare":
            st.dataframe(validation["table"], use_container_width=True, hide_index=True)
            st.plotly_chart(validation_bhambare_figure(validation["table"]), use_container_width=True)
        else:
            st.dataframe(validation["table"], use_container_width=True, hide_index=True)

    for key, title in (("rea_foz", "Rea Quille — Foz do Iguaçu — Tabela 10"), ("rea_alvorada", "Rea Quille — Alvorada do Norte — Tabela 11")):
        if key in st.session_state.validations:
            validation = st.session_state.validations[key]
            st.divider()
            st.subheader(title)
            metrics = validation["metrics"]
            metric_cols = st.columns(4)
            metric_cols[0].metric("MAPE Tout", f"{metrics['MAPE_Tout_pct']:.2f} %")
            metric_cols[1].metric("MAPE Q útil", f"{metrics['MAPE_Qutil_pct']:.2f} %")
            metric_cols[2].metric("MAPE η", f"{metrics['MAPE_eta_pct']:.2f} %")
            metric_cols[3].metric("Bias η", f"{metrics['Bias_eta_pp']:+.2f} pp")
            st.caption(validation["note"])
            st.dataframe(validation["table"], use_container_width=True, hide_index=True)
            st.plotly_chart(validation_tcc_figure(validation["table"]), use_container_width=True)

    if "prototype_table" in st.session_state.validations:
        validation = st.session_state.validations["prototype_table"]
        st.divider()
        st.subheader("Rea Quille / Fiamonzini — Tabela 8")
        prototype_cols = st.columns(4)
        prototype_cols[0].metric("η experimental media", f"{validation['eta_exp_mean_pct']:.2f} %")
        prototype_cols[1].metric("η TRNSYS media", f"{validation['eta_trnsys_mean_pct']:.2f} %")
        prototype_cols[2].metric("RMSE TRNSYS-exp", f"{validation['RMSE_pp']:.2f} pp")
        prototype_cols[3].metric("MAPE TRNSYS-exp", f"{validation['MAPE_pct']:.2f} %")
        st.caption(validation["note"])
        st.dataframe(validation["table"], use_container_width=True, hide_index=True)

with tab_report:
    report_text = build_technical_report(cfg)
    st.subheader("Reporte técnico en texto plano")
    st.code(report_text, language="text")
    project_payload = {
        "config": json_ready(cfg),
        "fluid_database": json_ready(fluid_db),
    }
    project_json = json.dumps(project_payload, indent=2, ensure_ascii=False).encode("utf-8")
    export_cols = st.columns(3)
    export_cols[0].download_button(
        "Descargar reporte TXT",
        data=report_text.encode("utf-8"),
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
                scenario_result.node_dataframe(len(scenario_result.t_s) - 1).to_excel(
                    writer, sheet_name=f"Nodos_final_{index}", index=False
                )
        export_cols[2].download_button(
            "Exportar resultados XLSX",
            data=buffer.getvalue(),
            file_name="resultados_ptc.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    else:
        export_cols[2].info("Ejecute una simulación para habilitar XLSX.")
