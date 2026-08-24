"""Interfaz Streamlit para el modelo nodal del colector PTC."""

from __future__ import annotations

import io
import json
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from defaults import default_config, default_fluid_database
from fluid_properties import FluidPropertyEvaluator, property_curve
from ptc_model import PTCSimulator, SimulationResult
from technical_report import build_technical_report, result_summary
from validations import prototype_tcc_table, validate_bhambare, validate_tcc_monthly
from visualizations import (
    axial_profiles,
    comparative_overview,
    dynamic_overview,
    node_balance,
    property_figure,
    ptc_schematic,
    thermal_resistance_network,
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
    if "config" not in st.session_state:
        st.session_state.config = default_config()
    if "fluid_database" not in st.session_state:
        st.session_state.fluid_database = default_fluid_database()
    if "results" not in st.session_state:
        st.session_state.results = {}
    if "result_signature" not in st.session_state:
        st.session_state.result_signature = None
    if "validations" not in st.session_state:
        st.session_state.validations = {}
    if "loaded_package_name" not in st.session_state:
        st.session_state.loaded_package_name = None


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
    st.session_state.config = default_config()
    st.session_state.fluid_database = default_fluid_database()
    st.session_state.results = {}
    st.session_state.validations = {}
    st.session_state.result_signature = None


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


initialize_state()
cfg = st.session_state.config
fluid_db = st.session_state.fluid_database

st.title("Modelo nodal de colector cilindro-parabólico")
st.caption(
    "Migración del modelo MATLAB a Python con solver BDF, propiedades dependientes de la temperatura, "
    "edición de parámetros y visualización térmica nodo por nodo."
)

with st.sidebar:
    st.header("Configuración")
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
        environment["sky_delta_K"] = st.number_input("Tamb - Tsky (K)", min_value=0.0, value=float(environment["sky_delta_K"]), step=0.5)
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
        # Instante representativo de los KPIs:
        # - con DNI constante, np.nanargmax devuelve siempre el primer punto (t0),
        #   donde por la condición inicial Tout = Tin y la eficiencia transitoria es 0 %.
        # - con irradiación variable, mantenemos el instante de DNI máximo.
        dni_series = np.asarray(result.scalar_diag["DNI_W_m2"], dtype=float)
        finite_idx = np.flatnonzero(np.isfinite(dni_series))
        if finite_idx.size == 0:
            k_ref = len(result.t_s) - 1
            kpi_caption = "KPIs en el estado final (DNI no disponible)."
        else:
            dni_valid = dni_series[finite_idx]
            dni_scale = max(float(np.nanmax(np.abs(dni_valid))), 1.0)
            dni_is_constant = float(np.nanmax(dni_valid) - np.nanmin(dni_valid)) <= 1e-8 * dni_scale
            if dni_is_constant:
                k_ref = int(finite_idx[-1])
                kpi_caption = "KPIs en el estado final: el DNI es constante, por lo que se evita usar el punto inicial transitorio."
            else:
                max_dni = float(np.nanmax(dni_valid))
                peak_candidates = finite_idx[np.isclose(dni_series[finite_idx], max_dni, rtol=1e-10, atol=1e-9)]
                k_ref = int(peak_candidates[-1]) if peak_candidates.size else int(finite_idx[np.nanargmax(dni_valid)])
                kpi_caption = "KPIs en el instante de DNI máximo."

        cols = st.columns(6)
        cols[0].metric("DNI", f"{result.scalar_diag['DNI_W_m2'][k_ref]:.1f} W/m²")
        cols[1].metric("T salida", f"{result.Tout_C[k_ref]:.2f} °C")
        cols[2].metric("T absorbedor", f"{result.Tabs_mean_C[k_ref]:.2f} °C")
        cols[3].metric("Q útil", f"{result.scalar_diag['Quseful_W'][k_ref]:.1f} W")
        cols[4].metric("Q pérdidas", f"{result.scalar_diag['Qloss_W'][k_ref]:.1f} W")
        cols[5].metric("Eficiencia", f"{result.scalar_diag['eta_pct'][k_ref]:.2f} %")
        st.caption(kpi_caption)
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
        time_index = st.slider(
            "Instante de análisis",
            min_value=0,
            max_value=len(result.t_s) - 1,
            value=int(np.nanargmax(result.scalar_diag["DNI_W_m2"])),
            format="índice %d",
        )
        node_number = st.slider("Nodo axial", min_value=1, max_value=result.n_segments, value=min(1, result.n_segments))
        node_index = node_number - 1
        snapshot = result.node_snapshot(time_index, node_index)
        st.caption(f"LAT = {snapshot['LAT_h']:.3f} h; nodo {node_number}; x = {(node_index + 0.5) * result.config['geometry']['L'] / result.n_segments:.3f} m")
        top_left, top_right = st.columns(2)
        with top_left:
            st.plotly_chart(ptc_schematic(result.config, node_index), use_container_width=True)
        with top_right:
            st.plotly_chart(thermal_resistance_network(snapshot, result.config), use_container_width=True)
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
    st.write(
        "Las validaciones reproducen los vectores incluidos en los dos documentos fuente y en la versión MATLAB. "
        "Se ejecutan con la base de propiedades actualmente editada."
    )
    val_cols = st.columns(3)
    if val_cols[0].button("Ejecutar Bhambare/Sukhatme", use_container_width=True):
        try:
            with st.spinner("Ejecutando validación Bhambare/Sukhatme"):
                st.session_state.validations["bhambare"] = validate_bhambare(cfg, fluid_db)
        except Exception as exc:
            st.exception(exc)
    if val_cols[1].button("Ejecutar validación mensual TCC", use_container_width=True):
        try:
            with st.spinner("Ejecutando 12 casos mensuales"):
                st.session_state.validations["tcc"] = validate_tcc_monthly(cfg, fluid_db)
        except Exception as exc:
            st.exception(exc)
    if val_cols[2].button("Cargar Tabla 8 del prototipo", use_container_width=True):
        st.session_state.validations["prototype"] = prototype_tcc_table()

    if "bhambare" in st.session_state.validations:
        st.subheader("Bhambare / Sukhatme")
        validation = st.session_state.validations["bhambare"]
        st.caption(validation["note"])
        st.dataframe(validation["table"], use_container_width=True, hide_index=True)
        st.plotly_chart(validation_bhambare_figure(validation["table"]), use_container_width=True)
    if "tcc" in st.session_state.validations:
        st.subheader("TCC / TRNSYS mensual")
        validation = st.session_state.validations["tcc"]
        metrics = validation["metrics"]
        metric_cols = st.columns(3)
        metric_cols[0].metric("MAPE Tout", f"{metrics['MAPE_Tout_pct']:.3f} %")
        metric_cols[1].metric("MAPE Q útil", f"{metrics['MAPE_Qutil_pct']:.3f} %")
        metric_cols[2].metric("MAPE eta", f"{metrics['MAPE_eta_pct']:.3f} %")
        st.caption(validation["note"])
        st.dataframe(validation["table"], use_container_width=True, hide_index=True)
        st.plotly_chart(validation_tcc_figure(validation["table"]), use_container_width=True)
    if "prototype" in st.session_state.validations:
        st.subheader("Tabla 8 del prototipo")
        validation = st.session_state.validations["prototype"]
        prototype_cols = st.columns(2)
        prototype_cols[0].metric("RMSE", f"{validation['RMSE_pp']:.3f} puntos porcentuales")
        prototype_cols[1].metric("MAPE", f"{validation['MAPE_pct']:.3f} %")
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
