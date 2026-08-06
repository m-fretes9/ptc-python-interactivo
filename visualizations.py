"""Visualizaciones interactivas del modelo PTC y su red térmica."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ptc_model import SimulationResult


def dynamic_overview(result: SimulationResult) -> go.Figure:
    figure = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=(
            "Temperatura de salida HTF",
            "Temperatura media del absorbedor",
            "Temperatura media del vidrio",
            "Ganancia útil",
            "Pérdidas térmicas",
            "Eficiencia térmica",
            "Coeficiente global U_L",
            "Reynolds interno a la salida",
            "h interno a la salida",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )
    x = result.LAT_h
    series = [
        (result.Tout_C, "T_out", "°C"),
        (result.Tabs_mean_C, "T_abs", "°C"),
        (result.Tglass_mean_C, "T_vid", "°C"),
        (result.scalar_diag["Quseful_W"], "Q_util", "W"),
        (result.scalar_diag["Qloss_W"], "Q_loss", "W"),
        (result.scalar_diag["eta_pct"], "eta", "%"),
        (result.scalar_diag["UL_W_m2K"], "U_L", "W/(m² K)"),
        (result.node_diag["Re_internal"][:, -1], "Re", "-"),
        (result.node_diag["h_internal_W_m2K"][:, -1], "h_int", "W/(m² K)"),
    ]
    for index, (values, name, unit) in enumerate(series):
        row = index // 3 + 1
        col = index % 3 + 1
        figure.add_trace(
            go.Scatter(x=x, y=values, mode="lines", name=name, showlegend=False),
            row=row,
            col=col,
        )
        figure.update_xaxes(title_text="LAT (h)", row=row, col=col)
        figure.update_yaxes(title_text=unit, row=row, col=col)
    figure.update_layout(height=900, title="Respuesta transitoria del PTC", hovermode="x unified")
    return figure


def axial_profiles(result: SimulationResult, time_index: int) -> go.Figure:
    k = int(np.clip(time_index, 0, len(result.t_s) - 1))
    n = result.n_segments
    length = float(result.config["geometry"]["L"])
    x = (np.arange(n) + 0.5) * length / n
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Temperaturas axiales",
            "Flujos térmicos por nodo",
            "Convección interna",
            "Propiedades locales del HTF",
        ),
    )
    for values, name in (
        (result.Tf_C[k, :], "HTF"),
        (result.Tabs_C[k, :], "Absorbedor"),
        (result.Tglass_C[k, :], "Vidrio"),
    ):
        figure.add_trace(go.Scatter(x=x, y=values, mode="lines+markers", name=name), row=1, col=1)

    flux_series = (
        (result.node_diag["Qsolar_abs_node_W"][k, :], "Q solar absorbedor"),
        (result.node_diag["Qfluid_W"][k, :], "Q a HTF"),
        (result.node_diag["Qconv_external_W"][k, :], "Q conv. exterior"),
        (result.node_diag["Qrad_sky_W"][k, :], "Q rad. cielo"),
    )
    for values, name in flux_series:
        figure.add_trace(go.Scatter(x=x, y=values, mode="lines+markers", name=name), row=1, col=2)

    figure.add_trace(
        go.Scatter(x=x, y=result.node_diag["Re_internal"][k, :], mode="lines+markers", name="Re"),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(x=x, y=result.node_diag["Nu_internal"][k, :], mode="lines+markers", name="Nu"),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(x=x, y=result.node_diag["h_internal_W_m2K"][k, :], mode="lines+markers", name="h_int"),
        row=2,
        col=1,
    )

    figure.add_trace(
        go.Scatter(x=x, y=result.node_diag["rho_kg_m3"][k, :], mode="lines+markers", name="rho"),
        row=2,
        col=2,
    )
    figure.add_trace(
        go.Scatter(x=x, y=result.node_diag["Cp_J_kgK"][k, :], mode="lines+markers", name="Cp"),
        row=2,
        col=2,
    )
    figure.add_trace(
        go.Scatter(x=x, y=1e3 * result.node_diag["mu_Pa_s"][k, :], mode="lines+markers", name="1000·mu"),
        row=2,
        col=2,
    )
    figure.add_trace(
        go.Scatter(x=x, y=1e3 * result.node_diag["k_W_mK"][k, :], mode="lines+markers", name="1000·k"),
        row=2,
        col=2,
    )

    for row in (1, 2):
        for col in (1, 2):
            figure.update_xaxes(title_text="Posición axial x (m)", row=row, col=col)
    figure.update_yaxes(title_text="Temperatura (°C)", row=1, col=1)
    figure.update_yaxes(title_text="Potencia por nodo (W)", row=1, col=2)
    figure.update_yaxes(title_text="Re, Nu, h", row=2, col=1)
    figure.update_yaxes(title_text="Valores escalados", row=2, col=2)
    figure.update_layout(
        height=800,
        title=f"Perfiles axiales a LAT = {result.LAT_h[k]:.3f} h",
        hovermode="x unified",
    )
    return figure


def node_balance(snapshot: Mapping[str, float], has_glass: bool) -> go.Figure:
    inputs = {
        "Solar al absorbedor": snapshot["Qsolar_abs_node_W"],
        "Advección HTF": snapshot["Qadvection_W"],
    }
    outputs = {
        "Absorbedor -> HTF": -snapshot["Qfluid_W"],
        "Soportes": -snapshot["Qsupports_W"],
    }
    if has_glass:
        outputs.update(
            {
                "Abs. -> vidrio (rad.)": -snapshot["Qrad_abs_glass_W"],
                "Abs. -> vidrio (conv.)": -snapshot["Qconv_annulus_W"],
                "Vidrio -> ambiente": -snapshot["Qconv_external_W"],
                "Vidrio -> cielo": -snapshot["Qrad_sky_W"],
            }
        )
        inputs["Solar al vidrio"] = snapshot["Qsolar_glass_node_W"]
    else:
        outputs.update(
            {
                "Abs. -> ambiente": -snapshot["Qconv_external_W"],
                "Abs. -> cielo": -snapshot["Qrad_sky_W"],
            }
        )
    labels = list(inputs) + list(outputs)
    values = list(inputs.values()) + list(outputs.values())
    figure = go.Figure(go.Bar(x=labels, y=values))
    figure.add_hline(y=0.0)
    figure.update_layout(
        height=430,
        title="Flujos del nodo seleccionado: positivo entra, negativo sale",
        xaxis_title="Mecanismo",
        yaxis_title="Potencia (W)",
    )
    return figure


def ptc_schematic(config: Mapping[str, Any], selected_node: int) -> go.Figure:
    g = config["geometry"]
    n = int(g["Nseg"])
    width = float(g["W"])
    focus = float(g["f"])
    x = np.linspace(-width / 2.0, width / 2.0, 300)
    y = x**2 / (4.0 * focus)
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Sección transversal", "Discretización axial del receptor"),
        column_widths=[0.48, 0.52],
    )
    figure.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Reflector"), row=1, col=1)
    figure.add_trace(
        go.Scatter(
            x=[0.0],
            y=[focus],
            mode="markers+text",
            text=["Receptor"],
            textposition="top center",
            marker={"size": 18, "symbol": "circle-open"},
            name="Receptor",
        ),
        row=1,
        col=1,
    )
    ray_x = np.linspace(-0.42 * width, 0.42 * width, 6)
    for ray in ray_x:
        y_surface = ray**2 / (4.0 * focus)
        figure.add_annotation(
            x=ray,
            y=y_surface,
            ax=ray,
            ay=y_surface + 0.55 * focus + 0.15,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            text="",
        )
    figure.add_annotation(
        x=0.0,
        y=focus,
        ax=0.35 * width,
        ay=0.1 * focus,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        text="Rayo reflejado",
    )

    boundaries = np.linspace(0.0, float(g["L"]), n + 1)
    for i in range(n):
        fill = "rgba(100,100,100,0.55)" if i == selected_node else "rgba(100,100,100,0.12)"
        figure.add_shape(
            type="rect",
            x0=boundaries[i],
            x1=boundaries[i + 1],
            y0=-0.12,
            y1=0.12,
            line={"width": 1},
            fillcolor=fill,
            row=1,
            col=2,
        )
        figure.add_annotation(
            x=0.5 * (boundaries[i] + boundaries[i + 1]),
            y=0.0,
            text=str(i + 1),
            showarrow=False,
            xref="x2",
            yref="y2",
        )
    figure.add_annotation(
        x=0.5 * (boundaries[selected_node] + boundaries[selected_node + 1]),
        y=0.25,
        text=f"Nodo seleccionado: {selected_node + 1}",
        showarrow=True,
        ax=0,
        ay=-35,
        xref="x2",
        yref="y2",
    )
    figure.update_xaxes(title_text="Abertura (m)", scaleanchor="y", scaleratio=1, row=1, col=1)
    figure.update_yaxes(title_text="Profundidad (m)", row=1, col=1)
    figure.update_xaxes(title_text="Longitud del colector (m)", row=1, col=2)
    figure.update_yaxes(visible=False, range=[-0.35, 0.35], row=1, col=2)
    figure.update_layout(height=480, showlegend=False, title="Geometría y nodos del PTC")
    return figure


def thermal_resistance_network(
    snapshot: Mapping[str, float], config: Mapping[str, Any]
) -> go.Figure:
    has_glass = bool(config["model"]["has_glass"])
    figure = go.Figure()
    nodes = {
        "Solar": (0.07, 0.82),
        "HTF aguas arriba": (0.07, 0.18),
        "Absorbedor": (0.38, 0.60),
        "HTF nodo": (0.38, 0.18),
        "Vidrio": (0.65, 0.60),
        "Ambiente": (0.92, 0.76),
        "Cielo": (0.92, 0.38),
    }
    if not has_glass:
        nodes.pop("Vidrio")

    for label, (x, y) in nodes.items():
        if label == "Absorbedor":
            temperature = snapshot["Tabs_C"]
        elif label == "HTF nodo":
            temperature = snapshot["Tf_C"]
        elif label == "Vidrio":
            temperature = snapshot["Tglass_C"]
        elif label == "Ambiente":
            temperature = config["environment"]["Tamb_K"] - 273.15
        elif label == "Cielo":
            temperature = (
                config["environment"]["Tamb_K"]
                - config["environment"]["sky_delta_K"]
                - 273.15
            )
        else:
            temperature = None
        text = label if temperature is None else f"{label}<br>{temperature:.2f} °C"
        figure.add_shape(
            type="rect",
            x0=x - 0.085,
            x1=x + 0.085,
            y0=y - 0.055,
            y1=y + 0.055,
            line={"width": 1.5},
            fillcolor="rgba(150,150,150,0.10)",
        )
        figure.add_annotation(x=x, y=y, text=text, showarrow=False, align="center")

    _add_heat_arrow(
        figure,
        nodes["Solar"],
        nodes["Absorbedor"],
        f"Qsolar = {snapshot['Qsolar_abs_node_W']:.2f} W",
        y_offset=0.07,
    )
    _add_heat_arrow(
        figure,
        nodes["HTF aguas arriba"],
        nodes["HTF nodo"],
        f"Qadv = {snapshot['Qadvection_W']:.2f} W",
        y_offset=-0.07,
    )
    _add_resistor(
        figure,
        nodes["Absorbedor"],
        nodes["HTF nodo"],
        f"Rint = {_fmt_resistance(snapshot['R_internal_K_W'])}\nQ = {snapshot['Qfluid_W']:.2f} W",
        orientation="vertical",
    )

    if has_glass:
        _add_resistor(
            figure,
            (nodes["Absorbedor"][0], nodes["Absorbedor"][1] + 0.035),
            (nodes["Vidrio"][0], nodes["Vidrio"][1] + 0.035),
            f"Rrad,a-g = {_fmt_resistance(snapshot['R_rad_abs_glass_K_W'])}\nQrad = {snapshot['Qrad_abs_glass_W']:.2f} W",
            orientation="horizontal",
        )
        _add_resistor(
            figure,
            (nodes["Absorbedor"][0], nodes["Absorbedor"][1] - 0.055),
            (nodes["Vidrio"][0], nodes["Vidrio"][1] - 0.055),
            f"Rconv,an = {_fmt_resistance(snapshot['R_conv_annulus_K_W'])}\nQconv = {snapshot['Qconv_annulus_W']:.2f} W",
            orientation="horizontal",
        )
        _add_resistor(
            figure,
            nodes["Vidrio"],
            nodes["Ambiente"],
            f"Rconv,ext = {_fmt_resistance(snapshot['R_conv_external_K_W'])}\nQ = {snapshot['Qconv_external_W']:.2f} W",
            orientation="horizontal",
        )
        _add_resistor(
            figure,
            nodes["Vidrio"],
            nodes["Cielo"],
            f"Rrad,cielo = {_fmt_resistance(snapshot['R_rad_sky_K_W'])}\nQ = {snapshot['Qrad_sky_W']:.2f} W",
            orientation="horizontal",
        )
    else:
        _add_resistor(
            figure,
            nodes["Absorbedor"],
            nodes["Ambiente"],
            f"Rconv,ext = {_fmt_resistance(snapshot['R_conv_external_K_W'])}\nQ = {snapshot['Qconv_external_W']:.2f} W",
            orientation="horizontal",
        )
        _add_resistor(
            figure,
            nodes["Absorbedor"],
            nodes["Cielo"],
            f"Rrad,cielo = {_fmt_resistance(snapshot['R_rad_sky_K_W'])}\nQ = {snapshot['Qrad_sky_W']:.2f} W",
            orientation="horizontal",
        )

    figure.update_xaxes(range=[-0.03, 1.03], visible=False)
    figure.update_yaxes(range=[0.02, 0.98], visible=False)
    figure.update_layout(
        height=580,
        title=(
            f"Red térmica del nodo {int(snapshot['node_index']) + 1} "
            f"a LAT {snapshot['LAT_h']:.3f} h"
        ),
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )
    return figure


def property_figure(curve: pd.DataFrame, fluid_name: str) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Densidad", "Viscosidad dinámica", "Calor específico", "Conductividad térmica"),
    )
    columns = ["rho_kg_m3", "mu_Pa_s", "Cp_J_kgK", "k_W_mK"]
    units = ["kg/m³", "Pa·s", "J/(kg K)", "W/(m K)"]
    for index, (column, unit) in enumerate(zip(columns, units, strict=True)):
        row = index // 2 + 1
        col = index % 2 + 1
        figure.add_trace(
            go.Scatter(x=curve["T_C"], y=curve[column], mode="lines", name=column, showlegend=False),
            row=row,
            col=col,
        )
        figure.update_xaxes(title_text="T (°C)", row=row, col=col)
        figure.update_yaxes(title_text=unit, row=row, col=col)
    figure.update_layout(height=650, title=f"Propiedades termofísicas: {fluid_name}")
    return figure


def validation_tcc_figure(table: pd.DataFrame) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Temperatura de salida", "Ganancia útil", "Eficiencia", "Errores relativos"),
    )
    x = table["Mes"]
    for y, name in (("Tout_ref_C", "TCC/TRNSYS"), ("Tout_sim_C", "Modelo Python")):
        figure.add_trace(go.Scatter(x=x, y=table[y], mode="lines+markers", name=name), row=1, col=1)
    for y, name in (("Qutil_ref_W", "Derivado TCC"), ("Qutil_sim_W", "Modelo Python")):
        figure.add_trace(go.Scatter(x=x, y=table[y], mode="lines+markers", name=name), row=1, col=2)
    for y, name in (("Eta_ref_pct", "TCC/TRNSYS"), ("Eta_sim_pct", "Modelo Python")):
        figure.add_trace(go.Scatter(x=x, y=table[y], mode="lines+markers", name=name), row=2, col=1)
    for y, name in (
        ("Err_Tout_pct", "Tout"),
        ("Err_Qutil_pct", "Qutil"),
        ("Err_Eta_pct", "eta"),
    ):
        figure.add_trace(go.Bar(x=x, y=table[y], name=name), row=2, col=2)
    figure.update_yaxes(title_text="°C", row=1, col=1)
    figure.update_yaxes(title_text="W", row=1, col=2)
    figure.update_yaxes(title_text="%", row=2, col=1)
    figure.update_yaxes(title_text="Error (%)", row=2, col=2)
    figure.update_layout(height=760, title="Validación mensual TCC/TRNSYS", barmode="group")
    return figure


def validation_bhambare_figure(table: pd.DataFrame) -> go.Figure:
    reduced = table.iloc[:4].copy()
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Comparación", "Error relativo"))
    for column, name in (
        ("Referencia", "Referencia"),
        ("Modelo_articulo", "Modelo del artículo"),
        ("Modelo_Python", "Modelo Python"),
    ):
        figure.add_trace(go.Bar(x=reduced["Magnitud"], y=reduced[column], name=name), row=1, col=1)
    figure.add_trace(
        go.Bar(x=table["Magnitud"], y=table["Error_rel_pct"], name="Error relativo"),
        row=1,
        col=2,
    )
    figure.update_yaxes(title_text="Valor", row=1, col=1)
    figure.update_yaxes(title_text="Error (%)", row=1, col=2)
    figure.update_layout(height=500, title="Validación Bhambare/Sukhatme", barmode="group")
    return figure


def _add_heat_arrow(
    figure: go.Figure,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    y_offset: float = 0.0,
) -> None:
    figure.add_annotation(
        x=end[0],
        y=end[1],
        ax=start[0],
        ay=start[1],
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        text="",
    )
    figure.add_annotation(
        x=0.5 * (start[0] + end[0]),
        y=0.5 * (start[1] + end[1]) + y_offset,
        text=label,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.75)",
    )


def _add_resistor(
    figure: go.Figure,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    orientation: str,
) -> None:
    x0, y0 = start
    x1, y1 = end
    if orientation == "vertical":
        ys = np.linspace(y0, y1, 10)
        amplitude = 0.018
        xs = np.full_like(ys, x0)
        xs[2:-2] += amplitude * np.array([1, -1, 1, -1, 1, -1])[: len(xs[2:-2])]
    else:
        xs = np.linspace(x0, x1, 10)
        amplitude = 0.018
        ys = np.linspace(y0, y1, 10)
        normal_x = -(y1 - y0)
        normal_y = x1 - x0
        norm = max(np.hypot(normal_x, normal_y), 1e-12)
        normal_x /= norm
        normal_y /= norm
        zig = np.zeros(10)
        zig[2:-2] = np.array([1, -1, 1, -1, 1, -1])[:6] * amplitude
        xs = xs + zig * normal_x
        ys = ys + zig * normal_y
    figure.add_trace(go.Scatter(x=xs, y=ys, mode="lines", showlegend=False, hoverinfo="skip"))
    figure.add_annotation(
        x=end[0],
        y=end[1],
        ax=0.5 * (start[0] + end[0]),
        ay=0.5 * (start[1] + end[1]),
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        text="",
    )
    figure.add_annotation(
        x=0.5 * (start[0] + end[0]),
        y=0.5 * (start[1] + end[1]) + (0.07 if orientation == "horizontal" else 0.0),
        text=label.replace("\n", "<br>"),
        showarrow=False,
        bgcolor="rgba(255,255,255,0.82)",
        align="center",
    )


def _fmt_resistance(value: float) -> str:
    if not np.isfinite(value):
        return "infinito"
    return f"{value:.4g} K/W"


def comparative_overview(results: Mapping[str, SimulationResult]) -> go.Figure:
    """Dashboard 3x3 equivalente al barrido de caudales del MATLAB."""
    figure = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=(
            "Temperatura de salida HTF",
            "Temperatura media del absorbedor",
            "Temperatura media del vidrio",
            "Ganancia útil",
            "Pérdidas térmicas",
            "Eficiencia térmica",
            "Coeficiente global U_L",
            "Reynolds interno a la salida",
            "h interno a la salida",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )
    for label, result in results.items():
        series = [
            (result.Tout_C, "°C"),
            (result.Tabs_mean_C, "°C"),
            (result.Tglass_mean_C, "°C"),
            (result.scalar_diag["Quseful_W"], "W"),
            (result.scalar_diag["Qloss_W"], "W"),
            (result.scalar_diag["eta_pct"], "%"),
            (result.scalar_diag["UL_W_m2K"], "W/(m² K)"),
            (result.node_diag["Re_internal"][:, -1], "-"),
            (result.node_diag["h_internal_W_m2K"][:, -1], "W/(m² K)"),
        ]
        for index, (values, unit) in enumerate(series):
            row = index // 3 + 1
            col = index % 3 + 1
            figure.add_trace(
                go.Scatter(
                    x=result.LAT_h,
                    y=values,
                    mode="lines",
                    name=label,
                    legendgroup=label,
                    showlegend=index == 0,
                ),
                row=row,
                col=col,
            )
            figure.update_xaxes(title_text="LAT (h)", row=row, col=col)
            figure.update_yaxes(title_text=unit, row=row, col=col)
    figure.update_layout(
        height=900,
        title="Comparación de escenarios del PTC",
        hovermode="x unified",
    )
    return figure
