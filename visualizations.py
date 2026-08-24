"""Visualizaciones interactivas del modelo PTC y su red térmica."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ptc_model import SimulationResult


def daily_irradiance_histogram(hourly: pd.DataFrame) -> go.Figure:
    """Histograma horario del DNI y de la componente proyectada sobre la apertura."""
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=hourly["hour_label"],
            y=hourly["DNI_mean_W_m2"],
            name="DNI medio",
            hovertemplate="%{x}<br>DNI medio = %{y:.1f} W/m²<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=hourly["hour_label"],
            y=hourly["beam_on_aperture_mean_W_m2"],
            mode="lines+markers",
            name="DNI · cos(theta)",
            hovertemplate="%{x}<br>Sobre apertura = %{y:.1f} W/m²<extra></extra>",
        )
    )
    figure.update_layout(
        height=430,
        title="Histograma diario de irradiación directa",
        xaxis_title="Hora LAT",
        yaxis_title="Irradiancia (W/m²)",
        hovermode="x unified",
        bargap=0.16,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    figure.update_xaxes(tickangle=-45)
    return figure


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
            "Eficiencia térmica HTF",
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
    if "eta_optical_abs_pct" in result.scalar_diag:
        figure.add_trace(
            go.Scatter(
                x=x,
                y=result.scalar_diag["eta_optical_abs_pct"],
                mode="lines",
                name="η óptica al absorbedor",
                line={"dash": "dash"},
                showlegend=True,
            ),
            row=2,
            col=3,
        )
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
    """Dibuja la red térmica física del receptor para un volumen axial.

    La topología reproduce el circuito clásico de siete nodos:

        (1)--conv.--(2)--cond.--(3)==rad./conv.==(4)--cond.--(5)
                                                        |--conv.--(6)
                                                        |--rad.---(7)

    En el modelo numérico actual las paredes son capacitancias concentradas,
    por lo que T2 = T3 = Tabs y T4 = T5 = Tglass. Las resistencias de
    conducción se muestran explícitamente para conservar la lectura física
    del circuito.
    """
    has_glass = bool(config["model"]["has_glass"])
    if not has_glass:
        return _thermal_resistance_network_without_glass(snapshot, config)

    g = config["geometry"]
    absorber = config["materials"]["absorber"]
    glass = config["materials"]["glass"]
    environment = config["environment"]

    n_segments = max(int(g["Nseg"]), 1)
    dx = float(g["L"]) / n_segments
    area_inner = np.pi * float(g["D2"]) * dx
    h_internal = max(float(snapshot["h_internal_W_m2K"]), np.finfo(float).eps)

    r12 = 1.0 / (h_internal * area_inner)
    r23 = np.log(float(g["D3"]) / float(g["D2"])) / (
        2.0 * np.pi * float(absorber["k"]) * dx
    )
    r34_rad = float(snapshot["R_rad_abs_glass_K_W"])
    r34_conv = float(snapshot["R_conv_annulus_K_W"])
    r45 = np.log(float(g["D5"]) / float(g["D4"])) / (
        2.0 * np.pi * float(glass["k"]) * dx
    )
    r56 = float(snapshot["R_conv_external_K_W"])
    r57 = float(snapshot["R_rad_sky_K_W"])

    q12 = float(snapshot["Qfluid_W"])
    q23 = q12
    q34_rad = float(snapshot["Qrad_abs_glass_W"])
    q34_conv = float(snapshot["Qconv_annulus_W"])
    q56 = float(snapshot["Qconv_external_W"])
    q57 = float(snapshot["Qrad_sky_W"])
    q45 = q56 + q57

    tamb_c = float(environment["Tamb_K"]) - 273.15
    tsky_c = tamb_c - float(environment["sky_delta_K"])
    temperatures = {
        1: float(snapshot["Tf_C"]),
        2: float(snapshot["Tabs_C"]),
        3: float(snapshot["Tabs_C"]),
        4: float(snapshot["Tglass_C"]),
        5: float(snapshot["Tglass_C"]),
        6: tamb_c,
        7: tsky_c,
    }

    # Coordenadas: el trazado replica la disposición de la figura de referencia.
    y_mid = 0.50
    y_top = 0.73
    y_bottom = 0.27
    x1, x2, x3, x4, x5, x_end = 0.04, 0.17, 0.31, 0.55, 0.69, 0.94

    fig = go.Figure()

    # Ramas principales 1-2, 2-3 y 4-5.
    _add_black_resistor(
        fig, x1, y_mid, x2, y_mid,
        hover=_network_hover("Convección interna", "R12", r12, "Q12", q12),
    )
    _add_black_resistor(
        fig, x2, y_mid, x3, y_mid,
        hover=_network_hover("Conducción en el absorbedor", "R23", r23, "Q23", q23),
    )
    _add_black_resistor(
        fig, x4, y_mid, x5, y_mid,
        hover=_network_hover("Conducción en el vidrio", "R45", r45, "Q45", q45),
    )

    # Paralelo absorbedor-vidrio: radiación arriba y convección abajo.
    _add_black_polyline(fig, [x3, x3], [y_mid, y_top])
    _add_black_resistor(
        fig, x3, y_top, x4, y_top,
        hover=_network_hover("Radiación absorbedor-vidrio", "R34,rad", r34_rad, "Q34,rad", q34_rad),
    )
    _add_black_polyline(fig, [x4, x4], [y_top, y_mid])

    _add_black_polyline(fig, [x3, x3], [y_mid, y_bottom])
    _add_black_resistor(
        fig, x3, y_bottom, x4, y_bottom,
        hover=_network_hover("Convección en el anular", "R34,conv", r34_conv, "Q34,conv", q34_conv),
    )
    _add_black_polyline(fig, [x4, x4], [y_bottom, y_mid])

    # Paralelo vidrio-entorno: radiación al cielo y convección al ambiente.
    _add_black_polyline(fig, [x5, x5], [y_mid, y_top])
    _add_black_resistor(
        fig, x5, y_top, x_end, y_top,
        hover=_network_hover("Radiación al cielo", "R57", r57, "Q57", q57),
    )

    _add_black_polyline(fig, [x5, x5], [y_mid, y_bottom])
    _add_black_resistor(
        fig, x5, y_bottom, x_end, y_bottom,
        hover=_network_hover("Convección exterior", "R56", r56, "Q56", q56),
    )

    # Nodos físicos negros.
    node_coordinates = {
        1: (x1, y_mid),
        2: (x2, y_mid),
        3: (x3, y_mid),
        4: (x4, y_mid),
        5: (x5, y_mid),
        6: (x_end, y_bottom),
        7: (x_end, y_top),
    }
    for number, (x, y) in node_coordinates.items():
        _add_black_node(fig, x, y, number, temperatures[number])

    # Etiquetas de mecanismos, manteniendo la misma lectura de la referencia.
    _add_network_label(fig, 0.5 * (x1 + x2), 0.39, "convección interna", r12, q12)
    _add_network_label(fig, 0.5 * (x2 + x3), 0.39, "conducción", r23, q23)
    _add_network_label(fig, 0.5 * (x3 + x4), 0.86, "radiación", r34_rad, q34_rad)
    _add_network_label(fig, 0.5 * (x3 + x4), 0.10, "convección", r34_conv, q34_conv)
    _add_network_label(fig, 0.5 * (x4 + x5), 0.39, "conducción", r45, q45)
    _add_network_label(fig, 0.5 * (x5 + x_end), 0.86, "radiación", r57, q57)
    _add_network_label(fig, 0.5 * (x5 + x_end), 0.10, "convección", r56, q56)

    q_htf_rise = -float(snapshot["Qadvection_W"])
    q_htf_storage = q12 - q_htf_rise
    fig.add_annotation(
        x=0.50,
        y=-0.070,
        xref="paper",
        yref="paper",
        text=(
            "Nodos: 1 HTF; 2 pared interna del absorbedor; 3 pared externa del absorbedor; "
            "4 pared interna del vidrio; 5 pared externa del vidrio; 6 ambiente; 7 cielo.<br>"
            f"<b>Transporte axial HTF del volumen:</b> ΔH = {q_htf_rise:.2f} W &nbsp; | &nbsp; "
            f"Q absorbedor→HTF = {q12:.2f} W &nbsp; | &nbsp; "
            f"acumulación = {q_htf_storage:.3e} W. "
            "En régimen estacionario la acumulación tiende a cero, no los flujos."
        ),
        showarrow=False,
        font={"color": "black", "size": 11},
        align="center",
        bgcolor="white",
    )

    fig.update_xaxes(range=[0.0, 0.98], visible=False, fixedrange=True)
    fig.update_yaxes(range=[0.0, 0.94], visible=False, fixedrange=True)
    fig.update_layout(
        height=560,
        title={
            "text": (
                f"Circuito térmico del nodo axial {int(snapshot['node_index']) + 1} "
                f"· LAT {snapshot['LAT_h']:.3f} h"
            ),
            "x": 0.5,
            "xanchor": "center",
            "font": {"color": "black", "size": 18},
        },
        showlegend=False,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "black"},
        hoverlabel={"bgcolor": "white", "font_color": "black", "bordercolor": "black"},
        margin={"l": 12, "r": 12, "t": 70, "b": 105},
    )
    return fig


def _thermal_resistance_network_without_glass(
    snapshot: Mapping[str, float], config: Mapping[str, Any]
) -> go.Figure:
    """Red equivalente cuando el receptor se configura sin cubierta de vidrio."""
    g = config["geometry"]
    absorber = config["materials"]["absorber"]
    environment = config["environment"]
    n_segments = max(int(g["Nseg"]), 1)
    dx = float(g["L"]) / n_segments
    area_inner = np.pi * float(g["D2"]) * dx
    h_internal = max(float(snapshot["h_internal_W_m2K"]), np.finfo(float).eps)
    r12 = 1.0 / (h_internal * area_inner)
    r23 = np.log(float(g["D3"]) / float(g["D2"])) / (
        2.0 * np.pi * float(absorber["k"]) * dx
    )
    r36 = float(snapshot["R_conv_external_K_W"])
    r37 = float(snapshot["R_rad_sky_K_W"])
    q12 = float(snapshot["Qfluid_W"])
    q36 = float(snapshot["Qconv_external_W"])
    q37 = float(snapshot["Qrad_sky_W"])
    tamb_c = float(environment["Tamb_K"]) - 273.15
    tsky_c = tamb_c - float(environment["sky_delta_K"])

    fig = go.Figure()
    x1, x2, x3, x_end = 0.05, 0.22, 0.40, 0.94
    y_mid, y_top, y_bottom = 0.50, 0.72, 0.28
    _add_black_resistor(fig, x1, y_mid, x2, y_mid, hover=_network_hover("Convección interna", "R12", r12, "Q12", q12))
    _add_black_resistor(fig, x2, y_mid, x3, y_mid, hover=_network_hover("Conducción absorbedor", "R23", r23, "Q23", q12))
    _add_black_polyline(fig, [x3, x3], [y_mid, y_top])
    _add_black_resistor(fig, x3, y_top, x_end, y_top, hover=_network_hover("Radiación al cielo", "R37", r37, "Q37", q37))
    _add_black_polyline(fig, [x3, x3], [y_mid, y_bottom])
    _add_black_resistor(fig, x3, y_bottom, x_end, y_bottom, hover=_network_hover("Convección exterior", "R36", r36, "Q36", q36))

    temperatures = {
        1: float(snapshot["Tf_C"]),
        2: float(snapshot["Tabs_C"]),
        3: float(snapshot["Tabs_C"]),
        6: tamb_c,
        7: tsky_c,
    }
    coordinates = {1: (x1, y_mid), 2: (x2, y_mid), 3: (x3, y_mid), 6: (x_end, y_bottom), 7: (x_end, y_top)}
    for number, (x, y) in coordinates.items():
        _add_black_node(fig, x, y, number, temperatures[number])

    _add_network_label(fig, 0.5 * (x1 + x2), 0.39, "convección interna", r12, q12)
    _add_network_label(fig, 0.5 * (x2 + x3), 0.39, "conducción", r23, q12)
    _add_network_label(fig, 0.5 * (x3 + x_end), 0.84, "radiación", r37, q37)
    _add_network_label(fig, 0.5 * (x3 + x_end), 0.12, "convección", r36, q36)

    fig.add_annotation(
        x=0.5, y=-0.05, xref="paper", yref="paper",
        text="Configuración sin vidrio: los nodos 4 y 5 no forman parte del circuito.",
        showarrow=False, font={"color": "black", "size": 11},
    )
    fig.update_xaxes(range=[0.0, 0.98], visible=False, fixedrange=True)
    fig.update_yaxes(range=[0.0, 0.94], visible=False, fixedrange=True)
    fig.update_layout(
        height=540,
        title={"text": f"Circuito térmico del nodo axial {int(snapshot['node_index']) + 1}", "x": 0.5, "font": {"color": "black"}},
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "black"},
        hoverlabel={"bgcolor": "white", "font_color": "black", "bordercolor": "black"},
        showlegend=False,
        margin={"l": 12, "r": 12, "t": 70, "b": 70},
    )
    return fig


def _add_black_resistor(
    figure: go.Figure,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    hover: str,
    teeth: int = 6,
) -> None:
    """Agrega una resistencia en zigzag, siempre negra y sobre fondo blanco."""
    dx = x1 - x0
    dy = y1 - y0
    length = max(float(np.hypot(dx, dy)), np.finfo(float).eps)
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    lead = min(0.018, 0.14 * length)
    amplitude = min(0.020, 0.12 * length)

    start_x, start_y = x0 + lead * ux, y0 + lead * uy
    end_x, end_y = x1 - lead * ux, y1 - lead * uy
    n_points = 2 * teeth + 1
    base_x = np.linspace(start_x, end_x, n_points)
    base_y = np.linspace(start_y, end_y, n_points)
    zig = np.zeros(n_points)
    zig[1:-1] = amplitude * np.where(np.arange(1, n_points - 1) % 2 == 1, 1.0, -1.0)
    xs = np.concatenate(([x0], base_x + zig * nx, [x1]))
    ys = np.concatenate(([y0], base_y + zig * ny, [y1]))
    figure.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line={"color": "black", "width": 2.2},
            hovertemplate=hover + "<extra></extra>",
            showlegend=False,
        )
    )


def _add_black_polyline(figure: go.Figure, xs: list[float], ys: list[float]) -> None:
    figure.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line={"color": "black", "width": 2.2},
            hoverinfo="skip",
            showlegend=False,
        )
    )


def _add_black_node(
    figure: go.Figure,
    x: float,
    y: float,
    number: int,
    temperature_c: float,
) -> None:
    figure.add_trace(
        go.Scatter(
            x=[x],
            y=[y],
            mode="markers",
            marker={"size": 8, "color": "black", "line": {"color": "black", "width": 1}},
            hovertemplate=f"Nodo {number}<br>T{number} = {temperature_c:.2f} °C<extra></extra>",
            showlegend=False,
        )
    )
    horizontal_shift = -0.025 if number == 1 else (0.022 if number in {6, 7} else 0.0)
    vertical_shift = 0.0 if number in {1, 6, 7} else -0.052
    figure.add_annotation(
        x=x + horizontal_shift,
        y=y + vertical_shift,
        text=f"({number})",
        showarrow=False,
        font={"color": "black", "size": 13},
        xanchor="right" if number == 1 else ("left" if number in {6, 7} else "center"),
        yanchor="middle" if number in {1, 6, 7} else "top",
    )


def _add_network_label(
    figure: go.Figure,
    x: float,
    y: float,
    mechanism: str,
    resistance: float,
    heat_flow: float,
) -> None:
    figure.add_annotation(
        x=x,
        y=y,
        text=(
            f"<b>{mechanism}</b><br>"
            f"R = {_fmt_resistance(resistance)}<br>"
            f"Q = {heat_flow:.2f} W"
        ),
        showarrow=False,
        align="center",
        font={"color": "black", "size": 11},
        bgcolor="white",
        borderpad=1,
    )


def _network_hover(
    mechanism: str,
    resistance_name: str,
    resistance: float,
    heat_name: str,
    heat_flow: float,
) -> str:
    return (
        f"<b>{mechanism}</b><br>"
        f"{resistance_name} = {_fmt_resistance(resistance)}<br>"
        f"{heat_name} = {heat_flow:.4g} W"
    )

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
    """Comparación mensual Rea Quille/TRNSYS vs modelo Python."""
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Temperatura de salida", "Ganancia útil", "Eficiencia", "Errores relativos"),
    )
    x = table["Mes"]
    cols = {
        "tout_ref": "Tout_ref_C",
        "tout_sim": "Tout_Python_C" if "Tout_Python_C" in table.columns else "Tout_sim_C",
        "q_ref": "Qutil_ref_derivado_W" if "Qutil_ref_derivado_W" in table.columns else "Qutil_ref_W",
        "q_sim": "Qutil_Python_W" if "Qutil_Python_W" in table.columns else "Qutil_sim_W",
        "eta_ref": "Eta_ref_pct",
        "eta_sim": "Eta_Python_pct" if "Eta_Python_pct" in table.columns else "Eta_sim_pct",
    }
    for y, name in ((cols["tout_ref"], "Rea Quille/TRNSYS"), (cols["tout_sim"], "Modelo Python")):
        figure.add_trace(go.Scatter(x=x, y=table[y], mode="lines+markers", name=name), row=1, col=1)
    for y, name in ((cols["q_ref"], "Derivado de la referencia"), (cols["q_sim"], "Modelo Python")):
        figure.add_trace(go.Scatter(x=x, y=table[y], mode="lines+markers", name=name), row=1, col=2)
    for y, name in ((cols["eta_ref"], "Rea Quille/TRNSYS"), (cols["eta_sim"], "Modelo Python")):
        figure.add_trace(go.Scatter(x=x, y=table[y], mode="lines+markers", name=name), row=2, col=1)
    for y, name in (("Err_Tout_pct", "Tout"), ("Err_Qutil_pct", "Qutil"), ("Err_Eta_pct", "eta")):
        figure.add_trace(go.Bar(x=x, y=table[y], name=name), row=2, col=2)
    figure.update_yaxes(title_text="°C", row=1, col=1)
    figure.update_yaxes(title_text="W", row=1, col=2)
    figure.update_yaxes(title_text="%", row=2, col=1)
    figure.update_yaxes(title_text="Error (%)", row=2, col=2)
    figure.update_layout(height=760, title="Validación mensual Rea Quille / TRNSYS", barmode="group")
    return figure


def validation_bhambare_figure(table: pd.DataFrame) -> go.Figure:
    reduced = table.iloc[:4].copy()
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Comparación", "Error relativo"))
    ref_col = "Referencia_Sukhatme" if "Referencia_Sukhatme" in table.columns else "Referencia"
    art_col = "Modelo_Bhambare" if "Modelo_Bhambare" in table.columns else "Modelo_articulo"
    err_col = "Error_vs_Sukhatme_pct" if "Error_vs_Sukhatme_pct" in table.columns else "Error_rel_pct"
    for column, name in ((ref_col, "Sukhatme"), (art_col, "Bhambare"), ("Modelo_Python", "Modelo Python")):
        figure.add_trace(go.Bar(x=reduced["Magnitud"], y=reduced[column], name=name), row=1, col=1)
    figure.add_trace(go.Bar(x=table["Magnitud"], y=table[err_col], name="Error relativo"), row=1, col=2)
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
            "Eficiencia térmica HTF",
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
    if results:
        first_result = next(iter(results.values()))
        if "eta_optical_abs_pct" in first_result.scalar_diag:
            figure.add_trace(
                go.Scatter(
                    x=first_result.LAT_h,
                    y=first_result.scalar_diag["eta_optical_abs_pct"],
                    mode="lines",
                    name="η óptica al absorbedor",
                    line={"dash": "dash"},
                    showlegend=True,
                ),
                row=2,
                col=3,
            )
    figure.update_layout(
        height=900,
        title="Comparación de escenarios del PTC",
        hovermode="x unified",
    )
    return figure
