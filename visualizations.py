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
    tsky_c = float(snapshot.get("Tsky_C", tamb_c - float(environment.get("sky_delta_K", 6.0))))
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
    tsky_c = float(snapshot.get("Tsky_C", tamb_c - float(environment.get("sky_delta_K", 6.0))))

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

# -----------------------------------------------------------------------------
# Mejora visual 2026-08-27: circuito térmico enriquecido, selector axial
# clickeable y sección transversal interactiva con mapa de calor / ray tracing.
# -----------------------------------------------------------------------------


def node_flow_selector_figure(result: SimulationResult, time_index: int, selected_node: int) -> go.Figure:
    """Selector axial compacto con nodos clickeables y flujo entre volúmenes.

    Cada marcador representa un volumen axial. Entre nodos consecutivos se
    visualiza el transporte entálpico del HTF; además, el color de cada nodo se
    asocia al calor radial absorbido por el fluido en ese volumen.
    """
    k = int(np.clip(time_index, 0, len(result.t_s) - 1))
    n = result.n_segments
    length = float(result.config["geometry"]["L"])
    x = (np.arange(n) + 0.5) * length / n
    q_htf = np.asarray(result.node_diag["Qfluid_W"][k, :], dtype=float)
    dH = -np.asarray(result.node_diag["Qadvection_W"][k, :], dtype=float)
    tf = np.asarray(result.Tf_C[k, :], dtype=float)

    fig = go.Figure()

    # Línea base del receptor.
    fig.add_trace(
        go.Scatter(
            x=[0.0, length], y=[0.0, 0.0],
            mode="lines",
            line={"color": "rgba(90,90,90,0.35)", "width": 7},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    if n > 1:
        for i in range(n - 1):
            x0 = x[i]
            x1 = x[i + 1]
            q_link = 0.5 * (dH[i] + dH[i + 1])
            color = "rgba(244,106,38,0.85)" if q_link >= 0.0 else "rgba(59,130,246,0.85)"
            fig.add_annotation(
                x=x1 - 0.03 * length / max(n, 2),
                y=0.16,
                ax=x0 + 0.03 * length / max(n, 2),
                ay=0.16,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.1,
                arrowwidth=2.0,
                arrowcolor=color,
                text="",
            )
            fig.add_annotation(
                x=0.5 * (x0 + x1),
                y=0.24,
                xref="x",
                yref="y",
                showarrow=False,
                text=f"ΔH {q_link:.1f} W",
                font={"size": 10, "color": "#334155"},
                bgcolor="rgba(255,255,255,0.86)",
                bordercolor="rgba(148,163,184,0.5)",
                borderpad=2,
            )

    marker_sizes = np.full(n, 20.0)
    marker_sizes[int(np.clip(selected_node, 0, n - 1))] = 30.0
    marker_lines = ["#334155"] * n
    marker_line_widths = [1.5] * n
    marker_lines[int(np.clip(selected_node, 0, n - 1))] = "#111827"
    marker_line_widths[int(np.clip(selected_node, 0, n - 1))] = 3.0

    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.zeros(n),
            mode="markers+text",
            text=[str(i + 1) for i in range(n)],
            textposition="middle center",
            customdata=np.column_stack([np.arange(1, n + 1), q_htf, dH, tf]),
            marker={
                "size": marker_sizes.tolist(),
                "color": q_htf,
                "colorscale": "Turbo",
                "showscale": True,
                "colorbar": {"title": "Q abs→HTF (W)", "thickness": 12},
                "line": {"width": marker_line_widths, "color": marker_lines},
                "cmin": float(np.nanmin(q_htf)) if np.any(np.isfinite(q_htf)) else 0.0,
                "cmax": float(np.nanmax(q_htf)) if np.any(np.isfinite(q_htf)) else 1.0,
            },
            hovertemplate=(
                "<b>Nodo axial %{customdata[0]:.0f}</b><br>"
                "Q abs→HTF = %{customdata[1]:.2f} W<br>"
                "ΔH axial = %{customdata[2]:.2f} W<br>"
                "T agua = %{customdata[3]:.2f} °C<extra></extra>"
            ),
            showlegend=False,
        )
    )

    fig.add_annotation(
        x=x[int(np.clip(selected_node, 0, n - 1))],
        y=-0.18,
        xref="x",
        yref="y",
        text=f"Nodo activo: {int(np.clip(selected_node, 0, n - 1)) + 1}",
        showarrow=False,
        font={"size": 12, "color": "#111827"},
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(17,24,39,0.3)",
        borderpad=3,
    )

    fig.update_xaxes(title_text="Posición axial x (m)", range=[-0.03 * length, 1.03 * length])
    fig.update_yaxes(visible=False, range=[-0.28, 0.36], fixedrange=True)
    fig.update_layout(
        height=220,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        title={"text": "Deslizador axial clickeable y flujo entre nodos", "x": 0.5, "font": {"size": 17}},
        template="plotly_white",
        hovermode="closest",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "#111827"},
    )
    return fig


# ---- Ray tracing helpers ----------------------------------------------------

def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(float).eps:
        return vector
    return vector / norm


def _intersect_ray_parabola(x_launch: float, y_launch: float, direction: np.ndarray, focus: float) -> np.ndarray | None:
    dx, dy = float(direction[0]), float(direction[1])
    # y = x^2/(4f) ; x = x0 + t dx ; y = y0 + t dy
    a = dx * dx
    b = 2.0 * x_launch * dx - 4.0 * focus * dy
    c = x_launch * x_launch - 4.0 * focus * y_launch
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return None
        roots = [-c / b]
    else:
        roots = np.roots([a, b, c])
    valid = [float(np.real(root)) for root in roots if abs(np.imag(root)) < 1e-9 and float(np.real(root)) >= 0.0]
    if not valid:
        return None
    t = min(valid)
    return np.array([x_launch + t * dx, y_launch + t * dy], dtype=float)


def _parabola_normal(point: np.ndarray, focus: float) -> np.ndarray:
    x = float(point[0])
    normal = np.array([-x / (2.0 * focus), 1.0], dtype=float)
    return _normalize(normal)


def _reflect(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    d = _normalize(direction)
    n = _normalize(normal)
    return _normalize(d - 2.0 * np.dot(d, n) * n)


def _intersect_ray_circle(origin: np.ndarray, direction: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray | None:
    p = origin - center
    d = _normalize(direction)
    a = np.dot(d, d)
    b = 2.0 * np.dot(p, d)
    c = np.dot(p, p) - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    roots = [(-b - np.sqrt(disc)) / (2.0 * a), (-b + np.sqrt(disc)) / (2.0 * a)]
    valid = [float(root) for root in roots if float(root) >= 1e-6]
    if not valid:
        return None
    t = min(valid)
    return origin + t * d


def _trace_single_ray(config: Mapping[str, Any], incidence_angle_deg: float, aperture_fraction: float) -> dict[str, Any] | None:
    g = config["geometry"]
    width = float(g["W"])
    focus = float(g["f"])
    center = np.array([0.0, focus], dtype=float)
    radius_abs = 0.5 * float(g["D3"])
    angle = np.deg2rad(float(incidence_angle_deg))
    direction = _normalize(np.array([np.sin(angle), -np.cos(angle)], dtype=float))
    x_launch = (float(aperture_fraction) - 0.5) * width
    y_launch = max(width * width / (16.0 * focus) + 0.15 * width, focus + 0.25 * width)
    hit_reflector = _intersect_ray_parabola(x_launch, y_launch, direction, focus)
    if hit_reflector is None:
        return None
    normal = _parabola_normal(hit_reflector, focus)
    reflected = _reflect(direction, normal)
    hit_receiver = _intersect_ray_circle(hit_reflector + 1e-5 * reflected, reflected, center, radius_abs)
    return {
        "launch": np.array([x_launch, y_launch], dtype=float),
        "reflector": hit_reflector,
        "receiver": hit_receiver,
        "direction": direction,
        "reflected": reflected,
    }


def _receiver_heatmap(config: Mapping[str, Any], incidence_angle_deg: float, n_rays: int = 180, grid_n: int = 170) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    g = config["geometry"]
    width = float(g["W"])
    focus = float(g["f"])
    center = np.array([0.0, focus], dtype=float)
    radius_abs = 0.5 * float(g["D3"])
    radius_glass = 0.5 * float(g["D5"])

    x = np.linspace(-0.18 * width, 0.18 * width, grid_n)
    y = np.linspace(focus - 0.18 * width, focus + 0.18 * width, grid_n)
    xx, yy = np.meshgrid(x, y)
    heat = np.zeros_like(xx)
    hit_points: list[np.ndarray] = []
    for frac in np.linspace(0.04, 0.96, n_rays):
        ray = _trace_single_ray(config, incidence_angle_deg, float(frac))
        if ray is None or ray["receiver"] is None:
            continue
        hit = np.asarray(ray["receiver"], dtype=float)
        hit_points.append(hit)
        sigma = 0.18 * radius_abs
        heat += np.exp(-((xx - hit[0]) ** 2 + (yy - hit[1]) ** 2) / (2.0 * sigma * sigma))

    ring_abs = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    mask_abs = np.abs(ring_abs - radius_abs) <= max(radius_abs * 0.20, 0.0015)
    heat *= mask_abs
    heat[~mask_abs] = np.nan
    meta = {
        "center": center,
        "radius_abs": radius_abs,
        "radius_glass": radius_glass,
        "n_hits": len(hit_points),
        "hit_points": hit_points,
    }
    return x, y, heat, meta


def _draw_cross_section_base(fig: go.Figure, config: Mapping[str, Any], snapshot: Mapping[str, float] | None = None) -> dict[str, Any]:
    g = config["geometry"]
    width = float(g["W"])
    focus = float(g["f"])
    x_par = np.linspace(-width / 2.0, width / 2.0, 500)
    y_par = x_par ** 2 / (4.0 * focus)
    y_top = max(np.max(y_par) + 0.12 * width, focus + 0.2 * width)
    center = np.array([0.0, focus], dtype=float)
    radius_abs = 0.5 * float(g["D3"])
    radius_glass = 0.5 * float(g["D5"])
    has_glass = bool(config["model"].get("has_glass", True))

    fig.add_trace(
        go.Scatter(
            x=x_par, y=y_par, mode="lines", name="Reflector",
            line={"color": "#334155", "width": 3}, hoverinfo="skip",
        )
    )

    t = np.linspace(0.0, 2.0 * np.pi, 240)
    x_abs = center[0] + radius_abs * np.cos(t)
    y_abs = center[1] + radius_abs * np.sin(t)
    hover_abs = "Absorbedor"
    if snapshot is not None:
        hover_abs += f"<br>Tabs ≈ {float(snapshot['Tabs_C']):.2f} °C"
    fig.add_trace(
        go.Scatter(
            x=x_abs, y=y_abs, mode="lines", name="Absorbedor",
            line={"color": "#ef4444", "width": 4},
            hovertemplate=hover_abs + "<extra></extra>",
        )
    )
    if has_glass:
        x_gl = center[0] + radius_glass * np.cos(t)
        y_gl = center[1] + radius_glass * np.sin(t)
        hover_gl = "Cubierta de vidrio"
        if snapshot is not None:
            hover_gl += f"<br>Tvid ≈ {float(snapshot['Tglass_C']):.2f} °C"
        fig.add_trace(
            go.Scatter(
                x=x_gl, y=y_gl, mode="lines", name="Vidrio",
                line={"color": "#64748b", "width": 2, "dash": "dot"},
                hovertemplate=hover_gl + "<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[center[0]], y=[center[1]], mode="markers+text", name="Foco",
            text=["F"], textposition="bottom center",
            marker={"size": 8, "color": "#0f172a"}, hoverinfo="skip",
        )
    )
    fig.update_xaxes(title_text="Abertura x (m)", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text="Profundidad y (m)", autorange="reversed")
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "#111827"},
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    return {"width": width, "focus": focus, "y_top": y_top, "center": center, "radius_abs": radius_abs}


def ptc_schematic(
    config: Mapping[str, Any],
    snapshot: Mapping[str, float] | None,
    selected_node: int,
    view_mode: str = "heatmap",
    incidence_angle_deg: float = 0.0,
    ray_seed: int = 0,
) -> go.Figure:
    """Sección transversal del PTC con modo de mapa de calor o seguidor de rayo."""
    fig = go.Figure()
    base = _draw_cross_section_base(fig, config, snapshot)
    width = base["width"]
    y_top = base["y_top"]

    # Dirección incidente global.
    for x0 in np.linspace(-0.38 * width, 0.38 * width, 5):
        dx = 0.08 * width * np.sin(np.deg2rad(incidence_angle_deg))
        fig.add_annotation(
            x=x0,
            y=0.02,
            ax=x0 - dx,
            ay=-0.16 * width,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowwidth=1.4,
            arrowcolor="rgba(245,158,11,0.9)",
            text="",
        )
    fig.add_annotation(
        x=0.0,
        y=-0.18 * width,
        xref="x",
        yref="y",
        showarrow=False,
        text=f"Incidencia visualizada: θ = {float(incidence_angle_deg):.2f}°",
        font={"size": 11, "color": "#92400e"},
        bgcolor="rgba(255,251,235,0.85)",
        bordercolor="rgba(245,158,11,0.35)",
    )

    if view_mode == "heatmap":
        x_grid, y_grid, heat, meta = _receiver_heatmap(config, incidence_angle_deg)
        fig.add_trace(
            go.Heatmap(
                x=x_grid,
                y=y_grid,
                z=heat,
                colorscale="Turbo",
                zsmooth="best",
                opacity=0.82,
                colorbar={"title": "Intensidad relativa"},
                hovertemplate="x = %{x:.4f} m<br>y = %{y:.4f} m<br>I_rel = %{z:.3f}<extra></extra>",
                name="Mapa térmico",
            )
        )
        title = "Sección transversal del PTC · mapa de calor sobre el absorbedor"
        subtitle = (
            f"Impactos trazados: {meta['n_hits']} rayos sobre el absorbedor · "
            f"nodo axial resaltado: {int(selected_node) + 1}"
        )
    else:
        rng = np.random.default_rng(int(ray_seed) + int(selected_node) * 17)
        frac = float(rng.uniform(0.08, 0.92))
        ray = _trace_single_ray(config, incidence_angle_deg, frac)
        if ray is not None:
            p0 = ray["launch"]
            p1 = ray["reflector"]
            p2 = ray["receiver"]
            fig.add_trace(
                go.Scatter(
                    x=[p0[0], p1[0]], y=[p0[1], p1[1]], mode="lines+markers",
                    name="Rayo incidente", line={"color": "#f59e0b", "width": 3},
                    marker={"size": 6}, hoverinfo="skip",
                )
            )
            if p2 is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[p1[0], p2[0]], y=[p1[1], p2[1]], mode="lines+markers",
                        name="Rayo reflejado", line={"color": "#2563eb", "width": 3},
                        marker={"size": 6}, hoverinfo="skip",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[p2[0]], y=[p2[1]], mode="markers+text",
                        text=["impacto"], textposition="top center",
                        name="Impacto", marker={"size": 10, "color": "#111827"},
                        hovertemplate="Punto de impacto sobre el absorbedor<extra></extra>",
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=[p1[0]], y=[p1[1]], mode="markers+text",
                    text=["reflexión"], textposition="bottom center",
                    name="Punto de reflexión", marker={"size": 9, "color": "#7c3aed"},
                    hoverinfo="skip",
                )
            )
        title = "Sección transversal del PTC · seguidor de rayo"
        subtitle = f"Rayo aleatorio sobre la apertura · nodo axial resaltado: {int(selected_node) + 1}"

    fig.update_layout(
        height=560,
        title={"text": title + "<br><sup>" + subtitle + "</sup>", "x": 0.5},
    )
    return fig


# ---- Thermal network enhanced ----------------------------------------------

def _resistor_points(start: tuple[float, float], end: tuple[float, float], teeth: int = 6) -> tuple[np.ndarray, np.ndarray]:
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    length = max(float(np.hypot(dx, dy)), np.finfo(float).eps)
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    lead = min(0.018, 0.14 * length)
    amplitude = min(0.018, 0.12 * length)
    start_x, start_y = x0 + lead * ux, y0 + lead * uy
    end_x, end_y = x1 - lead * ux, y1 - lead * uy
    n_points = 2 * teeth + 1
    base_x = np.linspace(start_x, end_x, n_points)
    base_y = np.linspace(start_y, end_y, n_points)
    zig = np.zeros(n_points)
    zig[1:-1] = amplitude * np.where(np.arange(1, n_points - 1) % 2 == 1, 1.0, -1.0)
    xs = np.concatenate(([x0], base_x + zig * nx, [x1]))
    ys = np.concatenate(([y0], base_y + zig * ny, [y1]))
    return xs, ys


def _add_resistor_branch(fig: go.Figure, start: tuple[float, float], end: tuple[float, float], mechanism: str, resistance: float, heat_flow: float) -> None:
    xs, ys = _resistor_points(start, end)
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="lines",
            line={"color": "black", "width": 2.2},
            hovertemplate=(
                f"<b>{mechanism}</b><br>R = {_fmt_resistance(resistance)}<br>Q = {heat_flow:.3f} W<extra></extra>"
            ),
            showlegend=False,
        )
    )
    sx, sy = start
    ex, ey = end
    q_sign = float(np.sign(heat_flow))
    if q_sign == 0.0:
        q_sign = 1.0
    if q_sign > 0.0:
        arrow_start = (sx + 0.28 * (ex - sx), sy + 0.28 * (ey - sy))
        arrow_end = (sx + 0.72 * (ex - sx), sy + 0.72 * (ey - sy))
    else:
        arrow_start = (sx + 0.72 * (ex - sx), sy + 0.72 * (ey - sy))
        arrow_end = (sx + 0.28 * (ex - sx), sy + 0.28 * (ey - sy))
    fig.add_annotation(
        x=arrow_end[0], y=arrow_end[1], ax=arrow_start[0], ay=arrow_start[1],
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1.0, arrowwidth=1.8, arrowcolor="#dc2626", text="",
    )
    fig.add_annotation(
        x=0.5 * (sx + ex), y=0.5 * (sy + ey) - 0.07 if abs(ey - sy) < 0.02 else 0.5 * (sy + ey),
        xref="x", yref="y",
        showarrow=False,
        text=f"<b>{mechanism}</b><br>R={_fmt_resistance(resistance)}<br>Q={heat_flow:.2f} W",
        font={"size": 10, "color": "black"},
        bgcolor="white",
        bordercolor="rgba(0,0,0,0.15)", borderpad=2,
        align="center",
    )


def _add_named_node(fig: go.Figure, coord: tuple[float, float], number: int, label: str, temperature_c: float) -> None:
    x, y = coord
    fig.add_trace(
        go.Scatter(
            x=[x], y=[y], mode="markers+text",
            text=[f"({number})"], textposition="bottom center",
            marker={"size": 9, "color": "black"},
            hovertemplate=f"<b>{label}</b><br>T = {temperature_c:.2f} °C<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_annotation(
        x=x, y=y + 0.055, xref="x", yref="y", showarrow=False,
        text=f"<b>{label}</b>", font={"size": 11, "color": "#111827"},
        bgcolor="white", bordercolor="rgba(0,0,0,0.10)", borderpad=2,
    )


def thermal_resistance_network(snapshot: Mapping[str, float], config: Mapping[str, Any]) -> go.Figure:
    """Circuito térmico mejorado con flechas, nombres físicos y radiación solar."""
    has_glass = bool(config["model"].get("has_glass", True))
    g = config["geometry"]
    absorber_k = float(config["materials"]["absorber"]["k"])
    n_segments = max(int(g["Nseg"]), 1)
    dx = float(g["L"]) / n_segments
    area_inner = np.pi * float(g["D2"]) * dx
    h_internal = max(float(snapshot["h_internal_W_m2K"]), np.finfo(float).eps)
    r12 = 1.0 / (h_internal * area_inner)
    r23 = np.log(float(g["D3"]) / float(g["D2"])) / (2.0 * np.pi * absorber_k * dx)

    tamb_c = float(config["environment"]["Tamb_K"]) - 273.15
    tsky_c = float(snapshot.get("Tsky_C", tamb_c - float(config["environment"].get("sky_delta_K", 6.0))))
    fluid_key = str(config.get("operation", {}).get("fluid", "HTF"))
    fluid_label = "Tagua" if fluid_key.lower() in {"agua", "water"} else "THTF"

    # Coordinates
    y_mid, y_top, y_bottom, y_solar = 0.50, 0.76, 0.24, 0.92
    x1, x2, x3, x4, x5, x_end = 0.06, 0.19, 0.33, 0.55, 0.69, 0.93
    fig = go.Figure()

    # Solar input into absorber external surface/node 3
    q_solar_abs = float(snapshot.get("Qsolar_abs_node_W", 0.0))
    fig.add_annotation(
        x=x3, y=y_mid + 0.02, ax=x3, ay=y_solar,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2.2, arrowcolor="#f59e0b", text="",
    )
    fig.add_annotation(
        x=x3, y=y_solar + 0.01, xref="x", yref="y", showarrow=False,
        text=f"<b>Radiación solar</b><br>Qsolar = {q_solar_abs:.2f} W",
        bgcolor="rgba(255,251,235,0.95)", bordercolor="rgba(245,158,11,0.35)",
        font={"size": 11, "color": "#92400e"},
    )

    q12 = float(snapshot["Qfluid_W"])
    _add_resistor_branch(fig, (x1, y_mid), (x2, y_mid), "convección interna", r12, q12)
    _add_resistor_branch(fig, (x2, y_mid), (x3, y_mid), "conducción pared abs.", r23, q12)

    if has_glass:
        glass_k = float(config["materials"]["glass"]["k"])
        r34_rad = float(snapshot["R_rad_abs_glass_K_W"])
        r34_conv = float(snapshot["R_conv_annulus_K_W"])
        r45 = np.log(float(g["D5"]) / float(g["D4"])) / (2.0 * np.pi * glass_k * dx)
        r56 = float(snapshot["R_conv_external_K_W"])
        r57 = float(snapshot["R_rad_sky_K_W"])
        q34_rad = float(snapshot["Qrad_abs_glass_W"])
        q34_conv = float(snapshot["Qconv_annulus_W"])
        q56 = float(snapshot["Qconv_external_W"])
        q57 = float(snapshot["Qrad_sky_W"])
        q45 = q56 + q57

        # connectors up/down
        fig.add_trace(go.Scatter(x=[x3, x3], y=[y_mid, y_top], mode="lines", line={"color": "black", "width": 2}, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[x4, x4], y=[y_mid, y_top], mode="lines", line={"color": "black", "width": 2}, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[x3, x3], y=[y_mid, y_bottom], mode="lines", line={"color": "black", "width": 2}, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[x4, x4], y=[y_mid, y_bottom], mode="lines", line={"color": "black", "width": 2}, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[x5, x5], y=[y_mid, y_top], mode="lines", line={"color": "black", "width": 2}, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[x5, x5], y=[y_mid, y_bottom], mode="lines", line={"color": "black", "width": 2}, hoverinfo="skip", showlegend=False))

        _add_resistor_branch(fig, (x3, y_top), (x4, y_top), "radiación abs→vid", r34_rad, q34_rad)
        _add_resistor_branch(fig, (x3, y_bottom), (x4, y_bottom), "convección anular", r34_conv, q34_conv)
        _add_resistor_branch(fig, (x4, y_mid), (x5, y_mid), "conducción vidrio", r45, q45)
        _add_resistor_branch(fig, (x5, y_top), (x_end, y_top), "radiación al cielo", r57, q57)
        _add_resistor_branch(fig, (x5, y_bottom), (x_end, y_bottom), "convección al ambiente", r56, q56)

        coordinates = {
            1: (x1, y_mid), 2: (x2, y_mid), 3: (x3, y_mid), 4: (x4, y_mid), 5: (x5, y_mid), 6: (x_end, y_bottom), 7: (x_end, y_top)
        }
        labels = {
            1: fluid_label, 2: "Tabs,int", 3: "Tabs,ext", 4: "Tvid,int", 5: "Tvid,ext", 6: "Tamb", 7: "Tsky"
        }
        temperatures = {
            1: float(snapshot["Tf_C"]), 2: float(snapshot["Tabs_C"]), 3: float(snapshot["Tabs_C"]),
            4: float(snapshot["Tglass_C"]), 5: float(snapshot["Tglass_C"]), 6: tamb_c, 7: tsky_c,
        }
        for n_id, coord in coordinates.items():
            _add_named_node(fig, coord, n_id, labels[n_id], temperatures[n_id])
        q_external = q56 + q57 + float(snapshot.get("Qsupports_W", 0.0))
    else:
        r36 = float(snapshot["R_conv_external_K_W"])
        r37 = float(snapshot["R_rad_sky_K_W"])
        q36 = float(snapshot["Qconv_external_W"])
        q37 = float(snapshot["Qrad_sky_W"])
        fig.add_trace(go.Scatter(x=[x3, x3], y=[y_mid, y_top], mode="lines", line={"color": "black", "width": 2}, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[x3, x3], y=[y_mid, y_bottom], mode="lines", line={"color": "black", "width": 2}, hoverinfo="skip", showlegend=False))
        _add_resistor_branch(fig, (x3, y_top), (x_end, y_top), "radiación al cielo", r37, q37)
        _add_resistor_branch(fig, (x3, y_bottom), (x_end, y_bottom), "convección al ambiente", r36, q36)
        coordinates = {1: (x1, y_mid), 2: (x2, y_mid), 3: (x3, y_mid), 6: (x_end, y_bottom), 7: (x_end, y_top)}
        labels = {1: fluid_label, 2: "Tabs,int", 3: "Tabs,ext", 6: "Tamb", 7: "Tsky"}
        temperatures = {1: float(snapshot["Tf_C"]), 2: float(snapshot["Tabs_C"]), 3: float(snapshot["Tabs_C"]), 6: tamb_c, 7: tsky_c}
        for n_id, coord in coordinates.items():
            _add_named_node(fig, coord, n_id, labels[n_id], temperatures[n_id])
        q_external = q36 + q37 + float(snapshot.get("Qsupports_W", 0.0))

    q_htf_rise = -float(snapshot["Qadvection_W"])
    q_htf_storage = q12 - q_htf_rise
    fig.add_annotation(
        x=0.50, y=-0.10, xref="paper", yref="paper", showarrow=False,
        text=(
            f"<b>Resumen del volumen axial:</b> Qsolar={q_solar_abs:.2f} W &nbsp; | &nbsp; "
            f"Q abs→HTF={q12:.2f} W &nbsp; | &nbsp; ΔH axial={q_htf_rise:.2f} W &nbsp; | &nbsp; "
            f"pérdida exterior={q_external:.2f} W &nbsp; | &nbsp; acumulación HTF={q_htf_storage:.3e} W"
        ),
        font={"size": 11, "color": "#111827"},
        bgcolor="white",
        bordercolor="rgba(0,0,0,0.12)",
        borderpad=4,
        align="center",
    )

    fig.update_xaxes(range=[0.0, 0.98], visible=False, fixedrange=True)
    fig.update_yaxes(range=[0.08, 0.98], visible=False, fixedrange=True)
    fig.update_layout(
        height=620,
        title={"text": f"Circuito térmico del nodo axial {int(snapshot['node_index']) + 1} · LAT {snapshot['LAT_h']:.3f} h", "x": 0.5},
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "black"},
        hoverlabel={"bgcolor": "white", "font_color": "black", "bordercolor": "black"},
        margin={"l": 8, "r": 8, "t": 70, "b": 110},
        showlegend=False,
    )
    return fig
