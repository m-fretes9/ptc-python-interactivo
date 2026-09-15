"""Gráficos de la auditoría de flujo externo Rea Quille."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def external_h_figure(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not isinstance(table, pd.DataFrame) or table.empty:
        fig.update_layout(title="Sin datos para la auditoría de flujo externo")
        return fig
    x = table["Mes"].astype(str)
    fig.add_trace(go.Scatter(x=x, y=table["h_modelo_W_m2K"], mode="lines+markers", name="h modelo"))
    fig.add_trace(go.Scatter(x=x, y=table["h_requerido_W_m2K"], mode="lines+markers", name="h requerido"))
    fig.update_layout(
        title="Coeficiente convectivo externo · modelo vs requerido",
        xaxis_title="Mes", yaxis_title="h externo (W/m²K)", height=430, hovermode="x unified",
        legend={"orientation":"h","yanchor":"bottom","y":1.02,"xanchor":"left","x":0.0},
    )
    return fig


def external_wind_figure(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not isinstance(table, pd.DataFrame) or table.empty:
        fig.update_layout(title="Sin datos para la auditoría de flujo externo")
        return fig
    x = table["Mes"].astype(str)
    fig.add_trace(go.Scatter(x=x, y=table["Viento_modelo_m_s"], mode="lines+markers", name="Viento asumido"))
    fig.add_trace(go.Scatter(x=x, y=table["Viento_requerido_m_s"], mode="lines+markers", name="Viento implícito requerido"))
    fig.update_layout(
        title="¿Puede el viento asumido explicar el exceso de convección?",
        xaxis_title="Mes", yaxis_title="Velocidad de viento (m/s)", height=430, hovermode="x unified",
        legend={"orientation":"h","yanchor":"bottom","y":1.02,"xanchor":"left","x":0.0},
    )
    return fig


def external_dimensionless_figure(table: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if not isinstance(table, pd.DataFrame) or table.empty:
        fig.update_layout(title="Sin datos para la auditoría de flujo externo")
        return fig
    x = table["Mes"].astype(str)
    fig.add_trace(go.Scatter(x=x, y=table["Re_modelo"], mode="lines+markers", name="Re modelo"), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=table["Re_requerido"], mode="lines+markers", name="Re con viento requerido"), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=table["Nu_modelo"], mode="lines+markers", name="Nu modelo", line={"dash":"dot"}), secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=table["Nu_requerido"], mode="lines+markers", name="Nu requerido", line={"dash":"dot"}), secondary_y=True)
    fig.update_yaxes(title_text="Reynolds externo (-)", secondary_y=False)
    fig.update_yaxes(title_text="Nusselt externo (-)", secondary_y=True)
    fig.update_layout(
        title="Estado de la correlación Churchill–Bernstein",
        xaxis_title="Mes", height=450, hovermode="x unified",
        legend={"orientation":"h","yanchor":"bottom","y":1.02,"xanchor":"left","x":0.0},
    )
    return fig


def external_temperature_figure(table: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if not isinstance(table, pd.DataFrame) or table.empty:
        fig.update_layout(title="Sin datos para la auditoría de flujo externo")
        return fig
    x = table["Mes"].astype(str)
    fig.add_trace(go.Scatter(x=x, y=table["Tsuperficie_media_C"], mode="lines+markers", name="T superficie"), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=table["Tamb_C"], mode="lines+markers", name="T ambiente"), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=table["Tfilm_media_C"], mode="lines+markers", name="T película", line={"dash":"dot"}), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=table["Factor_h_requerido"], mode="lines+markers", name="h requerido / h modelo"), secondary_y=True)
    fig.add_hline(y=1.0, line_dash="dash", secondary_y=True)
    fig.update_yaxes(title_text="Temperatura (°C)", secondary_y=False)
    fig.update_yaxes(title_text="Factor h requerido (-)", secondary_y=True)
    fig.update_layout(
        title="Temperatura de película y corrección convectiva requerida",
        xaxis_title="Mes", height=450, hovermode="x unified",
        legend={"orientation":"h","yanchor":"bottom","y":1.02,"xanchor":"left","x":0.0},
    )
    return fig
