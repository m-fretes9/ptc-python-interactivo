"""Gráficos de la prueba de viento meteorológico mensual."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def wind_input_figure(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    x = table["Mes"].astype(str)
    fig.add_trace(go.Scatter(x=x, y=table["Viento_base_m_s"], mode="lines+markers", name="Baseline · 1 m/s"))
    fig.add_trace(go.Scatter(x=x, y=table["Viento_10m_m_s"], mode="lines+markers", name="Meteorología · 10 m"))
    fig.add_trace(go.Scatter(x=x, y=table["Viento_usado_m_s"], mode="lines+markers", name="Viento usado en receptor"))
    fig.update_layout(
        title="Entrada de viento · supuesto fijo vs meteorología independiente",
        xaxis_title="Mes", yaxis_title="Velocidad (m/s)", height=430, hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    return fig


def wind_eta_figure(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    x = table["Mes"].astype(str)
    fig.add_trace(go.Scatter(x=x, y=table["Eta_ref_pct"], mode="lines+markers", name="Referencia Rea"))
    fig.add_trace(go.Scatter(x=x, y=table["Eta_base_pct"], mode="lines+markers", name="Modelo · viento fijo"))
    fig.add_trace(go.Scatter(x=x, y=table["Eta_meteo_pct"], mode="lines+markers", name="Modelo · viento meteorológico"))
    fig.update_layout(
        title="¿El viento meteorológico mejora la curva de eficiencia?",
        xaxis_title="Mes", yaxis_title="Eficiencia térmica (%)", height=450, hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    return fig


def wind_tout_figure(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    x = table["Mes"].astype(str)
    fig.add_trace(go.Scatter(x=x, y=table["Tout_ref_C"], mode="lines+markers", name="Tout referencia"))
    fig.add_trace(go.Scatter(x=x, y=table["Tout_base_C"], mode="lines+markers", name="Tout · viento fijo"))
    fig.add_trace(go.Scatter(x=x, y=table["Tout_meteo_C"], mode="lines+markers", name="Tout · viento meteorológico"))
    fig.update_layout(
        title="Temperatura de salida · efecto exclusivo del viento mensual",
        xaxis_title="Mes", yaxis_title="Tout (°C)", height=430, hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    return fig


def wind_convection_figure(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    x = table["Mes"].astype(str)
    fig.add_trace(go.Bar(x=x, y=table["Qconv_base_W"], name="Qconv · viento fijo"))
    fig.add_trace(go.Bar(x=x, y=table["Qconv_meteo_W"], name="Qconv · viento meteorológico"))
    fig.update_layout(
        title="Cambio inducido en pérdidas convectivas",
        xaxis_title="Mes", yaxis_title="Qconv externa (W)", barmode="group", height=430,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    return fig
