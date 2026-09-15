"""Gráficos de la prueba de agregación temporal de irradiancia."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def temporal_profile_figure(profiles: pd.DataFrame, month_label: str) -> go.Figure:
    df = profiles[profiles["Mes"].astype(str) == str(month_label)].copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Hora_solar_h"], y=df["DNI_W_m2"], mode="lines+markers", name="DNI temporal · igual energía"
    ))
    if not df.empty:
        mean_val = float(df["DNI_W_m2"].mean())
        fig.add_hline(y=mean_val, line_dash="dash", annotation_text=f"DNI medio = {mean_val:.1f} W/m²")
    fig.update_layout(
        title=f"Perfil representativo de DNI · {month_label}",
        xaxis_title="Hora solar", yaxis_title="DNI (W/m²)", height=420, hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    return fig


def temporal_eta_figure(table: pd.DataFrame) -> go.Figure:
    x = table["Mes"].astype(str)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=table["Eta_ref_pct"], mode="lines+markers", name="Referencia Rea"))
    fig.add_trace(go.Scatter(x=x, y=table["Eta_DNI_medio_pct"], mode="lines+markers", name="Modelo · DNI medio"))
    fig.add_trace(go.Scatter(
        x=x, y=table["Eta_agregacion_temporal_pct"], mode="lines+markers", name="Modelo · agregación temporal"
    ))
    fig.update_layout(
        title="¿Distribuir el mismo DNI medio cambia la curva de eficiencia?",
        xaxis_title="Mes", yaxis_title="Eficiencia térmica (%)", height=440, hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    return fig


def temporal_tout_figure(table: pd.DataFrame) -> go.Figure:
    x = table["Mes"].astype(str)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=table["Tout_ref_C"], mode="lines+markers", name="Tout referencia"))
    fig.add_trace(go.Scatter(x=x, y=table["Tout_DNI_medio_C"], mode="lines+markers", name="Tout · DNI medio"))
    fig.add_trace(go.Scatter(
        x=x, y=table["Tout_agregacion_temporal_C"], mode="lines+markers", name="Tout · agregación temporal"
    ))
    fig.update_layout(
        title="Temperatura de salida · promedio mensual vs agregación temporal",
        xaxis_title="Mes", yaxis_title="Tout (°C)", height=420, hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    return fig


def temporal_jensen_gap_figure(table: pd.DataFrame) -> go.Figure:
    x = table["Mes"].astype(str)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=table["Delta_eta_agregacion_pp"], name="Δη temporal − DNI medio"))
    fig.add_hline(y=0.0, line_dash="dash")
    fig.update_layout(
        title="Efecto puro de agregación · f(promedio) vs promedio(f)",
        xaxis_title="Mes", yaxis_title="Δ eficiencia (puntos porcentuales)", height=420,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    return fig
