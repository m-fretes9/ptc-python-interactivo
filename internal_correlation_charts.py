"""Gráficos de la auditoría de régimen/correlación interna V14.12."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def regime_map_figure(monthly: pd.DataFrame, bh: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for city in ("Foz", "Alvorada"):
        d = monthly[(monthly["Ciudad_caso"] == city) & (monthly["Correlacion_id"] == "laminar_436_forzado")]
        fig.add_trace(go.Scatter(x=d["Mes"], y=d["Re_mean"], mode="lines+markers", name=f"{city} · agua"))
    fig.add_trace(go.Scatter(x=["Bhambare"], y=[float(bh.iloc[0]["Re_mean"])], mode="markers", marker_size=11, name="Bhambare · Paratherm"))
    fig.add_hline(y=2300, line_dash="dash", annotation_text="Re=2300 · límite laminar usado")
    fig.add_hline(y=4000, line_dash="dot", annotation_text="Re=4000 · turbulento del modelo")
    fig.update_layout(title="Mapa de régimen interno · agua y Paratherm", yaxis_title="Reynolds (-)", xaxis_title="Mes / caso", margin=dict(l=20,r=20,t=70,b=40))
    return fig


def nu_by_month_figure(monthly: pd.DataFrame, city: str) -> go.Figure:
    fig = go.Figure()
    d0 = monthly[monthly["Ciudad_caso"] == city]
    for corr, d in d0.groupby("Correlacion", sort=False):
        fig.add_trace(go.Scatter(x=d["Mes"], y=d["Nu_mean"], mode="lines+markers", name=corr))
    fig.update_layout(title=f"Nusselt interno según correlación · {city}", yaxis_title="Nu medio (-)", xaxis_title="Mes", margin=dict(l=20,r=20,t=70,b=40), legend_orientation="h")
    return fig


def rmse_comparison_figure(metrics: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for city in ("Foz", "Alvorada"):
        d = metrics[metrics["Ciudad"] == city]
        fig.add_trace(go.Bar(x=d["Correlacion"], y=d["RMSE_eta_pp"], name=city, text=[f"{x:.2f}" for x in d["RMSE_eta_pp"]], textposition="outside"))
    fig.update_layout(title="Efecto de la correlación interna sobre el RMSE de eficiencia", yaxis_title="RMSE η (pp)", xaxis_title="Correlación", barmode="group", margin=dict(l=20,r=20,t=70,b=80), legend_orientation="h")
    return fig


def efficiency_curves_figure(monthly: pd.DataFrame, city: str) -> go.Figure:
    fig = go.Figure()
    d0 = monthly[monthly["Ciudad_caso"] == city]
    ref = d0[d0["Correlacion_id"] == "laminar_436_forzado"]
    fig.add_trace(go.Scatter(x=ref["Mes"], y=ref["Eta_ref_pct"], mode="lines+markers", name="Referencia Rea"))
    for corr, d in d0.groupby("Correlacion", sort=False):
        fig.add_trace(go.Scatter(x=d["Mes"], y=d["Eta_pct"], mode="lines+markers", name=corr))
    fig.update_layout(title=f"Eficiencia mensual · cambio exclusivo de correlación interna · {city}", yaxis_title="Eficiencia (%)", xaxis_title="Mes", margin=dict(l=20,r=20,t=70,b=40), legend_orientation="h")
    return fig


def bhambare_correlation_figure(bh: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=bh["Correlacion"], y=bh["h_mean_W_m2K"], text=[f"{v:.1f}" for v in bh["h_mean_W_m2K"]], textposition="outside", name="h interno"))
    fig.update_layout(title="Bhambare · h interno resultante de cada correlación", yaxis_title="h (W/m²K)", xaxis_title="Correlación", margin=dict(l=20,r=20,t=70,b=80), showlegend=False)
    return fig
