"""Gráficos de la prueba de apantallamiento aerodinámico."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

CAL_MONTHS = {"Ene", "Abr", "Jul", "Oct"}


def shielding_objective_figure(trace: pd.DataFrame, sv_star: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trace["S_v"], y=trace["RMSE_cal_Foz_pp"], mode="lines+markers", name="RMSE calibración Foz"
    ))
    best = trace.iloc[(trace["S_v"] - float(sv_star)).abs().argsort()[:1]]
    if not best.empty:
        fig.add_trace(go.Scatter(
            x=best["S_v"], y=best["RMSE_cal_Foz_pp"], mode="markers", marker={"size": 12}, name=f"Óptimo S_v={sv_star:.3f}"
        ))
    fig.add_vline(x=1.0, line_dash="dash", annotation_text="Sin apantallamiento")
    fig.update_layout(
        title="Calibración controlada · un único factor S_v",
        xaxis_title="Factor de apantallamiento S_v (-)", yaxis_title="RMSE η · 4 meses Foz (pp)",
        height=420, hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    return fig


def _monthly_eta_figure(df: pd.DataFrame, title: str, shade_calibration: bool = False) -> go.Figure:
    x = df["Mes"].astype(str)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=df["Eta_ref_pct"], mode="lines+markers", name="Referencia Rea"))
    fig.add_trace(go.Scatter(x=x, y=df["Eta_baseline_pct"], mode="lines+markers", name="Baseline · S_v=1"))
    fig.add_trace(go.Scatter(x=x, y=df["Eta_apantallada_pct"], mode="lines+markers", name="S_v calibrado en Foz"))
    if shade_calibration:
        months = list(x)
        for idx, month in enumerate(months):
            if month in CAL_MONTHS:
                fig.add_vrect(x0=idx - 0.45, x1=idx + 0.45, fillcolor="rgba(120,120,120,0.10)", line_width=0)
    fig.update_layout(
        title=title, xaxis_title="Mes", yaxis_title="Eficiencia térmica (%)", height=440, hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    return fig


def shielding_foz_figure(table: pd.DataFrame) -> go.Figure:
    df = table[table["Caso"] == "rea_foz"].sort_values("Mes_num")
    return _monthly_eta_figure(df, "Foz · calibrar 4 meses y validar 8 meses no usados", shade_calibration=True)


def shielding_alvorada_figure(table: pd.DataFrame) -> go.Figure:
    df = table[table["Caso"] == "rea_alvorada"].sort_values("Mes_num")
    return _monthly_eta_figure(df, "Alvorada · transferencia del mismo S_v sin recalibración", shade_calibration=False)


def shielding_rmse_summary_figure(metrics: dict) -> go.Figure:
    cats = ["Foz · calibración", "Foz · hold-out", "Alvorada · externa"]
    groups = [metrics["foz_cal"], metrics["foz_holdout"], metrics["alvorada_external"]]
    before = [g["eta_baseline"]["rmse"] for g in groups]
    after = [g["eta_shielded"]["rmse"] for g in groups]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cats, y=before, name="Baseline"))
    fig.add_trace(go.Bar(x=cats, y=after, name="Con S_v congelado"))
    fig.update_layout(
        title="¿El factor generaliza fuera de la muestra usada para calibrarlo?",
        yaxis_title="RMSE eficiencia (pp)", height=420, barmode="group",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    return fig
