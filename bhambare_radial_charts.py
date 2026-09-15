"""Gráficos Plotly para la auditoría radial Bhambare/Sukhatme."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def external_loss_closure_figure(curve: pd.DataFrame, targets: pd.DataFrame, scenarios: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["Tglass_K"], y=curve["Qloss_W"], mode="lines", name="Bloque externo actual"))
    for name in ("Modelo Python", "Bhambare", "Sukhatme"):
        r = scenarios.loc[scenarios["Escenario"] == name].iloc[0]
        fig.add_trace(go.Scatter(x=[r["Tglass_K"]], y=[r["Qloss_recalculado_W"]], mode="markers", name=f"{name} · Tglass"))
    for name in ("Bhambare", "Sukhatme"):
        r = targets.loc[targets["Referencia"] == name].iloc[0]
        fig.add_hline(y=float(r["Qloss_objetivo_W"]), line_dash="dot", annotation_text=f"Qloss {name}")
    fig.update_layout(
        title="¿El bloque externo reproduce Qloss si imponemos Tglass?",
        xaxis_title="Tglass (K)", yaxis_title="Pérdida externa (W)",
        legend_orientation="h", margin=dict(l=20, r=20, t=70, b=30),
    )
    return fig


def glass_balance_figure(scenarios: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=scenarios["Escenario"], y=scenarios["Qin_glass_W"], name="Entrada al vidrio"))
    fig.add_trace(go.Bar(x=scenarios["Escenario"], y=scenarios["Qout_glass_W"], name="Salida del vidrio"))
    fig.update_layout(
        title="Cierre del balance radial del vidrio con pares publicados",
        barmode="group", yaxis_title="Potencia (W)", legend_orientation="h",
        margin=dict(l=20, r=20, t=70, b=30),
    )
    return fig


def glass_residual_figure(scenarios: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=scenarios["Escenario"], y=scenarios["Residual_glass_pct_Qout"], name="Residual"))
    fig.add_hline(y=0.0, line_dash="dash")
    fig.update_layout(
        title="Residual del vidrio · (Qin − Qout) / Qout",
        yaxis_title="Residual (%)", margin=dict(l=20, r=20, t=70, b=30),
        showlegend=False,
    )
    return fig


def absorber_partition_figure(scenarios: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=scenarios["Escenario"], y=scenarios["Quseful_derivado_W"], name="Q útil derivado"))
    fig.add_trace(go.Bar(x=scenarios["Escenario"], y=scenarios["Qrad_abs_glass_W"], name="Q rad absorbedor→vidrio"))
    qsolar = float(scenarios["QsolarAbs_modelo_W"].iloc[0])
    fig.add_hline(y=qsolar, line_dash="dash", annotation_text="QsolarAbs del modelo")
    fig.update_layout(
        title="Partición cuasiestacionaria requerida en el absorbedor",
        barmode="stack", yaxis_title="Potencia (W)", legend_orientation="h",
        margin=dict(l=20, r=20, t=70, b=30),
    )
    return fig
