"""Gráficos aislados para la auditoría de pérdidas Rea Quille.

Se mantienen fuera de ``visualizations.py`` para evitar que un despliegue con una
copia antigua/cacheada de ese módulo impida iniciar toda la aplicación.
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
import plotly.graph_objects as go


def rea_loss_component_magnitude_figure(table: pd.DataFrame) -> go.Figure:
    """Muestra cuánto pesan convección/radiación frente a la corrección útil requerida."""
    figure = go.Figure()
    if not isinstance(table, pd.DataFrame) or table.empty:
        figure.update_layout(title="Sin datos para la auditoría de pérdidas")
        return figure
    x = table["Mes"].astype(str)
    figure.add_trace(
        go.Bar(
            x=x,
            y=table["Qconv_ext_W"],
            name="Convección externa",
            hovertemplate="%{x}<br>Qconv = %{y:.2f} W<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=x,
            y=table["Qrad_cielo_W"],
            name="Radiación al cielo",
            hovertemplate="%{x}<br>Qrad = %{y:.2f} W<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=table["Delta_Qutil_requerida_W"],
            mode="lines+markers",
            name="ΔQ útil requerido",
            line={"width": 3},
            hovertemplate="%{x}<br>Qref − Qmodelo = %{y:+.2f} W<extra></extra>",
        )
    )
    figure.add_hline(y=0.0, line_dash="dash")
    figure.update_layout(
        title="Magnitud disponible para corregir el déficit de potencia útil",
        xaxis_title="Mes",
        yaxis_title="Potencia (W)",
        barmode="group",
        height=450,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    return figure


def rea_loss_component_factors_figure(table: pd.DataFrame) -> go.Figure:
    """Factores requeridos si sólo se corrige convección o sólo radiación."""
    figure = go.Figure()
    if not isinstance(table, pd.DataFrame) or table.empty:
        figure.update_layout(title="Sin datos para la auditoría de pérdidas")
        return figure
    x = table["Mes"].astype(str)
    figure.add_hline(y=1.0, line_dash="dash", annotation_text="Sin corrección")
    figure.add_hline(y=0.0, line_dash="dot", annotation_text="Límite físico: pérdida nula")
    figure.add_trace(
        go.Scatter(
            x=x,
            y=table["Factor_conv_requerido"],
            mode="lines+markers",
            name="k conv requerido",
            hovertemplate="%{x}<br>kconv = %{y:.3f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=table["Factor_rad_requerido"],
            mode="lines+markers",
            name="k rad requerido",
            hovertemplate="%{x}<br>krad = %{y:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="¿Qué mecanismo puede absorber por sí solo la corrección?",
        xaxis_title="Mes",
        yaxis_title="Factor multiplicativo requerido (-)",
        height=450,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    return figure


def rea_loss_component_counterfactual_figure(table: pd.DataFrame, metrics: Mapping[str, Any]) -> go.Figure:
    """Compara la forma mensual con una corrección global de cada pérdida por separado."""
    figure = go.Figure()
    if not isinstance(table, pd.DataFrame) or table.empty:
        figure.update_layout(title="Sin datos para la auditoría de pérdidas")
        return figure
    x = table["Mes"].astype(str)
    traces = [
        ("Eta_ref_pct", "Referencia"),
        ("Eta_modelo_pct", "Modelo actual"),
        ("Eta_kconv_global_pct", f"Sólo convección · k={metrics.get('kconv_global_ls', float('nan')):.3f}"),
        ("Eta_krad_global_pct", f"Sólo radiación · k={metrics.get('krad_global_ls', float('nan')):.3f}"),
    ]
    for column, label in traces:
        figure.add_trace(
            go.Scatter(
                x=x,
                y=table[column],
                mode="lines+markers",
                name=label,
                hovertemplate=f"%{{x}}<br>{label} = %{{y:.2f}} %<extra></extra>",
            )
        )
    figure.update_layout(
        title="Contrafactual: una sola corrección global de convección o radiación",
        xaxis_title="Mes",
        yaxis_title="Eficiencia térmica (%)",
        height=470,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    return figure
