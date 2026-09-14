import pandas as pd

from visualizations import (
    validation_holdout_error_figure,
    validation_monthly_comparison_figure,
    validation_prediction_scatter_figure,
    validation_signed_residual_figure,
)


def _comparison():
    return pd.DataFrame([
        {"Conjunto":"calibración","Caso":"Ene","Magnitud":"Eta_pct","Referencia":47.6,"Modelo_inicial":46.1,"Error_inicial_pct":-3.2,"Residual_inicial":-0.03,"Modelo_calibrado":46.6,"Error_calibrado_pct":-2.2,"Residual_calibrado":-0.02},
        {"Conjunto":"validación","Caso":"Feb","Magnitud":"Eta_pct","Referencia":46.2,"Modelo_inicial":43.6,"Error_inicial_pct":-5.6,"Residual_inicial":-0.06,"Modelo_calibrado":45.6,"Error_calibrado_pct":-1.5,"Residual_calibrado":-0.01},
        {"Conjunto":"calibración","Caso":"Ene","Magnitud":"Tout_C","Referencia":41.9,"Modelo_inicial":41.8,"Error_inicial_pct":-0.4,"Residual_inicial":-0.03,"Modelo_calibrado":41.8,"Error_calibrado_pct":-0.3,"Residual_calibrado":-0.02},
        {"Conjunto":"validación","Caso":"Feb","Magnitud":"Tout_C","Referencia":40.9,"Modelo_inicial":40.5,"Error_inicial_pct":-1.0,"Residual_inicial":-0.06,"Modelo_calibrado":40.8,"Error_calibrado_pct":-0.3,"Residual_calibrado":-0.02},
    ])


def test_validation_graphs_build_from_comparison_table():
    df = _comparison()
    monthly = validation_monthly_comparison_figure(df, "Eta_pct")
    errors = validation_holdout_error_figure(df, "Eta_pct")
    scatter = validation_prediction_scatter_figure(df, "Tout_C")
    residual = validation_signed_residual_figure(df, "Tout_C")
    assert len(monthly.data) == 3
    assert len(errors.data) == 2
    assert len(scatter.data) == 3  # inicial, calibrado, y=x
    assert len(residual.data) == 1
