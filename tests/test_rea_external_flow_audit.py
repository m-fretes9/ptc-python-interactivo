from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from defaults import default_fluid_database
from rea_external_flow_audit import audit_rea_external_flow


def test_external_flow_audit_two_months_has_required_diagnostics():
    out = audit_rea_external_flow(
        "Foz do Iguaçu",
        default_fluid_database(),
        months=[1, 7],
    )
    table = out["table"]
    assert len(table) == 2
    expected = {
        "Tfilm_media_C",
        "Re_modelo",
        "Nu_modelo",
        "h_modelo_W_m2K",
        "h_requerido_W_m2K",
        "Viento_requerido_m_s",
    }
    assert expected.issubset(table.columns)
    assert (table["h_modelo_W_m2K"] > 0).all()
    assert (table["Re_modelo"] > 0).all()
    assert np.isfinite(table["Factor_h_requerido"]).all()
    assert out["metrics"]["current_wind_mean_m_s"] > 0
