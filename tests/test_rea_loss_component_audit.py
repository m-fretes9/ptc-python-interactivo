from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from defaults import default_fluid_database
from rea_loss_component_audit import audit_rea_loss_components


def test_rea_loss_component_audit_two_months_is_consistent():
    out = audit_rea_loss_components(
        "Foz do Iguaçu",
        default_fluid_database(),
        months=[1, 7],
    )
    table = out["table"]
    assert len(table) == 2
    assert {
        "Factor_conv_requerido",
        "Factor_rad_requerido",
        "Eta_kconv_global_pct",
        "Eta_krad_global_pct",
    }.issubset(table.columns)
    assert (table["Qconv_ext_W"] > 0).all()
    assert (table["Qrad_cielo_W"] > 0).all()
    assert out["metrics"]["max_storage_fraction_pct"] < 0.5
    assert out["metrics"]["kconv_global_ls"] > 0
    assert out["metrics"]["krad_global_ls"] > 0
