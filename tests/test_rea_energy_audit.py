from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from defaults import default_fluid_database
from rea_energy_audit import audit_rea_monthly_energy_balance


def test_rea_energy_audit_two_months_has_consistent_balance():
    out = audit_rea_monthly_energy_balance(
        "Alvorada do Norte",
        default_fluid_database(),
        months=[1, 8],
    )
    table = out["table"]
    assert len(table) == 2
    assert {"Factor_optico_requerido", "Factor_perdidas_requerido", "Qloss_requerido_W"}.issubset(table.columns)
    assert out["metrics"]["max_storage_fraction_pct"] < 0.5
    assert out["metrics"]["max_balance_closure_W"] < 0.1
    assert (table["Qsolar_absorbida_W"] > 0).all()
    assert (table["Qloss_modelo_W"] > 0).all()
