from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from defaults import default_fluid_database
from validations import (
    bhambare_user_inverse_template,
    identified_parameter_template,
    validate_bhambare_mode,
)


def test_bhambare_identified_template_values():
    spec = identified_parameter_template("bhambare")
    assert abs(spec["eta_opt_eff"] - 0.7671242147795794) < 1e-12
    assert abs(spec["eps_abs"] - 0.989999998086766) < 1e-12
    assert abs(spec["eps_glass"] - 0.9899999947177927) < 1e-12
    assert set(spec["near_bounds"]) == {"eta_opt_eff", "eps_abs", "eps_glass"}


def test_bhambare_xlsx_initial_regression():
    out = validate_bhambare_mode(
        default_fluid_database(),
        target_key="xlsx_initial",
        parameter_template="nominal",
    )
    # El target fue generado con BDF; el modo XLSX usa el mismo integrador.
    assert out["metrics"]["RMSRE_pct"] < 1e-6
    assert out["metrics"]["Error_max_pct"] < 1e-6


def test_bhambare_xlsx_identified_regression():
    out = validate_bhambare_mode(
        default_fluid_database(),
        target_key="xlsx_identified",
        parameter_template="identified",
    )
    assert out["metrics"]["RMSRE_pct"] < 1e-5
    assert out["metrics"]["Error_max_pct"] < 1e-5


def test_bhambare_template_tables():
    template = bhambare_user_inverse_template()
    assert len(template["parameter_table"]) == 3
    assert len(template["comparison_table"]) == 4
