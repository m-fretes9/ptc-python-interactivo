from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from defaults import default_fluid_database
from validations import inverse_parameter_options


def test_inverse_parameter_options_cover_all_documental_cases():
    db = default_fluid_database()
    for case in ("bhambare", "rea_foz", "rea_alvorada", "rea_prototype"):
        options = inverse_parameter_options(case, db)
        assert "eta_opt_eff" in options
        lo, hi = options["eta_opt_eff"]["bounds"]
        nominal = options["eta_opt_eff"]["nominal"]
        assert 0.0 < lo < nominal < hi <= 1.0


def test_glass_emissivity_only_available_with_glass_case():
    db = default_fluid_database()
    assert "eps_glass" in inverse_parameter_options("bhambare", db)
    assert "eps_glass" not in inverse_parameter_options("rea_foz", db)
