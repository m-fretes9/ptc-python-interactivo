from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from defaults import default_fluid_database
from rea_eq10_validation import validate_rea_eq10_from_tout


def test_rea_eq10_simple_alvorada_diagnostic():
    out = validate_rea_eq10_from_tout("Alvorada do Norte", default_fluid_database())
    assert len(out["table"]) == 12
    # La tabla publicada se reconstruye casi exactamente con su propia Ec. (10).
    assert out["metrics"]["max_reference_eq10_mismatch_pp"] < 0.1
    # En el preset mensual actual la eta interna y la reconstruida desde Tout
    # son prácticamente la misma: el diagnóstico debe dejarlo explícito.
    assert out["metrics"]["max_python_native_eq10_mismatch_pp"] < 0.01
