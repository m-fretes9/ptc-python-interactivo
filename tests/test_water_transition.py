"""Regresión: la transición de Reynolds del agua no debe crear saltos de h."""
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from defaults import default_config, default_fluid_database
from ptc_model import PTCSimulator

cfg = default_config()
cfg["operation"]["fluid"] = "Agua"
cfg["operation"]["mdot"] = 0.045
result = PTCSimulator(deepcopy(cfg), default_fluid_database()).simulate()

h_out = result.node_diag["h_internal_W_m2K"][:, -1]
max_jump = float(np.max(np.abs(np.diff(h_out))))
assert max_jump < 5.0, f"Salto artificial de h detectado: {max_jump:.3f} W/(m2 K)"

re_out = result.node_diag["Re_internal"][:, -1]
assert float(np.nanmax(re_out)) > 2300.0, "El caso de regresión no cruza la transición."
print(f"Transición suave correcta. max |Δh| = {max_jump:.3f} W/(m2 K)")
