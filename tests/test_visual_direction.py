"""Regresión visual: el sentido del calor absorbedor->HTF no debe invertirse."""
from pathlib import Path
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presets import build_preset
from ptc_model import PTCSimulator
from interactive_visuals import thermal_circuit_component_html

cfg, db = build_preset("rea_prototype")
result = PTCSimulator(cfg, db).simulate()
snapshot = result.node_snapshot(len(result.t_s) - 1, 0)
assert snapshot["Tabs_C"] > snapshot["Tf_C"]
assert snapshot["Qfluid_W"] > 0.0
html = thermal_circuit_component_html(cfg, snapshot)
match = re.search(r"const D=(\{.*?\}), svg=", html, re.S)
assert match, "No se encontró el bloque de datos del circuito SVG."
data = json.loads(match.group(1))
assert data["Q"]["q12dir"] < 0.0
assert data["Q"]["q23dir"] < 0.0
assert abs(data["Q"]["q12"] - snapshot["Qfluid_W"]) < 1e-9
print("Dirección visual correcta: absorbedor -> HTF cuando Qfluid > 0.")
