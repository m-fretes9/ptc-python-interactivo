"""Prueba rápida sin interfaz gráfica."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from defaults import default_config, default_fluid_database
from ptc_model import PTCSimulator


def main() -> None:
    cfg = default_config()
    cfg["geometry"]["Nseg"] = 4
    cfg["operation"]["t_start_s"] = 8.0 * 3600.0
    cfg["operation"]["t_end_s"] = 8.2 * 3600.0
    cfg["operation"]["output_step_s"] = 60.0
    cfg["solver"]["max_step_s"] = 30.0
    result = PTCSimulator(cfg, default_fluid_database()).simulate()
    assert len(result.t_s) > 2
    assert result.y_K.shape[1] == 3 * cfg["geometry"]["Nseg"]
    assert result.Tout_C[-1] > -273.15
    print("Prueba correcta")
    print(f"Tout final = {result.Tout_C[-1]:.3f} °C")
    print(f"Qutil final = {result.scalar_diag['Quseful_W'][-1]:.3f} W")


if __name__ == "__main__":
    main()
