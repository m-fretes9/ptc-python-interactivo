"""Prueba directa de las Ecs. (12)-(14) de Rea Quille."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ptc_model import effective_sky_temperature


def main() -> None:
    env = {
        "Tamb_K": 298.15,
        "sky_model": "rea_quille",
        "dew_point_C": 15.0,
        "cloud_adjustment": False,
        "cloud_factor": 0.0,
        "cloud_emissivity": 1.0,
        "cloud_formula": "rea_quille_impresa",
        "pressure_Pa": 101325.0,
    }
    time_s = 12.0 * 3600.0
    got = effective_sky_temperature(time_s, env)
    P_mbar = env["pressure_Pa"] / 100.0
    expected_eps = (
        0.711
        + 0.56 * (15.0 / 100.0)
        + 0.73 * (15.0 / 100.0) ** 2
        + 0.013 * np.cos(2.0 * np.pi * 12.0 / 24.0)
        + (0.012 / 100.0) * (P_mbar - 1000.0)
    )
    expected_tsky = expected_eps ** 0.25 * env["Tamb_K"]
    assert abs(got["eps_clear"] - expected_eps) < 1e-12
    assert abs(got["Tsky_K"] - expected_tsky) < 1e-12
    print(f"Modelo Tsky correcto: eps={got['eps_sky']:.6f}, Tsky={got['Tsky_K']-273.15:.3f} °C")


if __name__ == "__main__":
    main()
