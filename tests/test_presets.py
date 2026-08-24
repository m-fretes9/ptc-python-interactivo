"""Pruebas de regresión de los presets documentales."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presets import build_preset
from ptc_model import PTCSimulator
from validations import validate_active_preset


def main() -> None:
    cfg, db = build_preset("rea_prototype")
    assert cfg["operation"]["fluid"] == "Agua"
    assert abs(cfg["operation"]["mdot"] - 0.0192) < 1e-12
    assert abs(cfg["solar"]["latitude_deg"] + 25.43816) < 1e-8
    assert cfg["solar"]["day_of_year"] == 296
    assert abs(cfg["geometry"]["W"] - 1.10) < 1e-12
    assert abs(cfg["geometry"]["L"] - 2.00) < 1e-12
    assert cfg["model"]["has_glass"] is False
    result = PTCSimulator(cfg, db).simulate()
    assert len(result.t_s) > 2
    validation = validate_active_preset(cfg, db)
    assert validation["kind"] == "prototype"

    cfg, db = build_preset("rea_foz_monthly", month=10)
    ref = cfg["preset_meta"]["reference"]
    assert abs(ref["DNI_W_m2"] - 157.03) < 1e-12
    assert abs(cfg["operation"]["mdot"] - 0.0064) < 1e-12
    assert abs(ref["eta_pct"] - 43.86) < 1e-12

    cfg, db = build_preset("rea_alvorada_monthly", month=8)
    ref = cfg["preset_meta"]["reference"]
    assert abs(cfg["solar"]["latitude_deg"] + 14.600) < 1e-12
    assert abs(ref["DNI_W_m2"] - 334.45) < 1e-12
    assert abs(ref["eta_pct"] - 33.91) < 1e-12

    cfg, db = build_preset("bhambare")
    assert cfg["operation"]["fluid"] == "ParathermNF"
    assert abs(cfg["operation"]["mdot"] - 0.0986) < 1e-12
    assert abs(cfg["operation"]["Tin_K"] - 423.15) < 1e-9
    assert cfg["model"]["has_glass"] is True
    print("Presets documentales correctos")


if __name__ == "__main__":
    main()
