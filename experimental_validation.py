"""Validación con series de termopares, independiente de los presets documentales.

ÚNICAS observaciones usadas: DATE, TIME, termopar_1 (Tin), termopar_2
(Tout). La radiación, caudal y demás columnas de origen se descartan.
La irradiancia es una PREDICCIÓN de cielo claro; no es una medición.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, date, time
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ptc_model import PTCSimulator

BUNDLED_FOZ_CSV = Path(__file__).resolve().parent / "data" / "foz_2026_11_16_termopares.csv"
# Foz do Iguaçu. Coordenadas aproximadas del centro urbano, no del lugar del sensor.
FOZ_LATITUDE = -25.516
FOZ_LONGITUDE = -54.585
FOZ_UTC_OFFSET = -3.0


def _parse_datetime(date_value: Any, time_value: Any) -> datetime:
    if isinstance(date_value, datetime):
        d = date_value.date()
    elif isinstance(date_value, date):
        d = date_value
    else:
        raw_date = str(date_value).strip().split(" ")[0]
        d = None
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
            try:
                d = datetime.strptime(raw_date, fmt).date()
                break
            except ValueError:
                continue
        if d is None:
            raise ValueError(f"Fecha no reconocida: {date_value!r}")

    if isinstance(time_value, datetime):
        t = time_value.time()
    elif isinstance(time_value, time):
        t = time_value
    else:
        raw_time = str(time_value).strip()
        t = None
        for fmt in ("%H:%M:%S", "%H:%M", "%H:%M:%S.%f"):
            try:
                t = datetime.strptime(raw_time, fmt).time()
                break
            except ValueError:
                continue
        if t is None:
            raise ValueError(f"Hora no reconocida: {time_value!r}")
    return datetime.combine(d, t)


def _temperature(value: Any, title: str) -> float:
    try:
        out = float(str(value).strip().replace(",", "."))
    except (ValueError, TypeError):
        raise ValueError(f"Lectura no numérica en {title}: {value!r}") from None
    if not np.isfinite(out):
        raise ValueError(f"Lectura no finita en {title}: {value!r}")
    return out


def normalise_measurements(raw: pd.DataFrame) -> pd.DataFrame:
    """Conserva únicamente los instantes y las dos temperaturas identificadas.

    La columna mal escrita `temopar_1` del XLSX original está vacía y NO
    sustituye a `termopar_1`. Se admiten encabezados canónicos del CSV limpio.
    """
    labels = {str(col).strip().casefold(): col for col in raw.columns}
    aliases = {
        "date": ("date", "data"),
        "time": ("time", "hora"),
        "tin": ("termopar_1_c", "termopar_1"),
        "tout": ("termopar_2_c", "termopar_2"),
    }
    found = {}
    for role, choices in aliases.items():
        key = next((labels[c] for c in choices if c in labels), None)
        if key is None:
            raise ValueError(f"Falta columna para {role}: {', '.join(choices)}")
        found[role] = key

    points: list[tuple[datetime, float, float]] = []
    for index, row in raw.iterrows():
        fields = [row[found[k]] for k in ("date", "time", "tin", "tout")]
        if all(pd.isna(item) or str(item).strip() == "" for item in fields):
            continue
        if any(pd.isna(item) for item in fields):
            raise ValueError(f"DATE/TIME/T1/T2 incompletos en la fila {index + 2}")
        moment = _parse_datetime(fields[0], fields[1])
        points.append((moment, _temperature(fields[2], "termopar_1"), _temperature(fields[3], "termopar_2")))
    if len(points) < 2:
        raise ValueError("Se necesitan al menos dos registros con fecha, hora, T1 y T2.")
    points.sort(key=lambda item: item[0])
    if any(b[0] <= a[0] for a, b in zip(points, points[1:])):
        raise ValueError("Hay instantes de medición repetidos; revise las fechas y horas.")
    base_day = datetime.combine(points[0][0].date(), time.min)
    return pd.DataFrame({
        "datetime_local": [p[0] for p in points],
        "DATE": [p[0].strftime("%d/%m/%Y") for p in points],
        "TIME": [p[0].strftime("%H:%M:%S") for p in points],
        "t_s": [(p[0] - base_day).total_seconds() for p in points],
        "Tin_exp_C": [p[1] for p in points],
        "Tout_exp_C": [p[2] for p in points],
    })


def load_bundled_measurements() -> pd.DataFrame:
    # El CSV entregado contiene SOLO DATE/TIME y los dos termopares.
    raw = pd.read_csv(BUNDLED_FOZ_CSV, sep=",", encoding="utf-8-sig", dtype=str)
    return normalise_measurements(raw)


def load_measurements_upload(payload: bytes, filename: str) -> pd.DataFrame:
    """Importa un CSV/XLSX futuro sin usar radiación ni columnas adicionales."""
    filename = str(filename).lower()
    if filename.endswith(".xlsx"):
        # Dependencia openpyxl ya declarada por la aplicación para importar XLSX.
        raw = pd.read_excel(BytesIO(payload), dtype=str)
    elif filename.endswith(".csv"):
        raw = pd.read_csv(BytesIO(payload), sep=None, engine="python", dtype=str, encoding="utf-8-sig")
    else:
        raise ValueError("Formato no soportado; utilice CSV o XLSX.")
    return normalise_measurements(raw)


def apply_measurements(config: Mapping[str, Any], measurements: pd.DataFrame, *, source: str) -> dict[str, Any]:
    """Añade condiciones de borde T1(t) y referencias T2(t), sin calibrar parámetros."""
    df = measurements.copy()
    if df.empty:
        raise ValueError("La serie experimental está vacía.")
    cfg = deepcopy(dict(config))
    times = df["t_s"].astype(float).tolist()
    inlet = df["Tin_exp_C"].astype(float).tolist()
    outlet = df["Tout_exp_C"].astype(float).tolist()
    dates = df["DATE"].tolist()
    hours = df["TIME"].tolist()
    first_date = df.iloc[0]["datetime_local"].date()

    op = cfg["operation"]
    op["Tin_K"] = inlet[0] + 273.15
    op["Tin_profile"] = {"t_s": times, "Tin_K": [v + 273.15 for v in inlet]}
    op["output_times_s"] = times
    op["t_start_s"], op["t_end_s"] = times[0], times[-1]
    op["output_step_s"] = float(np.median(np.diff(times)))
    op["name"] = f"Validación experimental · {first_date.isoformat()}"

    solar = cfg["solar"]
    solar["date_origin"] = first_date.isoformat()
    solar["day_of_year"] = first_date.timetuple().tm_yday
    cfg["experimental_validation"] = {
        "source": source,
        "columns_used": ["DATE", "TIME", "termopar_1", "termopar_2"],
        "DATE": dates,
        "TIME": hours,
        "t_s": times,
        "Tin_exp_C": inlet,
        "Tout_exp_C": outlet,
        "timezone": "America/Sao_Paulo (UTC−03)",
    }
    cfg["preset_meta"] = {
        **cfg.get("preset_meta", {}),
        "family": "Experimental Foz termopares",
        "source": source,
        "date_label": f"{first_date.isoformat()} · {hours[0]}–{hours[-1]}",
        "city": "Foz do Iguaçu, PR, Brasil",
        "assumptions": [
            "T1(t) es la condición de entrada; T2(t) es exclusivamente la referencia de salida.",
            "Caudal, geometría, viento y temperatura ambiente son parámetros editables; no provienen de esta planilla.",
            "La radiación se calcula con cielo claro y geometría solar, sin leer el sensor del XLSX.",
            "La condición térmica inicial de los sólidos es supuesta; no se ajusta con T2.",
        ],
    }
    return cfg


def run_experimental_validation(config: Mapping[str, Any], fluid_database: Mapping[str, Any]) -> dict[str, Any]:
    """Predice T2 en cada instante medido y calcula errores sin ajuste de parámetros."""
    raw = config.get("experimental_validation")
    if not isinstance(raw, dict):
        raise ValueError("La configuración activa no contiene una serie de termopares.")
    result = PTCSimulator(config, fluid_database).simulate()
    times = np.asarray(raw["t_s"], dtype=float)
    if len(result.t_s) != len(times) or not np.allclose(result.t_s, times, atol=1e-6, rtol=0):
        raise RuntimeError("La salida del solver no coincide con la malla temporal experimental.")
    ref = np.asarray(raw["Tout_exp_C"], dtype=float)
    pred = np.asarray(result.Tout_C, dtype=float)
    tin = np.asarray(raw["Tin_exp_C"], dtype=float)
    resid = pred - ref
    metrics = {
        "n": int(len(times)),
        "RMSE_Tout_C": float(np.sqrt(np.mean(resid**2))),
        "MAE_Tout_C": float(np.mean(np.abs(resid))),
        "Bias_Tout_C": float(np.mean(resid)),
        "Max_abs_error_C": float(np.max(np.abs(resid))),
        "n_Tout_below_Tin": int(np.sum(ref < tin)),
    }
    reference = pd.DataFrame({
        "DATE": raw["DATE"],
        "TIME": raw["TIME"],
        "Tin_exp_C": tin,
        "Tout_exp_C": ref,
        "Tout_model_C": pred,
        "Error_model_minus_exp_C": resid,
        "DNI_synthetic_W_m2": result.scalar_diag["DNI_W_m2"],
        "Cos_theta_model": result.scalar_diag["cosTheta"],
        "G_aperture_model_W_m2": result.scalar_diag["G_aperture_W_m2"],
    })
    return {"table": reference, "result": result, "metrics": metrics,
            "note": "Validación comparativa de temperatura de salida; radiación de cielo claro estimada; caudal y ambiente editables, no medidos en la serie utilizada."}
