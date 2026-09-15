"""Presets completos y referencias documentales para el modelo PTC.

Los presets separan valores publicados de hipótesis necesarias para completar
parámetros que las fuentes no informan. Los valores asumidos se exponen en
``preset_meta['assumptions']`` y nunca se presentan como datos documentales.
"""

from __future__ import annotations

from calendar import monthrange
from copy import deepcopy
from datetime import date
from typing import Any

from defaults import default_config, default_fluid_database


MONTH_NAMES_ES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]
MONTH_ABBR_ES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

# Tabla 10 — Rea Quille, Foz do Iguaçu, PTC.
REA_FOZ_MONTHLY = {
    "Tin_C": [34.73, 34.06, 34.17, 34.02, 32.89, 32.71, 32.31, 33.07, 31.74, 32.57, 34.60, 34.92],
    "Tout_ref_C": [41.95, 40.91, 41.25, 40.71, 38.57, 38.11, 37.57, 38.46, 36.07, 38.25, 41.96, 42.48],
    "Tamb_C": [27.46, 25.13, 24.27, 24.56, 20.24, 18.79, 17.14, 17.80, 18.33, 23.93, 24.12, 25.83],
    "DNI_W_m2": [252.38, 224.26, 228.56, 224.70, 174.61, 164.10, 146.87, 181.28, 120.74, 157.03, 247.59, 262.18],
    "mdot_kg_s": [0.0087, 0.0080, 0.0082, 0.0076, 0.0063, 0.0065, 0.0060, 0.0067, 0.0054, 0.0064, 0.0087, 0.0090],
    "eta_ref_pct": [47.63, 46.25, 48.14, 43.04, 38.97, 40.55, 40.54, 37.98, 36.50, 43.86, 49.25, 49.41],
    "annual": {"Tin_C": 33.48, "Tout_ref_C": 39.69, "Tamb_C": 22.30, "DNI_W_m2": 198.69, "mdot_kg_s": 0.0073, "eta_ref_pct": 43.27},
}

# Tabla 11 — Rea Quille, Alvorada do Norte, PTC.
REA_ALVORADA_MONTHLY = {
    "Tin_C": [32.60, 33.43, 32.87, 33.30, 35.56, 34.93, 35.37, 36.40, 34.66, 34.23, 33.32, 33.23],
    "Tout_ref_C": [38.60, 39.80, 39.21, 39.62, 43.78, 42.66, 43.42, 45.01, 42.36, 41.47, 39.72, 39.59],
    "Tamb_C": [23.97, 23.72, 23.30, 23.66, 23.28, 22.67, 22.57, 24.15, 27.21, 28.58, 24.32, 24.11],
    "DNI_W_m2": [153.26, 192.31, 165.87, 188.71, 294.57, 263.57, 284.16, 334.45, 250.41, 228.46, 186.76, 182.83],
    "mdot_kg_s": [0.00692, 0.00692, 0.00692, 0.00692, 0.00692, 0.00692, 0.00692, 0.00692, 0.00695, 0.00695, 0.00695, 0.00695],
    "eta_ref_pct": [51.55, 43.63, 50.36, 44.14, 36.77, 38.62, 37.37, 33.91, 40.68, 41.91, 45.29, 46.03],
    "annual": {"Tin_C": 34.16, "Tout_ref_C": 41.27, "Tamb_C": 24.29, "DNI_W_m2": 227.11, "mdot_kg_s": 0.00693, "eta_ref_pct": 41.31},
}

REA_PROTOTYPE_HOURS = {
    "hours": list(range(9, 17)),
    "mdot_kg_s": [0.0192] * 8,
    "eta_exp_pct": [28.70, 34.40, 26.51, 30.06, 27.50, 30.01, 28.60, 27.10],
    "eta_trnsys_pct": [27.400, 27.300, 26.500, 26.200, 35.300, 32.900, 32.100, 30.000],
}

# Exportación entregada por el usuario el 14/09/2026 (V10).
# Se conserva como curva benchmark de salida, NO como conjunto de parámetros de entrada:
# el CSV no contiene los parámetros identificados que produjeron esas predicciones.
REA_PROTOTYPE_USER_EXPORT_2026_09_14 = {
    "hours": list(range(9, 17)),
    "eta_reference_pct": [28.70, 34.40, 26.51, 30.06, 27.50, 30.01, 28.60, 27.10],
    "eta_model_initial_pct": [
        54.458875343535006, 54.45119066969734, 54.44631167069509, 54.44457088402858,
        54.44608696022283, 54.45075657060333, 54.45826144306864, 54.46809007699063,
    ],
    "eta_model_identified_pct": [
        28.882567305668545, 28.876696305769354, 28.87297278911823, 28.871650510011225,
        28.87281958360406, 28.876400340231577, 28.88214874860653, 28.889673058221174,
    ],
}


def _midmonth_doy(month: int, year: int = 2021) -> int:
    """Día representativo solo para metadatos; no es dato de Rea Quille."""
    day = min(15, monthrange(year, month)[1])
    return date(year, month, day).timetuple().tm_yday


def _apply_rea_geometry(cfg: dict[str, Any]) -> None:
    cfg["geometry"].update(
        {
            "W": 1.10,
            "L": 2.00,
            "f": 0.16,
            "D2": 0.039,
            "D3": 0.042,
            # No existe vidrio en el prototipo físico. D4/D5 quedan definidos
            # únicamente porque el esquema de configuración requiere todos los campos.
            "D4": 0.050,
            "D5": 0.055,
            "Nseg": 12,
        }
    )
    cfg["materials"]["absorber"].update(
        {
            # El TCC identifica tubo de cobre en la comparación del prototipo.
            # Las propiedades termofísicas son valores de ingeniería del modelo,
            # ya que Rea Quille no las tabula.
            "rho": 8933.0,
            "Cp": 385.0,
            "k": 385.0,
            "alpha": 0.97,
            "eps": 0.90,
        }
    )
    cfg["model"].update(
        {
            "has_glass": False,
            "annulus": "vacio_ideal",
            "internal_correlation": "automatica",
            "Re_laminar_max": 2300.0,
            "Re_turbulent_min": 4000.0,
            "include_supports": False,
        }
    )
    cfg["optics"].update(
        {
            # Rea Quille no publica rho ni gamma efectivos del prototipo.
            # Se mantienen como hipótesis explícitas y editables, no como datos fuente.
            "reflectivity": 0.85,
            "intercept_factor": 0.95,
            "dirt_factor": 1.0,
            "shade_factor": 1.0,
            "tracking": "Fijo / eje N-S (sin tracking en la reproducción TRNSYS)",
        }
    )
    cfg["environment"].update(
        {
            "sky_model": "rea_quille",
            "dew_point_C": 15.0,
            "cloud_adjustment": False,
            "cloud_factor": 0.0,
            "cloud_emissivity": 1.0,
            "cloud_formula": "rea_quille_impresa",
            "sky_delta_K": 6.0,  # legado; no se usa mientras sky_model=rea_quille
            "wind_m_s": 1.0,
            "pressure_Pa": 101325.0,
        }
    )
    cfg["solver"].update({"rtol": 1e-6, "atol": 1e-7, "max_step_s": 20.0, "use_jac_sparsity": True})


def _rea_assumptions(monthly: bool = False) -> list[str]:
    items = [
        "Reflectividad efectiva 0.85 e interceptación 0.95: no están tabuladas por Rea Quille; son hipótesis editables heredadas del modelo base.",
        "Viento 1.0 m/s: la tabla de Rea Quille no publica este valor; se conserva únicamente como baseline del preset. En validación mensual el viento se trata como entrada meteorológica independiente y no como parámetro calibrable del colector.",
        "Temperatura de cielo: se implementan las Ecs. (12)-(14) de Rea Quille/Martin-Berdahl. Las tablas 8, 10 y 11 no publican Tdp, f_nuvem ni epsilon_nuvem; el preset usa Tdp=15 °C y cielo claro (sin corrección de nubosidad) como hipótesis explícita editable.",
        "Propiedades rho, Cp y k del cobre: valores de ingeniería del modelo; el TCC identifica tubo de cobre pero no tabula estas propiedades.",
        "D4/D5 se conservan solo por compatibilidad de estructura; has_glass=False y no participan del circuito físico del prototipo.",
    ]
    if monthly:
        items.append(
            "El trabajo publica promedios mensuales y no un día específico. El día 15 se guarda solo como metadato y no afecta la simulación porque el preset usa irradiación constante igual al valor mensual tabulado."
        )
        items.append(
            "El valor mensual de radiación/DNI se trata como irradiancia efectiva constante con ángulo 0° para comparar el punto medio mensual; no reproduce el archivo meteorológico Type15 del TRNSYS."
        )
    return items


def build_rea_monthly_preset(city: str, month: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Preset completo para una fila mensual de las Tablas 10 u 11."""
    if not 1 <= month <= 12:
        raise ValueError("month debe estar entre 1 y 12")
    city_key = city.strip().lower()
    if city_key in {"foz", "foz do iguacu", "foz do iguaçu"}:
        data = REA_FOZ_MONTHLY
        city_name = "Foz do Iguaçu"
        lat, lon = -25.43816, -54.59679
        table = "Tabela 10"
    elif city_key in {"alvorada", "alvorada do norte"}:
        data = REA_ALVORADA_MONTHLY
        city_name = "Alvorada do Norte"
        lat, lon = -14.600, -46.649
        table = "Tabela 11"
    else:
        raise ValueError(f"Ciudad Rea Quille desconocida: {city}")

    i = month - 1
    cfg = default_config()
    _apply_rea_geometry(cfg)
    cfg["operation"].update(
        {
            "name": f"Rea Quille — {city_name} — {MONTH_NAMES_ES[i]}",
            "fluid": "Agua",
            "mdot": float(data["mdot_kg_s"][i]),
            "Tin_K": float(data["Tin_C"][i] + 273.15),
            "t_start_s": 8.0 * 3600.0,
            "t_end_s": 12.0 * 3600.0,
            "output_step_s": 60.0,
        }
    )
    cfg["environment"]["Tamb_K"] = float(data["Tamb_C"][i] + 273.15)
    cfg["solar"].update(
        {
            "mode": "constante",
            "DNI_constant_W_m2": float(data["DNI_W_m2"][i]),
            "angle_constant_deg": 0.0,
            "latitude_deg": lat,
            "longitude_deg": lon,
            "day_of_year": _midmonth_doy(month),
        }
    )
    cfg["preset_meta"] = {
        "id": f"rea_{'foz' if city_name.startswith('Foz') else 'alvorada'}_{month:02d}",
        "family": "Rea Quille mensual",
        "source": "Washington Adrian Rea Quille (2025), Análise e comparação do desempenho térmico de coletores solares planos e parabólicos no TRNSYS",
        "reference_table": table,
        "city": city_name,
        "month": MONTH_NAMES_ES[i],
        "date_label": f"Promedio mensual — {MONTH_NAMES_ES[i]}",
        "latitude_deg": lat,
        "longitude_deg": lon,
        "strict_reference": False,
        "reference": {
            "Tin_C": float(data["Tin_C"][i]),
            "Tout_C": float(data["Tout_ref_C"][i]),
            "Tamb_C": float(data["Tamb_C"][i]),
            "DNI_W_m2": float(data["DNI_W_m2"][i]),
            "mdot_kg_s": float(data["mdot_kg_s"][i]),
            "eta_pct": float(data["eta_ref_pct"][i]),
        },
        "assumptions": _rea_assumptions(monthly=True),
    }
    return cfg, default_fluid_database()


def build_rea_annual_preset(city: str) -> tuple[dict[str, Any], dict[str, Any]]:
    city_key = city.strip().lower()
    if city_key in {"foz", "foz do iguacu", "foz do iguaçu"}:
        data = REA_FOZ_MONTHLY["annual"]
        city_name = "Foz do Iguaçu"
        lat, lon = -25.43816, -54.59679
        table = "Tabela 10 — Total"
    elif city_key in {"alvorada", "alvorada do norte"}:
        data = REA_ALVORADA_MONTHLY["annual"]
        city_name = "Alvorada do Norte"
        lat, lon = -14.600, -46.649
        table = "Tabela 11 — Total"
    else:
        raise ValueError(f"Ciudad Rea Quille desconocida: {city}")

    cfg = default_config()
    _apply_rea_geometry(cfg)
    cfg["operation"].update(
        {
            "name": f"Rea Quille — {city_name} — promedio anual",
            "fluid": "Agua",
            "mdot": float(data["mdot_kg_s"]),
            "Tin_K": float(data["Tin_C"] + 273.15),
            "t_start_s": 8.0 * 3600.0,
            "t_end_s": 12.0 * 3600.0,
            "output_step_s": 60.0,
        }
    )
    cfg["environment"]["Tamb_K"] = float(data["Tamb_C"] + 273.15)
    cfg["solar"].update(
        {
            "mode": "constante",
            "DNI_constant_W_m2": float(data["DNI_W_m2"]),
            "angle_constant_deg": 0.0,
            "latitude_deg": lat,
            "longitude_deg": lon,
            "day_of_year": 183,
        }
    )
    cfg["preset_meta"] = {
        "id": f"rea_{'foz' if city_name.startswith('Foz') else 'alvorada'}_annual",
        "family": "Rea Quille anual",
        "source": "Washington Adrian Rea Quille (2025), Análise e comparação do desempenho térmico de coletores solares planos e parabólicos no TRNSYS",
        "reference_table": table,
        "city": city_name,
        "month": "Promedio anual",
        "date_label": "Promedio anual",
        "latitude_deg": lat,
        "longitude_deg": lon,
        "strict_reference": False,
        "reference": {
            "Tin_C": float(data["Tin_C"]),
            "Tout_C": float(data["Tout_ref_C"]),
            "Tamb_C": float(data["Tamb_C"]),
            "DNI_W_m2": float(data["DNI_W_m2"]),
            "mdot_kg_s": float(data["mdot_kg_s"]),
            "eta_pct": float(data["eta_ref_pct"]),
        },
        "assumptions": _rea_assumptions(monthly=True),
    }
    return cfg, default_fluid_database()


def build_rea_prototype_preset(
    validation_mode: str = "trnsys_published",
    dni_source: str = "nominal",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Preset del prototipo Fiamonzini/Rea Quille con dos interpretaciones.

    ``trnsys_published`` reproduce la idealización declarada por Rea Quille:
    DNI nominal 905 W/m², IAM=1 y sin tracking. ``physical_fixed_ns`` mantiene
    el colector fijo y usa la geometría solar para calcular el ángulo de
    incidencia/IAM. En este segundo modo el DNI puede mantenerse nominal o
    calcularse con el modelo claro A·exp(-B/cos(z)); no existe una serie DNI
    medida en la Tabela 8.
    """
    mode = str(validation_mode).strip().lower()
    if mode not in {"trnsys_published", "physical_fixed_ns"}:
        raise ValueError(f"Modo de validación del prototipo desconocido: {validation_mode}")
    dni_key = str(dni_source).strip().lower()
    if dni_key not in {"nominal", "clear_sky", "temporal_clear_sky_905"}:
        raise ValueError(f"Fuente DNI del prototipo desconocida: {dni_source}")

    cfg = default_config()
    _apply_rea_geometry(cfg)
    cfg["operation"].update(
        {
            "name": "Rea Quille / Fiamonzini — prototipo 23/10/2021",
            "fluid": "Agua",
            "mdot": 0.0192,
            # La Tabla 8 NO publica Tin. Se usa Tamb como hipótesis inicial explícita.
            "Tin_K": 25.0 + 273.15,
            "t_start_s": 8.0 * 3600.0,
            "t_end_s": 16.0 * 3600.0,
            "output_step_s": 60.0,
        }
    )
    cfg["environment"].update({"Tamb_K": 25.0 + 273.15, "wind_m_s": 1.0})

    if mode == "trnsys_published":
        cfg["solar"].update(
            {
                "mode": "constante",
                "DNI_constant_W_m2": 905.0,
                "angle_constant_deg": 0.0,
                "latitude_deg": -25.43816,
                "longitude_deg": -54.59679,
                "day_of_year": 296,
                "fixed_dni_source": "nominal",
            }
        )
        cfg["optics"]["tracking"] = "Mode 1 / fixed surface; IAM=1 en la idealización TRNSYS reportada"
        mode_label = "TRNSYS publicado · DNI=905 · IAM=1"
    else:
        cfg["solar"].update(
            {
                "mode": "fijo_horizontal",
                "DNI_constant_W_m2": 905.0,
                "fixed_dni_source": dni_key,
                "latitude_deg": -25.43816,
                "longitude_deg": -54.59679,
                "day_of_year": 296,
                "utc_offset_h": -3.0,
                "clock_time_correction": True,
                "DNI_noon_W_m2": 905.0,
                # A/B permanecen editables. Si se elige clear_sky son un modelo,
                # no datos medidos del ensayo.
                "A": float(cfg["solar"].get("A", 713.35)),
                "B": float(cfg["solar"].get("B", 0.131)),
            }
        )
        cfg["optics"]["tracking"] = "Fijo / eje N-S; incidencia e IAM calculados con geometría solar"
        if dni_key == "nominal":
            mode_label = "Físico corregido · fijo N-S · IAM variable · DNI nominal"
        elif dni_key == "temporal_clear_sky_905":
            mode_label = "Físico corregido · fijo N-S · DNI temporal (905 al mediodía) · IAM variable"
        else:
            mode_label = "Físico corregido · fijo N-S · IAM y DNI claro variables"

    assumptions = _rea_assumptions(monthly=False) + [
        "Tin no aparece en la Tabela 8. El preset usa Tin=Tamb=25 °C únicamente como hipótesis inicial editable; por este motivo la salida Python no constituye una validación estricta de las ocho eficiencias experimentales.",
    ]
    if mode == "trnsys_published":
        assumptions.append(
            "Reproducción del procedimiento declarado por Rea Quille: DNI nominal 905 W/m², superficie fija y IAM=1. Se representa con DNI constante y θ=0° para conservar exactamente esa idealización óptica."
        )
    else:
        assumptions.append(
            "Modo físico exploratorio: el colector permanece fijo con apertura horizontal y eje N-S; θ se calcula como el ángulo cenital e IAM/EndLoss varían con la hora. El trabajo no publica una serie DNI medida."
        )
        if dni_key == "nominal":
            assumptions.append(
                "DNI se mantiene en 905 W/m² mientras el Sol está sobre el horizonte. Así se aísla el efecto geométrico/IAM sin inventar una meteorología horaria."
            )
        elif dni_key == "temporal_clear_sky_905":
            assumptions.append(
                "DNI varía temporalmente con A·exp(-B/cos z), reescalado para valer 905 W/m² al mediodía solar. Es un perfil físico sintético, no una medición horaria del ensayo."
            )
            assumptions.append(
                "Las horas de la Tabla 8 se interpretan como hora civil UTC−3 y se convierten a hora solar aparente usando longitud y ecuación del tiempo."
            )
        else:
            assumptions.append(
                "DNI variable se estima con A·exp(-B/cos z). A y B son parámetros editables del modelo de cielo claro y NO mediciones del ensayo de Fiamonzini/Rea Quille."
            )

    cfg["preset_meta"] = {
        "id": f"rea_prototype_2021_10_23_{mode}_{dni_key}",
        "family": "Rea Quille / Fiamonzini prototipo",
        "source": "Rea Quille (2025), Tabela 8; prototipo de Fiamonzini (2022)",
        "reference_table": "Tabela 8",
        "city": "Foz do Iguaçu",
        "date_label": "23/10/2021, 09:00–16:00",
        "latitude_deg": -25.43816,
        "longitude_deg": -54.59679,
        "strict_reference": False,
        "prototype_validation_mode": mode,
        "prototype_validation_mode_label": mode_label,
        "prototype_dni_source": dni_key,
        "reference": {
            "Tamb_C": 25.0,
            "DNI_W_m2": 905.0,
            "mdot_kg_s": 0.0192,
            "eta_exp_mean_pct": sum(REA_PROTOTYPE_HOURS["eta_exp_pct"]) / 8.0,
            "eta_trnsys_mean_pct": sum(REA_PROTOTYPE_HOURS["eta_trnsys_pct"]) / 8.0,
            "eta_exp_max_pct": max(REA_PROTOTYPE_HOURS["eta_exp_pct"]),
            "eta_collector_max_reported_pct": 36.50,
        },
        "assumptions": assumptions,
    }
    return cfg, default_fluid_database()


def build_bhambare_sukhatme_preset() -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = default_config()
    cfg["geometry"].update(
        {
            "L": 3.657,
            "W": 1.25,
            "f": 0.60,
            "D2": 0.03810,
            "D3": 0.04135,
            "D4": 0.05600,
            "D5": 0.06300,
            "Nseg": 12,
        }
    )
    cfg["materials"]["absorber"].update({"rho": 8933.0, "Cp": 385.0, "k": 385.0, "eps": 0.95, "alpha": 0.95})
    cfg["materials"]["glass"].update({"rho": 4500.0, "Cp": 840.0, "k": 1.2, "eps": 0.88, "tau": 0.85, "alpha": 0.02})
    cfg["optics"].update(
        {
            "reflectivity": 0.85,
            "intercept_factor": 0.95,
            "dirt_factor": 1.0,
            "shade_factor": 1.0,
            "tracking": "N-S horizontal, un eje",
        }
    )
    cfg["environment"].update({
        "Tamb_K": 31.9 + 273.15,
        "sky_model": "delta_constante",
        "sky_delta_K": 6.0,
        "wind_m_s": 5.3,
        "pressure_Pa": 101325.0,
    })
    cfg["solar"].update(
        {
            "mode": "constante",
            "A": 713.35,
            "B": 0.131,
            "day_of_year": 105,
            "latitude_deg": 18.53,
            "longitude_deg": 73.85,
            "DNI_constant_W_m2": 705.0,
            "angle_constant_deg": 0.0,
        }
    )
    cfg["operation"].update(
        {
            "name": "Bhambare / Sukhatme — Pune 15/04, 12:30 LAT",
            "fluid": "ParathermNF",
            "mdot": 0.0986,
            "Tin_K": 150.0 + 273.15,
            # Ocho horas de calentamiento previo y final exactamente en 12:30 LAT.
            "t_start_s": 4.5 * 3600.0,
            "t_end_s": 12.5 * 3600.0,
            "output_step_s": 120.0,
        }
    )
    cfg["model"].update(
        {
            "has_glass": True,
            "annulus": "vacio_ideal",
            "internal_correlation": "dittusboelter_forzado",
            "include_supports": False,
        }
    )
    cfg["solver"].update({"rtol": 1e-6, "atol": 1e-7, "max_step_s": 20.0, "use_jac_sparsity": True})
    cfg["preset_meta"] = {
        "id": "bhambare_sukhatme_validation",
        "family": "Bhambare / Sukhatme",
        "source": "Parimal S. Bhambare (2020), A Solar Parabolic Trough Concentrator (PTC) Model using Matlab Simulink; validación contra Sukhatme & Nayak",
        "reference_table": "Table 4",
        "city": "Pune, India",
        "date_label": "15/04, 12:30 LAT",
        "latitude_deg": 18.53,
        "longitude_deg": 73.85,
        "strict_reference": True,
        "reference": {
            "Tin_C": 150.0,
            "mdot_kg_s": 0.0986,
            "beam_W_m2": 705.0,
            "Tamb_C": 31.9,
            "wind_m_s": 5.3,
            "Tglass_book_K": 333.39,
            "Tabs_book_K": 441.13,
            "Tout_book_C": 155.34,
            "Qloss_book_W": 857.6,
            "Tglass_article_K": 331.4,
            "Tabs_article_K": 465.4,
            "Tout_article_C": 154.1,
            "Qloss_article_W": 813.8,
        },
        "assumptions": [
            "Para la reproducción numérica se interpreta Ib=705 W/m² como irradiancia efectiva constante sobre la apertura y se fija ángulo=0°. El artículo especifica fecha/hora e Ib, pero no una serie transitoria previa.",
            "Las tablas completas de rho, mu y k de Paratherm NF no están publicadas por Bhambare; se usan las correlaciones/tablas de nuestra base de propiedades, mientras Cp conserva la relación publicada.",
        ],
    }
    return cfg, default_fluid_database()


def build_base_preset() -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = default_config()
    cfg["preset_meta"] = {
        "id": "base_editable",
        "family": "Base editable",
        "source": "Modelo base de esta implementación",
        "reference_table": "—",
        "city": "Pune (valores base)",
        "date_label": "Editable",
        "strict_reference": False,
        "reference": {},
        "assumptions": ["Caso libre; no representa por sí solo una validación documental."],
    }
    return cfg, default_fluid_database()


PRESET_FAMILY_LABELS = {
    "base": "Base editable",
    "rea_prototype": "Rea Quille — prototipo Foz 23/10/2021",
    "rea_foz_monthly": "Rea Quille — Foz do Iguaçu (Tabela 10)",
    "rea_alvorada_monthly": "Rea Quille — Alvorada do Norte (Tabela 11)",
    "bhambare": "Bhambare / Sukhatme — Pune 15/04",
}


def build_preset(
    family: str,
    month: int | None = None,
    annual: bool = False,
    *,
    variant: str | None = None,
    dni_source: str = "nominal",
) -> tuple[dict[str, Any], dict[str, Any]]:
    if family == "base":
        return build_base_preset()
    if family == "rea_prototype":
        return build_rea_prototype_preset(variant or "trnsys_published", dni_source=dni_source)
    if family == "rea_foz_monthly":
        return build_rea_annual_preset("foz") if annual else build_rea_monthly_preset("foz", int(month or 1))
    if family == "rea_alvorada_monthly":
        return build_rea_annual_preset("alvorada") if annual else build_rea_monthly_preset("alvorada", int(month or 1))
    if family == "bhambare":
        return build_bhambare_sukhatme_preset()
    raise ValueError(f"Preset desconocido: {family}")


def preset_summary_rows(cfg: dict[str, Any]) -> list[tuple[str, str]]:
    meta = cfg.get("preset_meta", {})
    op = cfg["operation"]
    env = cfg["environment"]
    solar = cfg["solar"]
    g = cfg["geometry"]
    return [
        ("Fuente", str(meta.get("source", "—"))),
        ("Caso", str(meta.get("date_label", "—"))),
        ("Ciudad", str(meta.get("city", "—"))),
        ("Fluido", str(op.get("fluid", "—"))),
        ("mdot", f"{float(op['mdot']):.5f} kg/s"),
        ("Tin", f"{float(op['Tin_K']) - 273.15:.2f} °C"),
        ("Tamb", f"{float(env['Tamb_K']) - 273.15:.2f} °C"),
        ("Irradiación", f"{float(solar.get('DNI_constant_W_m2', 0.0)):.2f} W/m² ({solar.get('mode', '—')})"),
        ("Latitud", f"{float(solar.get('latitude_deg', 0.0)):.5f}°"),
        ("Geometría", f"W={float(g['W']):.3f} m; L={float(g['L']):.3f} m; Dint={float(g['D2']):.4f} m; Dext={float(g['D3']):.4f} m"),
        ("Vidrio", "Sí" if cfg["model"]["has_glass"] else "No"),
    ]
