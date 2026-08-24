"""Configuración y propiedades predeterminadas del modelo PTC.

Los valores reproducen la versión MATLAB refactorizada entregada previamente.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


PARATHERM_TABLE = {
    "T_C": [20, 40, 60, 80, 100, 120, 150, 180, 200, 220, 250, 280, 300],
    "rho_kg_m3": [878, 866, 854, 842, 830, 818, 800, 782, 770, 758, 740, 722, 710],
    "mu_Pa_s": [0.0085, 0.0058, 0.0041, 0.0031, 0.00245, 0.00200, 0.00155, 0.00125, 0.00110, 0.00098, 0.00084, 0.00073, 0.00067],
    "k_W_mK": [0.137, 0.135, 0.133, 0.131, 0.129, 0.127, 0.124, 0.121, 0.119, 0.117, 0.114, 0.111, 0.109],
}


def _water_table() -> dict[str, list[float]]:
    temperatures = np.linspace(0.0, 150.0, 16)
    rho = 1000.0 * (
        1.0
        - ((temperatures + 288.9414) / (508929.2 * (temperatures + 68.12963)))
        * (temperatures - 3.9863) ** 2
    )
    mu = 2.414e-5 * 10.0 ** (247.8 / (temperatures + 133.15))
    cp = 4179.6 - 0.090 * temperatures + 0.0050 * temperatures**2
    k = 0.561 + 0.00190 * temperatures - 6.0e-6 * temperatures**2
    return {
        "T_C": temperatures.tolist(),
        "rho_kg_m3": rho.tolist(),
        "mu_Pa_s": mu.tolist(),
        "Cp_J_kgK": cp.tolist(),
        "k_W_mK": k.tolist(),
    }


DEFAULT_FLUID_DATABASE: dict[str, dict[str, Any]] = {
    "ParathermNF": {
        "display_name": "Paratherm NF",
        "mode": "original",
        "cp_model": "linear_kelvin",
        "cp_a": 3.6161,
        "cp_b": 814.37,
        "table": PARATHERM_TABLE,
        "multipliers": {
            "rho": 1.0,
            "mu": 1.0,
            "Cp": 1.0,
            "k": 1.0,
        },
        "minimums": {
            "rho": 500.0,
            "mu": 1e-5,
            "Cp": 500.0,
            "k": 0.05,
        },
    },
    "Agua": {
        "display_name": "Agua",
        "mode": "original",
        "table": _water_table(),
        "multipliers": {
            "rho": 1.0,
            "mu": 1.0,
            "Cp": 1.0,
            "k": 1.0,
        },
        "minimums": {
            "rho": 850.0,
            "mu": 1e-5,
            "Cp": 3500.0,
            "k": 0.40,
        },
    },
    "Personalizado": {
        "display_name": "Fluido personalizado",
        "mode": "constant",
        "constants": {
            "rho_kg_m3": 850.0,
            "mu_Pa_s": 0.0020,
            "Cp_J_kgK": 2200.0,
            "k_W_mK": 0.125,
        },
        "multipliers": {
            "rho": 1.0,
            "mu": 1.0,
            "Cp": 1.0,
            "k": 1.0,
        },
        "minimums": {
            "rho": 1.0,
            "mu": 1e-8,
            "Cp": 1.0,
            "k": 1e-4,
        },
    },
}


def default_config() -> dict[str, Any]:
    """Retorna una copia independiente de la configuración base."""
    return {
        "geometry": {
            "L": 3.657,
            "W": 1.25,
            "f": 0.60,
            "D2": 0.03810,
            "D3": 0.04135,
            "D4": 0.05600,
            "D5": 0.06300,
            "Nseg": 12,
        },
        "materials": {
            "absorber": {
                "rho": 8933.0,
                "Cp": 385.0,
                "k": 385.0,
                "eps": 0.95,
                "alpha": 0.95,
            },
            "glass": {
                "rho": 4500.0,
                "Cp": 840.0,
                "k": 1.2,
                "eps": 0.88,
                "tau": 0.85,
                "alpha": 0.02,
            },
        },
        "optics": {
            "reflectivity": 0.85,
            "intercept_factor": 0.95,
            "dirt_factor": 1.00,
            "shade_factor": 1.00,
            "tracking": "N-S horizontal, un eje",
        },
        "environment": {
            "Tamb_K": 31.9 + 273.15,
            "sky_delta_K": 6.0,
            "wind_m_s": 5.3,
            "pressure_Pa": 101325.0,
        },
        "solar": {
            "mode": "Parishwad",
            "A": 713.35,
            "B": 0.131,
            "day_of_year": 105,
            "latitude_deg": 18.53,
            "longitude_deg": 73.85,
            "DNI_constant_W_m2": 705.0,
            "angle_constant_deg": 0.0,
            "profile": {
                "LAT_h": [8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
                "DNI_W_m2": [0.0, 500.0, 705.0, 620.0, 300.0, 0.0],
                "theta_deg": [70.0, 35.0, 0.0, 30.0, 55.0, 80.0],
            },
        },
        "operation": {
            "name": "Caso base",
            "fluid": "ParathermNF",
            "mdot": 0.015,
            "Tin_K": 31.9 + 273.15,
            "t_start_s": 8.0 * 3600.0,
            "t_end_s": 18.0 * 3600.0,
            "output_step_s": 60.0,
        },
        "model": {
            "has_glass": True,
            "annulus": "vacio_ideal",
            "annulus_pressure_Pa": 0.1,
            "annulus_h_effective_W_m2K": 0.05,
            "internal_correlation": "automatica",
            "Re_laminar_max": 2300.0,
            "Re_turbulent_min": 4000.0,
            "include_supports": False,
            "support_loss_fraction": 0.015,
            "UL_area": "apertura",
        },
        "solver": {
            "rtol": 1e-6,
            "atol": 1e-7,
            "max_step_s": 60.0,
            "use_jac_sparsity": True,
        },
        "constants": {
            "sigma": 5.670374419e-8,
            "R_universal": 8.314462618,
        },
    }


def default_fluid_database() -> dict[str, dict[str, Any]]:
    """Retorna una copia independiente de la base de propiedades."""
    return deepcopy(DEFAULT_FLUID_DATABASE)
