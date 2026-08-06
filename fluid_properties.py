"""Propiedades termofísicas dependientes de la temperatura."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


@dataclass(frozen=True)
class FluidProperties:
    rho: float
    mu: float
    Cp: float
    k: float

    @property
    def Pr(self) -> float:
        return self.Cp * self.mu / self.k


class FluidPropertyEvaluator:
    """Evalúa rho, mu, Cp y k para un fluido configurado.

    Modos soportados:
    - original: correlaciones de la versión MATLAB.
    - table: interpolación PCHIP de una tabla editable.
    - constant: propiedades constantes.
    """

    def __init__(self, fluid_key: str, fluid_database: Mapping[str, Mapping[str, Any]]):
        if fluid_key not in fluid_database:
            raise ValueError(f"Fluido no reconocido: {fluid_key}")
        self.fluid_key = fluid_key
        self.spec = dict(fluid_database[fluid_key])
        self.mode = str(self.spec.get("mode", "original")).lower()
        self.mult = dict(self.spec.get("multipliers", {}))
        self.minimums = dict(self.spec.get("minimums", {}))
        self._interpolators: dict[str, PchipInterpolator] = {}
        if self.mode == "table" or fluid_key == "ParathermNF":
            self._prepare_table_interpolators()

    def _prepare_table_interpolators(self) -> None:
        table = pd.DataFrame(self.spec.get("table", {})).copy()
        required = {"T_C", "rho_kg_m3", "mu_Pa_s", "k_W_mK"}
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(
                f"La tabla de {self.fluid_key} no contiene las columnas: {sorted(missing)}"
            )
        table = table.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required))
        table = table.sort_values("T_C").drop_duplicates("T_C", keep="last")
        if len(table) < 2:
            raise ValueError("Se requieren al menos dos temperaturas para interpolar propiedades.")
        x = table["T_C"].to_numpy(dtype=float)
        for column in ("rho_kg_m3", "mu_Pa_s", "k_W_mK"):
            self._interpolators[column] = PchipInterpolator(
                x, table[column].to_numpy(dtype=float), extrapolate=True
            )
        if "Cp_J_kgK" in table.columns and table["Cp_J_kgK"].notna().sum() >= 2:
            cp_table = table.dropna(subset=["Cp_J_kgK"])
            self._interpolators["Cp_J_kgK"] = PchipInterpolator(
                cp_table["T_C"].to_numpy(dtype=float),
                cp_table["Cp_J_kgK"].to_numpy(dtype=float),
                extrapolate=True,
            )

    def __call__(self, T_K: float) -> FluidProperties:
        T_K = float(T_K)
        T_C = T_K - 273.15

        if self.mode == "constant":
            constants = self.spec.get("constants", {})
            rho = float(constants["rho_kg_m3"])
            mu = float(constants["mu_Pa_s"])
            Cp = float(constants["Cp_J_kgK"])
            k = float(constants["k_W_mK"])
        elif self.mode == "table":
            rho = float(self._interpolators["rho_kg_m3"](T_C))
            mu = float(self._interpolators["mu_Pa_s"](T_C))
            k = float(self._interpolators["k_W_mK"](T_C))
            if "Cp_J_kgK" in self._interpolators:
                Cp = float(self._interpolators["Cp_J_kgK"](T_C))
            elif self.fluid_key == "ParathermNF":
                Cp = self._paratherm_cp(T_K)
            else:
                Cp = self._water_original(T_C).Cp
        elif self.fluid_key == "ParathermNF":
            rho = float(self._interpolators["rho_kg_m3"](T_C))
            mu = float(self._interpolators["mu_Pa_s"](T_C))
            k = float(self._interpolators["k_W_mK"](T_C))
            Cp = self._paratherm_cp(T_K)
        elif self.fluid_key == "Agua":
            original = self._water_original(T_C)
            rho, mu, Cp, k = original.rho, original.mu, original.Cp, original.k
        else:
            raise ValueError(
                f"El modo 'original' no está definido para el fluido {self.fluid_key}."
            )

        rho *= float(self.mult.get("rho", 1.0))
        mu *= float(self.mult.get("mu", 1.0))
        Cp *= float(self.mult.get("Cp", 1.0))
        k *= float(self.mult.get("k", 1.0))

        rho = max(rho, float(self.minimums.get("rho", 1e-12)))
        mu = max(mu, float(self.minimums.get("mu", 1e-12)))
        Cp = max(Cp, float(self.minimums.get("Cp", 1e-12)))
        k = max(k, float(self.minimums.get("k", 1e-12)))
        return FluidProperties(rho=rho, mu=mu, Cp=Cp, k=k)

    def _paratherm_cp(self, T_K: float) -> float:
        a = float(self.spec.get("cp_a", 3.6161))
        b = float(self.spec.get("cp_b", 814.37))
        return a * T_K + b

    @staticmethod
    def _water_original(T_C: float) -> FluidProperties:
        T = min(max(float(T_C), 0.0), 150.0)
        rho = 1000.0 * (
            1.0
            - ((T + 288.9414) / (508929.2 * (T + 68.12963))) * (T - 3.9863) ** 2
        )
        mu = 2.414e-5 * 10.0 ** (247.8 / (T + 133.15))
        Cp = 4179.6 - 0.090 * T + 0.0050 * T**2
        k = 0.561 + 0.00190 * T - 6.0e-6 * T**2
        return FluidProperties(
            rho=max(rho, 850.0),
            mu=max(mu, 1e-5),
            Cp=max(Cp, 3500.0),
            k=max(k, 0.40),
        )


def property_curve(
    fluid_key: str,
    fluid_database: Mapping[str, Mapping[str, Any]],
    T_min_C: float,
    T_max_C: float,
    n: int = 151,
) -> pd.DataFrame:
    evaluator = FluidPropertyEvaluator(fluid_key, fluid_database)
    temperatures = np.linspace(float(T_min_C), float(T_max_C), int(n))
    rows = []
    for temperature in temperatures:
        prop = evaluator(temperature + 273.15)
        rows.append(
            {
                "T_C": temperature,
                "rho_kg_m3": prop.rho,
                "mu_Pa_s": prop.mu,
                "Cp_J_kgK": prop.Cp,
                "k_W_mK": prop.k,
                "Pr": prop.Pr,
            }
        )
    return pd.DataFrame(rows)
