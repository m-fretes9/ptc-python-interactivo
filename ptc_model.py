"""Núcleo numérico del colector cilindro-parabólico (PTC).

Migración directa del modelo MATLAB refactorizado. El equivalente de ode15s
se implementa con ``scipy.integrate.solve_ivp(method='BDF')``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix, lil_matrix

from fluid_properties import FluidProperties, FluidPropertyEvaluator


@dataclass
class SimulationResult:
    config: dict[str, Any]
    t_s: np.ndarray
    y_K: np.ndarray
    Tf_C: np.ndarray
    Tabs_C: np.ndarray
    Tglass_C: np.ndarray
    Tout_C: np.ndarray
    Tabs_mean_C: np.ndarray
    Tglass_mean_C: np.ndarray
    scalar_diag: dict[str, np.ndarray]
    node_diag: dict[str, np.ndarray]
    solver_message: str
    nfev: int
    njev: int
    nlu: int

    @property
    def LAT_h(self) -> np.ndarray:
        return self.t_s / 3600.0

    @property
    def n_segments(self) -> int:
        return int(self.config["geometry"]["Nseg"])

    def scalar_dataframe(self) -> pd.DataFrame:
        data: dict[str, np.ndarray] = {
            "t_s": self.t_s,
            "LAT_h": self.LAT_h,
            "Tout_C": self.Tout_C,
            "Tabs_mean_C": self.Tabs_mean_C,
            "Tglass_mean_C": self.Tglass_mean_C,
        }
        data.update(self.scalar_diag)
        return pd.DataFrame(data)

    def node_dataframe(self, time_index: int) -> pd.DataFrame:
        k = int(np.clip(time_index, 0, len(self.t_s) - 1))
        n = self.n_segments
        dx = float(self.config["geometry"]["L"]) / n
        data: dict[str, Any] = {
            "node": np.arange(1, n + 1),
            "x_center_m": (np.arange(n) + 0.5) * dx,
            "Tf_C": self.Tf_C[k, :],
            "Tabs_C": self.Tabs_C[k, :],
            "Tglass_C": self.Tglass_C[k, :],
        }
        for name, values in self.node_diag.items():
            data[name] = values[k, :]
        return pd.DataFrame(data)

    def node_snapshot(self, time_index: int, node_index: int) -> dict[str, float]:
        k = int(np.clip(time_index, 0, len(self.t_s) - 1))
        i = int(np.clip(node_index, 0, self.n_segments - 1))
        snapshot: dict[str, float] = {
            "time_index": float(k),
            "node_index": float(i),
            "t_s": float(self.t_s[k]),
            "LAT_h": float(self.LAT_h[k]),
            "Tf_K": float(self.Tf_C[k, i] + 273.15),
            "Tabs_K": float(self.Tabs_C[k, i] + 273.15),
            "Tglass_K": float(self.Tglass_C[k, i] + 273.15),
            "Tf_C": float(self.Tf_C[k, i]),
            "Tabs_C": float(self.Tabs_C[k, i]),
            "Tglass_C": float(self.Tglass_C[k, i]),
        }
        for name, values in self.node_diag.items():
            snapshot[name] = float(values[k, i])
        for name, values in self.scalar_diag.items():
            snapshot[name] = float(values[k])
        return snapshot


class PTCSimulator:
    def __init__(
        self,
        config: Mapping[str, Any],
        fluid_database: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.cfg = _deep_plain_copy(config)
        self.fluid_database = _deep_plain_copy(fluid_database)
        self._validate_config()
        self.g = self.cfg["geometry"]
        self.n = int(self.g["Nseg"])
        self.dx = float(self.g["L"]) / self.n
        fluid_key = str(self.cfg["operation"]["fluid"])
        self.fluid = FluidPropertyEvaluator(fluid_key, self.fluid_database)

    def _validate_config(self) -> None:
        g = self.cfg["geometry"]
        diameters = [float(g[key]) for key in ("D2", "D3", "D4", "D5")]
        if not (0.0 < diameters[0] < diameters[1] < diameters[2] < diameters[3]):
            raise ValueError("Debe cumplirse 0 < D2 < D3 < D4 < D5.")
        if int(g["Nseg"]) < 1:
            raise ValueError("Nseg debe ser mayor o igual a 1.")
        operation = self.cfg["operation"]
        if float(operation["mdot"]) <= 0.0:
            raise ValueError("El caudal másico debe ser positivo.")
        if float(operation["t_end_s"]) <= float(operation["t_start_s"]):
            raise ValueError("El tiempo final debe ser mayor que el inicial.")
        if float(operation["output_step_s"]) <= 0.0:
            raise ValueError("El paso de salida debe ser positivo.")
        model = self.cfg["model"]
        re_lam = float(model.get("Re_laminar_max", 2300.0))
        re_turb = float(model.get("Re_turbulent_min", 4000.0))
        if re_lam <= 0.0 or re_turb <= re_lam:
            raise ValueError("Debe cumplirse 0 < Re_laminar_max < Re_turbulent_min.")

    def simulate(self) -> SimulationResult:
        operation = self.cfg["operation"]
        solver = self.cfg["solver"]
        t_start = float(operation["t_start_s"])
        t_end = float(operation["t_end_s"])
        output_step = float(operation["output_step_s"])
        t_eval = np.arange(t_start, t_end + 0.5 * output_step, output_step, dtype=float)
        t_eval = t_eval[t_eval <= t_end]
        if t_eval.size == 0 or t_eval[-1] < t_end:
            t_eval = np.append(t_eval, t_end)

        n_states = 3 * self.n
        y0 = np.empty(n_states, dtype=float)
        y0[: self.n] = float(operation["Tin_K"])
        y0[self.n : 2 * self.n] = float(self.cfg["environment"]["Tamb_K"])
        y0[2 * self.n :] = float(self.cfg["environment"]["Tamb_K"])

        kwargs: dict[str, Any] = {
            "method": "BDF",
            "t_eval": t_eval,
            "rtol": float(solver["rtol"]),
            "atol": np.full(n_states, float(solver["atol"]), dtype=float),
            "max_step": float(solver["max_step_s"]),
        }
        if bool(solver.get("use_jac_sparsity", True)):
            kwargs["jac_sparsity"] = create_jacobian_sparsity(self.n)

        solution = solve_ivp(self.rhs, (t_start, t_end), y0, **kwargs)
        if not solution.success:
            raise RuntimeError(f"El solver BDF no convergió: {solution.message}")

        # solve_ivp devuelve y con forma (n_estados, n_tiempos).
        y = solution.y.T
        scalar_names = [
            "DNI_W_m2",
            "theta_deg",
            "cosTheta",
            "cosZ",
            "IAM",
            "endLoss",
            "Qincident_W",
            "QsolarAbs_W",
            "QsolarGlass_W",
            "Quseful_W",
            "Qloss_W",
            "eta_pct",
            "eta_optical_abs_pct",
            "eta_balance_pct",
            "Qstorage_est_W",
            "UL_W_m2K",
            "Tsky_C",
            "eps_sky_clear",
            "eps_sky",
        ]
        node_names = [
            "Qfluid_W",
            "Qrad_abs_glass_W",
            "Qconv_annulus_W",
            "Qconv_external_W",
            "Qrad_sky_W",
            "Qsupports_W",
            "Qadvection_W",
            "Qsolar_abs_node_W",
            "Qsolar_glass_node_W",
            "Re_internal",
            "Pr_internal",
            "Pr_wall",
            "Nu_internal",
            "h_internal_W_m2K",
            "transition_weight",
            "h_external_W_m2K",
            "rho_kg_m3",
            "mu_Pa_s",
            "Cp_J_kgK",
            "k_W_mK",
            "R_internal_K_W",
            "R_rad_abs_glass_K_W",
            "R_conv_annulus_K_W",
            "R_conv_external_K_W",
            "R_rad_sky_K_W",
            "dTf_dt_K_s",
            "dTabs_dt_K_s",
            "dTglass_dt_K_s",
        ]
        scalar_diag = {name: np.full(solution.t.size, np.nan) for name in scalar_names}
        node_diag = {
            name: np.full((solution.t.size, self.n), np.nan) for name in node_names
        }

        for k, (time_s, state) in enumerate(zip(solution.t, y, strict=True)):
            fluxes = self.calculate_fluxes(float(time_s), state)
            for name in scalar_names:
                scalar_diag[name][k] = float(fluxes[name])
            for name in node_names:
                node_diag[name][k, :] = np.asarray(fluxes[name], dtype=float)

        Tf_C = y[:, : self.n] - 273.15
        Tabs_C = y[:, self.n : 2 * self.n] - 273.15
        Tglass_C = y[:, 2 * self.n :] - 273.15
        return SimulationResult(
            config=self.cfg,
            t_s=solution.t,
            y_K=y,
            Tf_C=Tf_C,
            Tabs_C=Tabs_C,
            Tglass_C=Tglass_C,
            Tout_C=Tf_C[:, -1],
            Tabs_mean_C=Tabs_C.mean(axis=1),
            Tglass_mean_C=Tglass_C.mean(axis=1),
            scalar_diag=scalar_diag,
            node_diag=node_diag,
            solver_message=str(solution.message),
            nfev=int(solution.nfev),
            njev=int(getattr(solution, "njev", 0) or 0),
            nlu=int(getattr(solution, "nlu", 0) or 0),
        )

    def rhs(self, time_s: float, state: np.ndarray) -> np.ndarray:
        return np.asarray(self.calculate_fluxes(time_s, state)["dYdt"], dtype=float)

    def calculate_fluxes(self, time_s: float, state: np.ndarray) -> dict[str, Any]:
        n = self.n
        g = self.g
        cfg = self.cfg
        sigma = float(cfg["constants"]["sigma"])
        Tf = np.asarray(state[:n], dtype=float)
        Tabs = np.asarray(state[n : 2 * n], dtype=float)
        Tglass = np.asarray(state[2 * n : 3 * n], dtype=float)

        solar = self.solar_model(time_s)
        q_solar_abs_node = float(solar["QsolarAbs_W"]) / n
        q_solar_glass_node = float(solar["QsolarGlass_W"]) / n

        arrays = {
            name: np.zeros(n, dtype=float)
            for name in (
                "dTf",
                "dTabs",
                "dTglass",
                "Qfluid_W",
                "Qrad_abs_glass_W",
                "Qconv_annulus_W",
                "Qconv_external_W",
                "Qrad_sky_W",
                "Qsupports_W",
                "Qadvection_W",
                "Re_internal",
                "Pr_internal",
                "Pr_wall",
                "Nu_internal",
                "h_internal_W_m2K",
                "transition_weight",
                "h_external_W_m2K",
                "rho_kg_m3",
                "mu_Pa_s",
                "Cp_J_kgK",
                "k_W_mK",
                "R_internal_K_W",
                "R_rad_abs_glass_K_W",
                "R_conv_annulus_K_W",
                "R_conv_external_K_W",
                "R_rad_sky_K_W",
            )
        }

        Vfluid = np.pi * float(g["D2"]) ** 2 / 4.0 * self.dx
        Vabs = np.pi / 4.0 * (float(g["D3"]) ** 2 - float(g["D2"]) ** 2) * self.dx
        Vglass = np.pi / 4.0 * (float(g["D5"]) ** 2 - float(g["D4"]) ** 2) * self.dx
        absorber = cfg["materials"]["absorber"]
        glass = cfg["materials"]["glass"]
        Cabs = float(absorber["rho"]) * Vabs * float(absorber["Cp"])
        Cglass = float(glass["rho"]) * Vglass * float(glass["Cp"])
        Tamb = float(cfg["environment"]["Tamb_K"])
        sky = effective_sky_temperature(time_s, cfg["environment"])
        Tsky = float(sky["Tsky_K"])
        mdot = float(cfg["operation"]["mdot"])
        Tin = float(cfg["operation"]["Tin_K"])

        for i in range(n):
            Tup = Tin if i == 0 else Tf[i - 1]
            prop = self.fluid(Tf[i])
            prop_wall = self.fluid(Tabs[i])
            prop_up = self.fluid(0.5 * (Tup + Tf[i]))
            conv_int = internal_convection(
                mdot,
                float(g["D2"]),
                prop,
                prop_wall,
                str(cfg["model"]["internal_correlation"]),
                float(cfg["model"].get("Re_laminar_max", 2300.0)),
                float(cfg["model"].get("Re_turbulent_min", 4000.0)),
            )

            Aint = np.pi * float(g["D2"]) * self.dx
            Rconv = 1.0 / max(conv_int["h_W_m2K"] * Aint, np.finfo(float).eps)
            Rwall_mean = np.log(float(g["D3"]) / float(g["D2"])) / (
                4.0 * np.pi * float(absorber["k"]) * self.dx
            )
            Rinternal = Rconv + Rwall_mean
            q_fluid = (Tabs[i] - Tf[i]) / max(Rinternal, np.finfo(float).eps)
            arrays["Qfluid_W"][i] = q_fluid
            arrays["R_internal_K_W"][i] = Rinternal

            Cfluid = max(prop.rho * Vfluid * prop.Cp, np.finfo(float).eps)
            q_advection = mdot * prop_up.Cp * (Tup - Tf[i])
            arrays["Qadvection_W"][i] = q_advection
            arrays["dTf"][i] = (q_advection + q_fluid) / Cfluid

            if bool(cfg["model"]["has_glass"]):
                denominator = 1.0 / float(absorber["eps"]) + (
                    float(g["D3"]) / float(g["D4"])
                ) * (1.0 / float(glass["eps"]) - 1.0)
                q_rad_abs_glass = (
                    sigma
                    * np.pi
                    * float(g["D3"])
                    * self.dx
                    * (Tabs[i] ** 4 - Tglass[i] ** 4)
                    / max(denominator, np.finfo(float).eps)
                )
                h_annulus = annulus_convection(Tabs[i], Tglass[i], cfg)
                q_conv_annulus = (
                    h_annulus
                    * np.pi
                    * float(g["D3"])
                    * self.dx
                    * (Tabs[i] - Tglass[i])
                )
                conv_ext = external_convection(
                    0.5 * (Tglass[i] + Tamb),
                    float(cfg["environment"]["wind_m_s"]),
                    float(g["D5"]),
                    float(cfg["environment"]["pressure_Pa"]),
                )
                q_conv_external = (
                    conv_ext["h_W_m2K"]
                    * np.pi
                    * float(g["D5"])
                    * self.dx
                    * (Tglass[i] - Tamb)
                )
                q_rad_sky = (
                    float(glass["eps"])
                    * sigma
                    * np.pi
                    * float(g["D5"])
                    * self.dx
                    * (Tglass[i] ** 4 - Tsky**4)
                )
                q_supports = 0.0
                if bool(cfg["model"]["include_supports"]):
                    q_supports = float(cfg["model"]["support_loss_fraction"]) * max(
                        q_rad_abs_glass + q_conv_annulus, 0.0
                    )

                arrays["Qrad_abs_glass_W"][i] = q_rad_abs_glass
                arrays["Qconv_annulus_W"][i] = q_conv_annulus
                arrays["Qconv_external_W"][i] = q_conv_external
                arrays["Qrad_sky_W"][i] = q_rad_sky
                arrays["Qsupports_W"][i] = q_supports
                arrays["h_external_W_m2K"][i] = conv_ext["h_W_m2K"]
                arrays["dTabs"][i] = (
                    q_solar_abs_node
                    - q_fluid
                    - q_rad_abs_glass
                    - q_conv_annulus
                    - q_supports
                ) / Cabs
                arrays["dTglass"][i] = (
                    q_solar_glass_node
                    + q_rad_abs_glass
                    + q_conv_annulus
                    - q_conv_external
                    - q_rad_sky
                ) / Cglass

                arrays["R_rad_abs_glass_K_W"][i] = effective_resistance(
                    Tabs[i] - Tglass[i], q_rad_abs_glass
                )
                arrays["R_conv_annulus_K_W"][i] = effective_resistance(
                    Tabs[i] - Tglass[i], q_conv_annulus
                )
                arrays["R_conv_external_K_W"][i] = effective_resistance(
                    Tglass[i] - Tamb, q_conv_external
                )
                arrays["R_rad_sky_K_W"][i] = effective_resistance(
                    Tglass[i] - Tsky, q_rad_sky
                )
            else:
                conv_ext = external_convection(
                    0.5 * (Tabs[i] + Tamb),
                    float(cfg["environment"]["wind_m_s"]),
                    float(g["D3"]),
                    float(cfg["environment"]["pressure_Pa"]),
                )
                q_conv_external = (
                    conv_ext["h_W_m2K"]
                    * np.pi
                    * float(g["D3"])
                    * self.dx
                    * (Tabs[i] - Tamb)
                )
                q_rad_sky = (
                    float(absorber["eps"])
                    * sigma
                    * np.pi
                    * float(g["D3"])
                    * self.dx
                    * (Tabs[i] ** 4 - Tsky**4)
                )
                q_supports = 0.0
                if bool(cfg["model"]["include_supports"]):
                    q_supports = float(cfg["model"]["support_loss_fraction"]) * max(
                        q_conv_external + q_rad_sky, 0.0
                    )

                arrays["Qconv_external_W"][i] = q_conv_external
                arrays["Qrad_sky_W"][i] = q_rad_sky
                arrays["Qsupports_W"][i] = q_supports
                arrays["h_external_W_m2K"][i] = conv_ext["h_W_m2K"]
                arrays["dTabs"][i] = (
                    q_solar_abs_node
                    - q_fluid
                    - q_conv_external
                    - q_rad_sky
                    - q_supports
                ) / Cabs
                arrays["dTglass"][i] = 0.0
                arrays["R_conv_external_K_W"][i] = effective_resistance(
                    Tabs[i] - Tamb, q_conv_external
                )
                arrays["R_rad_sky_K_W"][i] = effective_resistance(
                    Tabs[i] - Tsky, q_rad_sky
                )

            arrays["Re_internal"][i] = conv_int["Re"]
            arrays["Pr_internal"][i] = conv_int["Pr"]
            arrays["Pr_wall"][i] = conv_int["PrWall"]
            arrays["Nu_internal"][i] = conv_int["Nu"]
            arrays["h_internal_W_m2K"][i] = conv_int["h_W_m2K"]
            arrays["transition_weight"][i] = conv_int["transition_weight"]
            arrays["rho_kg_m3"][i] = prop.rho
            arrays["mu_Pa_s"][i] = prop.mu
            arrays["Cp_J_kgK"][i] = prop.Cp
            arrays["k_W_mK"][i] = prop.k

        prop_in = self.fluid(Tin)
        prop_out = self.fluid(Tf[-1])
        Cp_mean = 0.5 * (prop_in.Cp + prop_out.Cp)
        Quseful = mdot * Cp_mean * (Tf[-1] - Tin)
        Qloss = float(
            np.sum(
                arrays["Qconv_external_W"]
                + arrays["Qrad_sky_W"]
                + arrays["Qsupports_W"]
            )
        )
        delta_T_UL = float(np.mean(Tabs) - Tamb)
        UL = (
            Qloss / (float(g["W"]) * float(g["L"]) * delta_T_UL)
            if abs(delta_T_UL) > 1e-9
            else np.nan
        )
        Qincident = float(solar["Qincident_W"])
        eta = 100.0 * Quseful / Qincident if Qincident > 1e-9 else np.nan
        Qsolar_total = float(solar["QsolarAbs_W"]) + float(solar["QsolarGlass_W"])
        eta_optical_abs = (
            100.0 * float(solar["QsolarAbs_W"]) / Qincident
            if Qincident > 1e-9 else np.nan
        )
        # Diagnóstico de balance: en régimen cuasiestacionario eta_balance ~= eta_HTF.
        # La diferencia representa principalmente almacenamiento/liberación de energía
        # en el HTF, absorbedor y vidrio durante los transitorios.
        Qstorage_est = Qsolar_total - Qloss - Quseful
        eta_balance = (
            100.0 * (Qsolar_total - Qloss) / Qincident
            if Qincident > 1e-9 else np.nan
        )

        output: dict[str, Any] = {
            "dYdt": np.concatenate((arrays["dTf"], arrays["dTabs"], arrays["dTglass"])),
            **solar,
            "Quseful_W": Quseful,
            "Qloss_W": Qloss,
            "eta_pct": eta,
            "eta_optical_abs_pct": eta_optical_abs,
            "eta_balance_pct": eta_balance,
            "Qstorage_est_W": Qstorage_est,
            "UL_W_m2K": UL,
            "Tsky_C": float(Tsky - 273.15),
            "eps_sky_clear": float(sky["eps_clear"]),
            "eps_sky": float(sky["eps_sky"]),
            "Qsolar_abs_node_W": np.full(n, q_solar_abs_node),
            "Qsolar_glass_node_W": np.full(n, q_solar_glass_node),
            "dTf_dt_K_s": arrays["dTf"].copy(),
            "dTabs_dt_K_s": arrays["dTabs"].copy(),
            "dTglass_dt_K_s": arrays["dTglass"].copy(),
        }
        output.update({key: value for key, value in arrays.items() if not key.startswith("dT")})
        return output

    def solar_model(self, time_s: float) -> dict[str, float]:
        solar = self.cfg["solar"]
        mode = str(solar["mode"]).lower()
        LAT_h = float(time_s) / 3600.0

        if mode == "constante":
            DNI = max(float(solar["DNI_constant_W_m2"]), 0.0)
            theta_deg = max(float(solar["angle_constant_deg"]), 0.0)
            cos_theta = max(float(np.cos(np.deg2rad(theta_deg))), 0.0)
            cos_z = cos_theta
        elif mode == "perfil":
            profile = solar["profile"]
            times = np.asarray(profile["LAT_h"], dtype=float)
            dni_values = np.asarray(profile["DNI_W_m2"], dtype=float)
            theta_values = np.asarray(profile["theta_deg"], dtype=float)
            order = np.argsort(times)
            times = times[order]
            dni_values = dni_values[order]
            theta_values = theta_values[order]
            DNI = max(float(np.interp(LAT_h, times, dni_values)), 0.0)
            theta_deg = float(np.clip(np.interp(LAT_h, times, theta_values), 0.0, 90.0))
            cos_theta = max(float(np.cos(np.deg2rad(theta_deg))), 0.0)
            cos_z = cos_theta
        else:
            phi = float(solar["latitude_deg"])
            day = float(solar["day_of_year"])
            delta = 23.45 * np.sin(np.deg2rad(360.0 * (284.0 + day) / 365.0))
            omega = 15.0 * (12.0 - LAT_h)
            cos_z = (
                np.sin(np.deg2rad(phi)) * np.sin(np.deg2rad(delta))
                + np.cos(np.deg2rad(phi))
                * np.cos(np.deg2rad(delta))
                * np.cos(np.deg2rad(omega))
            )
            cos_theta_sq = cos_z**2 + (
                np.cos(np.deg2rad(delta)) * np.sin(np.deg2rad(omega))
            ) ** 2
            cos_theta = float(np.sqrt(max(cos_theta_sq, 0.0)))
            cos_theta = float(np.clip(cos_theta, 0.0, 1.0))
            theta_deg = float(np.rad2deg(np.arccos(cos_theta)))
            if cos_z > 0.0:
                DNI = float(solar["A"]) * np.exp(-float(solar["B"]) / max(cos_z, 1e-8))
            else:
                DNI = 0.0
                cos_theta = 0.0
                theta_deg = 90.0

        if DNI > 0.0 and cos_theta > 1e-8:
            IAM = (
                1.0
                + 0.000884 * theta_deg / cos_theta
                - 0.00005369 * theta_deg**2 / cos_theta
            )
            IAM = float(np.clip(IAM, 0.0, 1.2))
            end_loss = 1.0 - float(self.g["f"]) * np.tan(np.deg2rad(theta_deg)) / float(
                self.g["L"]
            )
            end_loss = float(np.clip(end_loss, 0.0, 1.0))
        else:
            IAM = 0.0
            end_loss = 0.0

        aperture_area = float(self.g["W"]) * float(self.g["L"])
        Qincident = DNI * cos_theta * aperture_area
        optics = self.cfg["optics"]
        absorber = self.cfg["materials"]["absorber"]
        glass = self.cfg["materials"]["glass"]
        if bool(self.cfg["model"]["has_glass"]):
            eta_opt_abs = (
                float(optics["reflectivity"])
                * float(optics["intercept_factor"])
                * float(glass["tau"])
                * float(absorber["alpha"])
                * float(optics["dirt_factor"])
                * float(optics["shade_factor"])
            )
            Qsolar_abs = Qincident * eta_opt_abs * IAM * end_loss
            Qsolar_glass = (
                DNI
                * cos_theta
                * float(self.g["D5"])
                * float(self.g["L"])
                * float(optics["reflectivity"])
                * float(optics["intercept_factor"])
                * float(glass["alpha"])
                * IAM
                * end_loss
            )
        else:
            eta_opt_abs = (
                float(optics["reflectivity"])
                * float(optics["intercept_factor"])
                * float(absorber["alpha"])
                * float(optics["dirt_factor"])
                * float(optics["shade_factor"])
            )
            Qsolar_abs = Qincident * eta_opt_abs * IAM * end_loss
            Qsolar_glass = 0.0

        return {
            "DNI_W_m2": float(DNI),
            "theta_deg": float(theta_deg),
            "cosTheta": float(cos_theta),
            "cosZ": float(cos_z),
            "IAM": float(IAM),
            "endLoss": float(end_loss),
            "Qincident_W": float(Qincident),
            "QsolarAbs_W": float(Qsolar_abs),
            "QsolarGlass_W": float(Qsolar_glass),
        }


def air_properties(T_K: float, pressure_Pa: float) -> dict[str, float]:
    R = 287.058
    rho = pressure_Pa / (R * T_K)
    mu0 = 1.716e-5
    T0 = 273.15
    S = 111.0
    mu = mu0 * (T_K / T0) ** 1.5 * (T0 + S) / (T_K + S)
    Cp = 1006.0 + 0.1 * (T_K - 300.0)
    Pr = 0.707
    k = mu * Cp / Pr
    return {"rho": rho, "mu": mu, "Cp": Cp, "k": k, "Pr": Pr}


def internal_convection(
    mdot: float,
    diameter_m: float,
    prop: FluidProperties,
    prop_wall: FluidProperties,
    mode: str,
    re_laminar_max: float = 2300.0,
    re_turbulent_min: float = 4000.0,
) -> dict[str, Any]:
    """Coeficiente convectivo interno con transición continua de régimen.

    El modelo anterior conmutaba de Nu=4.36 a Gnielinski exactamente en
    Re=2300. Esa discontinuidad en Nu y h producía picos artificiales cuando
    la viscosidad dependiente de T hacía cruzar el umbral durante el día.

    En modo automático/Gnielinski se usa Nu=4.36 hasta re_laminar_max,
    Gnielinski a partir de re_turbulent_min y una mezcla smoothstep entre
    ambos en la zona de transición. Dittus-Boelter forzado conserva su
    comportamiento explícito para estudios de sensibilidad.
    """
    Re = 4.0 * mdot / (np.pi * diameter_m * prop.mu)
    Pr = prop.Pr
    Pr_wall = prop_wall.Pr
    normalized_mode = mode.lower()
    re_lam = float(re_laminar_max)
    re_turb = float(re_turbulent_min)
    if re_lam <= 0.0 or re_turb <= re_lam:
        raise ValueError("Debe cumplirse 0 < re_laminar_max < re_turbulent_min.")

    def _smoothstep_weight(reynolds: float) -> float:
        x = float(np.clip((reynolds - re_lam) / (re_turb - re_lam), 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _dittus_boelter_nu(reynolds: float) -> float:
        return max(0.023 * max(reynolds, 1.0) ** 0.8 * max(Pr, 0.1) ** 0.4, 4.36)

    def _gnielinski_nu(reynolds: float) -> float:
        friction = (1.82 * np.log10(max(reynolds, 2301.0)) - 1.64) ** -2
        value = (
            (friction / 8.0)
            * (reynolds - 1000.0)
            * Pr
            / (1.0 + 12.7 * np.sqrt(friction / 8.0) * (Pr ** (2.0 / 3.0) - 1.0))
            * (Pr / max(Pr_wall, np.finfo(float).eps)) ** 0.11
        )
        return max(float(value), 4.36)

    if normalized_mode == "dittusboelter_forzado":
        Nu = _dittus_boelter_nu(Re)
        weight = 1.0
        correlation = "Dittus-Boelter forzado"
    elif Re <= re_lam:
        Nu = 4.36
        weight = 0.0
        correlation = "Laminar Nu=4.36"
    elif normalized_mode == "dittusboelter":
        weight = _smoothstep_weight(Re)
        Nu_turb = _dittus_boelter_nu(Re)
        Nu = (1.0 - weight) * 4.36 + weight * Nu_turb
        correlation = (
            "Transicion suave laminar-Dittus-Boelter"
            if Re < re_turb else "Dittus-Boelter"
        )
    else:
        weight = _smoothstep_weight(Re)
        Nu_turb = _gnielinski_nu(Re)
        Nu = (1.0 - weight) * 4.36 + weight * Nu_turb
        correlation = (
            "Transicion suave laminar-Gnielinski"
            if Re < re_turb else "Gnielinski-Forristall"
        )

    return {
        "Re": float(Re),
        "Pr": float(Pr),
        "PrWall": float(Pr_wall),
        "Nu": float(Nu),
        "h_W_m2K": float(Nu * prop.k / diameter_m),
        "transition_weight": float(weight),
        "correlation": correlation,
    }


def effective_sky_temperature(time_s: float, environment: Mapping[str, Any]) -> dict[str, float]:
    """Temperatura efectiva del cielo.

    Modo ``rea_quille``: implementa las Ecs. (12)-(14) del TCC de
    Rea Quille (2025), basado en Martin y Berdahl (1984). La ecuación
    (13) se puede evaluar literalmente tal como está impresa
    (1 + eps0) o, para análisis de sensibilidad, con la variante física
    habitual (1 - eps0). No se sustituye silenciosamente una por otra.

    Modo ``delta_constante``: conserva el comportamiento legado
    Tsky = Tamb - sky_delta_K.
    """
    Tamb_K = float(environment["Tamb_K"])
    mode = str(environment.get("sky_model", "delta_constante")).lower()
    if mode == "delta_constante":
        delta = float(environment.get("sky_delta_K", 6.0))
        Tsky_K = Tamb_K - delta
        eps = max((Tsky_K / Tamb_K) ** 4, np.finfo(float).eps)
        return {"Tsky_K": float(Tsky_K), "eps_clear": float(eps), "eps_sky": float(eps)}

    if mode != "rea_quille":
        raise ValueError(f"Modelo de temperatura del cielo no reconocido: {mode}")

    tm_h = (float(time_s) / 3600.0) % 24.0
    Tdp_C = float(environment.get("dew_point_C", 20.0))
    P_mbar = float(environment.get("pressure_Pa", 101325.0)) / 100.0

    # Ec. (12) de Rea Quille: emisividad efectiva de cielo claro.
    eps_clear = (
        0.711
        + 0.56 * (Tdp_C / 100.0)
        + 0.73 * (Tdp_C / 100.0) ** 2
        + 0.013 * np.cos(2.0 * np.pi * tm_h / 24.0)
        + (0.012 / 100.0) * (P_mbar - 1000.0)
    )

    eps_sky = eps_clear
    if bool(environment.get("cloud_adjustment", False)):
        f_cloud = float(environment.get("cloud_factor", 0.0))
        eps_cloud = float(environment.get("cloud_emissivity", 1.0))
        formula = str(environment.get("cloud_formula", "rea_quille_impresa")).lower()
        if formula == "rea_quille_impresa":
            # Ec. (13) tal como aparece impresa en el TCC: (1 + eps0).
            eps_sky = eps_clear + (1.0 + eps_clear) * f_cloud * eps_cloud
        elif formula == "variante_fisica":
            # Variante disponible solo para sensibilidad; no se atribuye al TCC.
            eps_sky = eps_clear + (1.0 - eps_clear) * f_cloud * eps_cloud
            eps_sky = float(np.clip(eps_sky, np.finfo(float).eps, 1.0))
        else:
            raise ValueError(f"Fórmula de nubosidad no reconocida: {formula}")

    if eps_sky <= 0.0:
        raise ValueError("La emisividad efectiva del cielo debe ser positiva.")

    # Ec. (14): Tsky[K] = eps_sky^0.25 * Tamb[K].
    Tsky_K = float(eps_sky ** 0.25 * Tamb_K)
    return {"Tsky_K": Tsky_K, "eps_clear": float(eps_clear), "eps_sky": float(eps_sky)}


def external_convection(
    film_temperature_K: float,
    wind_m_s: float,
    diameter_m: float,
    pressure_Pa: float,
) -> dict[str, float]:
    air = air_properties(film_temperature_K, pressure_Pa)
    Re = air["rho"] * abs(wind_m_s) * diameter_m / air["mu"]
    Pr = air["Pr"]
    if wind_m_s <= 1e-9:
        Nu = 0.36
    else:
        Nu = (
            0.3
            + (0.62 * np.sqrt(Re) * Pr ** (1.0 / 3.0))
            / (1.0 + (0.4 / Pr) ** (2.0 / 3.0)) ** 0.25
            * (1.0 + (Re / 282000.0) ** (5.0 / 8.0)) ** (4.0 / 5.0)
        )
    return {
        "Re": float(Re),
        "Pr": float(Pr),
        "Nu": float(Nu),
        "h_W_m2K": float(Nu * air["k"] / diameter_m),
    }


def annulus_convection(Tabs_K: float, Tglass_K: float, cfg: Mapping[str, Any]) -> float:
    model = cfg["model"]
    mode = str(model["annulus"]).lower()
    if mode == "vacio_ideal":
        return 0.0
    if mode == "vacio_efectivo":
        return float(model["annulus_h_effective_W_m2K"])
    if mode != "aire":
        raise ValueError(f"Modelo anular no reconocido: {mode}")

    geometry = cfg["geometry"]
    environment = cfg["environment"]
    Tmean = 0.5 * (Tabs_K + Tglass_K)
    air = air_properties(Tmean, float(environment["pressure_Pa"]))
    beta = 1.0 / Tmean
    nu = air["mu"] / air["rho"]
    alpha = air["k"] / (air["rho"] * air["Cp"])
    delta_T = abs(Tabs_K - Tglass_K)
    Ra_D3 = (
        9.81
        * beta
        * delta_T
        * float(geometry["D3"]) ** 3
        / max(nu * alpha, np.finfo(float).eps)
    )
    geometry_factor = (
        1.0 + (float(geometry["D3"]) / float(geometry["D4"])) ** (3.0 / 5.0)
    ) ** (5.0 / 4.0)
    q_prime = (
        2.425
        * air["k"]
        * delta_T
        * (air["Pr"] * max(Ra_D3, 0.0) / (0.861 + air["Pr"])) ** 0.25
        / geometry_factor
    )
    return float(q_prime / max(np.pi * float(geometry["D3"]) * delta_T, np.finfo(float).eps))


def effective_resistance(delta_temperature_K: float, heat_flow_W: float) -> float:
    if abs(heat_flow_W) <= 1e-12:
        return np.inf
    return float(abs(delta_temperature_K / heat_flow_W))


def create_jacobian_sparsity(n_segments: int) -> csr_matrix:
    n = 3 * int(n_segments)
    pattern = lil_matrix((n, n), dtype=int)
    for i in range(int(n_segments)):
        i_fluid = i
        i_abs = int(n_segments) + i
        i_glass = 2 * int(n_segments) + i
        pattern[i_fluid, i_fluid] = 1
        pattern[i_fluid, i_abs] = 1
        if i > 0:
            pattern[i_fluid, i_fluid - 1] = 1
        pattern[i_abs, i_fluid] = 1
        pattern[i_abs, i_abs] = 1
        pattern[i_abs, i_glass] = 1
        pattern[i_glass, i_abs] = 1
        pattern[i_glass, i_glass] = 1
    return pattern.tocsr()


def _deep_plain_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _deep_plain_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_plain_copy(item) for item in value]
    if isinstance(value, tuple):
        return [_deep_plain_copy(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
