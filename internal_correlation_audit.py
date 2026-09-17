"""Auditoría directa de régimen y correlación de convección interna.

Objetivo V14.12
---------------
Aislar el efecto de la correlación interna sin recalibrar propiedades del HTF.
Se comparan, con la MISMA base de propiedades y las mismas condiciones:

* Nu = 4.36 (laminar, plenamente desarrollado, q'' uniforme)
* Hausen con corrección de entrada laminar y asíntota 4.36
* Sieder–Tate laminar de entrada
* Dittus–Boelter forzado (control documental, fuera de rango si Re es bajo)

La prueba se ejecuta tanto con agua (Rea Quille: Foz y Alvorada) como con
Paratherm NF (Bhambare/Sukhatme), precisamente para separar "propiedades del
aceite" de "selección de régimen/correlación".

Es una prueba diagnóstica. Las ramas Hausen/Sieder–Tate NO se convierten en
modelo definitivo automáticamente.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping
import math

import numpy as np
import pandas as pd

from presets import (
    MONTH_ABBR_ES,
    REA_ALVORADA_MONTHLY,
    REA_FOZ_MONTHLY,
    build_bhambare_sukhatme_preset,
    build_rea_monthly_preset,
)
from ptc_model import PTCSimulator


CORRELATIONS: dict[str, str] = {
    "laminar_436_forzado": "Legado desarrollado · Nu=4.36",
    "hausen_laminar": "Hausen · entrada laminar",
    "sieder_tate_laminar": "Sieder–Tate · entrada laminar",
    "dittusboelter_forzado": "Dittus–Boelter forzado · control",
}


def _fast_cfg(cfg: dict[str, Any], mode: str) -> dict[str, Any]:
    out = deepcopy(cfg)
    out["geometry"]["Nseg"] = min(int(out["geometry"].get("Nseg", 12)), 6)
    out["solver"]["method"] = "BDF"
    out["solver"]["max_step_s"] = max(float(out["solver"].get("max_step_s", 20.0)), 180.0)
    out["solver"]["rtol"] = min(float(out["solver"].get("rtol", 1e-6)), 2e-6)
    out["solver"]["atol"] = min(float(out["solver"].get("atol", 1e-7)), 2e-7)
    out["model"]["internal_correlation"] = mode
    return out


def _final_row(result: Any, *, correlation: str, city: str, month: int | None,
               eta_ref: float | None, tout_ref: float | None) -> dict[str, Any]:
    k = len(result.t_s) - 1
    nd = result.node_diag
    sd = result.scalar_diag
    re = np.asarray(nd["Re_internal"][k], dtype=float)
    pr = np.asarray(nd["Pr_internal"][k], dtype=float)
    nu = np.asarray(nd["Nu_internal"][k], dtype=float)
    h = np.asarray(nd["h_internal_W_m2K"][k], dtype=float)
    gz = np.asarray(nd["Graetz_internal"][k], dtype=float) if "Graetz_internal" in nd else np.full_like(re, np.nan)
    D = float(result.config["geometry"]["D2"])
    L = float(result.config["geometry"]["L"])
    # Escala clásica orientativa del desarrollo térmico laminar.
    lth = 0.05 * re * pr * D
    lh = 0.05 * re * D
    eta = float(sd["eta_dni_basis_pct"][k])
    tout = float(result.Tout_C[k])
    return {
        "Ciudad_caso": city,
        "Mes_num": month,
        "Mes": MONTH_ABBR_ES[month - 1] if month else "—",
        "Correlacion_id": correlation,
        "Correlacion": CORRELATIONS[correlation],
        "Fluido": str(result.config["operation"]["fluid"]),
        "Re_mean": float(np.mean(re)),
        "Re_min": float(np.min(re)),
        "Re_max": float(np.max(re)),
        "Pr_mean": float(np.mean(pr)),
        "Graetz_mean": float(np.nanmean(gz)),
        "Nu_mean": float(np.mean(nu)),
        "h_mean_W_m2K": float(np.mean(h)),
        "Lth_mean_m": float(np.mean(lth)),
        "Lh_mean_m": float(np.mean(lh)),
        "L_colector_m": L,
        "Lth_sobre_L": float(np.mean(lth) / L),
        "Lh_sobre_L": float(np.mean(lh) / L),
        "Tout_C": tout,
        "Tout_ref_C": tout_ref,
        "Eta_pct": eta,
        "Eta_ref_pct": eta_ref,
        "Quseful_W": float(sd["Quseful_W"][k]),
        "Tabs_mean_C": float(result.Tabs_mean_C[k]),
        "Tglass_mean_C": float(result.Tglass_mean_C[k]),
        "Qloss_W": float(sd["Qloss_W"][k]),
        "Max_abs_dTdt_K_s": np.nan,
    }


def _metrics(rows: pd.DataFrame) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    for (city, corr_id, corr), g in rows.groupby(["Ciudad_caso", "Correlacion_id", "Correlacion"], sort=False):
        if g["Eta_ref_pct"].notna().sum() < 2:
            continue
        eta_err = g["Eta_pct"].to_numpy(float) - g["Eta_ref_pct"].to_numpy(float)
        t_err = g["Tout_C"].to_numpy(float) - g["Tout_ref_C"].to_numpy(float)
        r = float(np.corrcoef(g["Eta_pct"].to_numpy(float), g["Eta_ref_pct"].to_numpy(float))[0, 1])
        out.append({
            "Ciudad": city,
            "Correlacion_id": corr_id,
            "Correlacion": corr,
            "RMSE_eta_pp": float(np.sqrt(np.mean(eta_err**2))),
            "MAE_eta_pp": float(np.mean(np.abs(eta_err))),
            "Bias_eta_pp": float(np.mean(eta_err)),
            "r_eta": r,
            "RMSE_Tout_C": float(np.sqrt(np.mean(t_err**2))),
            "MAE_Tout_C": float(np.mean(np.abs(t_err))),
            "Re_mean": float(g["Re_mean"].mean()),
            "Nu_mean": float(g["Nu_mean"].mean()),
            "h_mean_W_m2K": float(g["h_mean_W_m2K"].mean()),
            "Lth_sobre_L_mean": float(g["Lth_sobre_L"].mean()),
            "Lh_sobre_L_mean": float(g["Lh_sobre_L"].mean()),
        })
    return pd.DataFrame(out)


def _bhambare_error_table(rows: pd.DataFrame) -> pd.DataFrame:
    # Dos benchmarks publicados. Se reporta error por magnitud; no se fuerza una
    # "mejor" referencia porque el trabajo de Bhambare no reproduce exactamente Sukhatme.
    refs = {
        "Bhambare": {"Tout_C": 154.1, "Tabs_K": 465.4, "Tglass_K": 331.4, "Qloss_W": 813.8},
        "Sukhatme": {"Tout_C": 155.34, "Tabs_K": 441.13, "Tglass_K": 333.39, "Qloss_W": 857.6},
    }
    out: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        for ref_name, ref in refs.items():
            model_vals = {
                "Tout_C": float(row["Tout_C"]),
                "Tabs_K": float(row["Tabs_mean_C"] + 273.15),
                "Tglass_K": float(row["Tglass_mean_C"] + 273.15),
                "Qloss_W": float(row["Qloss_W"]),
            }
            rel = {k: 100.0 * (model_vals[k] - ref[k]) / ref[k] for k in ref}
            rmsre = math.sqrt(sum((v / 100.0) ** 2 for v in rel.values()) / len(rel)) * 100.0
            out.append({
                "Referencia": ref_name,
                "Correlacion_id": row["Correlacion_id"],
                "Correlacion": row["Correlacion"],
                "Re_mean": row["Re_mean"],
                "Pr_mean": row["Pr_mean"],
                "Graetz_mean": row["Graetz_mean"],
                "Nu_mean": row["Nu_mean"],
                "h_mean_W_m2K": row["h_mean_W_m2K"],
                "Lth_sobre_L": row["Lth_sobre_L"],
                "Tout_C": model_vals["Tout_C"],
                "Tabs_K": model_vals["Tabs_K"],
                "Tglass_K": model_vals["Tglass_K"],
                "Qloss_W": model_vals["Qloss_W"],
                "Error_Tout_pct": rel["Tout_C"],
                "Error_Tabs_pct": rel["Tabs_K"],
                "Error_Tglass_pct": rel["Tglass_K"],
                "Error_Qloss_pct": rel["Qloss_W"],
                "RMSRE_multivariable_pct": rmsre,
            })
    return pd.DataFrame(out)


def run_internal_correlation_audit(
    fluid_database: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    monthly_rows: list[dict[str, Any]] = []
    for city, source in (("Foz", REA_FOZ_MONTHLY), ("Alvorada", REA_ALVORADA_MONTHLY)):
        for month in range(1, 13):
            cfg, _ = build_rea_monthly_preset(city.lower(), month)
            for corr in CORRELATIONS:
                result = PTCSimulator(_fast_cfg(cfg, corr), fluid_database).simulate()
                monthly_rows.append(_final_row(
                    result,
                    correlation=corr,
                    city=city,
                    month=month,
                    eta_ref=float(source["eta_ref_pct"][month - 1]),
                    tout_ref=float(source["Tout_ref_C"][month - 1]),
                ))
    monthly = pd.DataFrame(monthly_rows)
    metrics = _metrics(monthly)

    bh_cfg, _ = build_bhambare_sukhatme_preset()
    bh_rows: list[dict[str, Any]] = []
    for corr in CORRELATIONS:
        result = PTCSimulator(_fast_cfg(bh_cfg, corr), fluid_database).simulate()
        bh_rows.append(_final_row(
            result, correlation=corr, city="Bhambare", month=None,
            eta_ref=None, tout_ref=None,
        ))
    bh = pd.DataFrame(bh_rows)
    bh_errors = _bhambare_error_table(bh)

    # Tabla compacta de régimen: suficiente para responder si el agua está en el
    # mismo problema de régimen que el aceite.
    regime = pd.concat([
        monthly.groupby("Ciudad_caso", as_index=False).agg(
            Fluido=("Fluido", "first"),
            Re_min=("Re_min", "min"), Re_mean=("Re_mean", "mean"), Re_max=("Re_max", "max"),
            Pr_mean=("Pr_mean", "mean"), Graetz_mean=("Graetz_mean", "mean"),
            Lth_sobre_L=("Lth_sobre_L", "mean"),
            Lh_sobre_L=("Lh_sobre_L", "mean"),
        ),
        pd.DataFrame([{
            "Ciudad_caso": "Bhambare",
            "Fluido": str(bh.iloc[0]["Fluido"]),
            "Re_min": float(bh.iloc[0]["Re_min"]),
            "Re_mean": float(bh.iloc[0]["Re_mean"]),
            "Re_max": float(bh.iloc[0]["Re_max"]),
            "Pr_mean": float(bh.iloc[0]["Pr_mean"]),
            "Graetz_mean": float(bh.iloc[0]["Graetz_mean"]),
            "Lth_sobre_L": float(bh.iloc[0]["Lth_sobre_L"]),
            "Lh_sobre_L": float(bh.iloc[0]["Lh_sobre_L"]),
        }])
    ], ignore_index=True)

    # Identifica cuál rama reduce RMSE en ambas ciudades sin decidir todavía que
    # deba adoptarse físicamente.
    pivot = metrics.pivot(index="Correlacion_id", columns="Ciudad", values="RMSE_eta_pp")
    combined = {}
    for corr in CORRELATIONS:
        if corr in pivot.index:
            combined[corr] = float(np.sqrt(np.mean([pivot.loc[corr, "Foz"]**2, pivot.loc[corr, "Alvorada"]**2])))
    best = min(combined, key=combined.get) if combined else None

    water_re_max = float(monthly["Re_max"].max())
    water_lth = float(monthly["Lth_sobre_L"].mean())
    bh_re = float(bh.iloc[0]["Re_mean"])
    if water_re_max < 2300.0 and water_lth > 0.5:
        diagnosis = (
            f"El agua de Rea Quille también está inequívocamente en régimen laminar (Re máximo≈{water_re_max:.0f}), "
            f"y la escala de entrada térmica estimada es comparable o mayor que una fracción importante del colector (Lth/L medio≈{water_lth:.2f}). "
            "Por tanto, el problema no puede atribuirse exclusivamente a propiedades de Paratherm NF: la elección entre Nu=4.36 plenamente desarrollado y una correlación laminar en desarrollo afecta también a las validaciones con agua. "
            f"Bhambare queda cerca del borde laminar con las propiedades actuales (Re≈{bh_re:.0f}), por lo que Dittus–Boelter forzado se conserva aquí únicamente como control documental."
        )
    else:
        diagnosis = (
            "La auditoría muestra que la selección de régimen/correlación cambia de forma apreciable entre los casos. "
            "Debe elegirse la correlación a partir de su dominio de validez antes de ajustar propiedades del fluido."
        )

    return {
        "correlations": CORRELATIONS.copy(),
        "monthly_table": monthly,
        "metrics_table": metrics,
        "regime_table": regime,
        "bhambare_table": bh,
        "bhambare_errors": bh_errors,
        "best_combined_id": best,
        "best_combined_label": CORRELATIONS.get(best, "—") if best else "—",
        "combined_rmse": combined,
        "diagnosis": diagnosis,
        "note": (
            "Hausen y Sieder–Tate se incluyen como hipótesis diagnósticas de desarrollo laminar. "
            "Nu=4.36 representa el límite plenamente desarrollado con flujo de calor uniforme. "
            "Dittus–Boelter forzado se muestra como control, no como recomendación cuando Re está fuera de su dominio turbulento."
        ),
    }
