# Correspondencia MATLAB → Python

| MATLAB | Python | Función |
|---|---|---|
| `crearConfiguracionBase` | `defaults.default_config` | Parámetros geométricos, ópticos, ambientales y del solver |
| `propiedadesParathermNF` / `propiedadesAgua` | `fluid_properties.FluidPropertyEvaluator` | Propiedades dependientes de temperatura |
| `modeloSolar` | `PTCSimulator.solar_model` | DNI, ángulo, IAM, pérdidas de extremo y potencia óptica |
| `coefConveccionInterna` | `ptc_model.internal_convection` | Laminar, Dittus-Boelter y Gnielinski-Forristall |
| `coefConveccionExterna` | `ptc_model.external_convection` | Churchill-Bernstein sobre cilindro |
| `coefConveccionAnular` | `ptc_model.annulus_convection` | Vacío ideal, vacío efectivo o aire |
| `calcularFlujos` | `PTCSimulator.calculate_fluxes` | Balances y flujos por nodo |
| `rhsPTC` | `PTCSimulator.rhs` | Sistema de EDO |
| `ode15s` | `solve_ivp(method="BDF")` | Integración rígida |
| `validarBhambare` | `validations.validate_bhambare` | Comparación Bhambare/Sukhatme |
| `validarTCCMensual` | `validations.validate_tcc_monthly` | Comparación mensual TCC/TRNSYS |
| `reproducirValidacionPrototipoTCC` | `validations.prototype_tcc_table` | Tabla 8 del prototipo |

## Extensiones añadidas

- almacenamiento de todos los flujos y resistencias por nodo y por instante;
- selección interactiva de tiempo y nodo;
- derivadas locales `dTf/dt`, `dTabs/dt` y `dTvid/dt`;
- esquema del colector y su discretización axial;
- red de resistencias actualizada con los valores del nodo seleccionado;
- perfiles de irradiación editables;
- tablas de propiedades editables mediante PCHIP;
- exportación JSON, CSV y XLSX;
- barrido comparativo de caudales.

## Corrección conservada en la migración

La variable `PrWall` pertenece exclusivamente a la convección interna. La función de convección externa no la calcula ni la utiliza.


## Revisión 2026-08-24: agua y transición de Reynolds

- Se eliminó el salto discontinuo Nu=4.36 -> Gnielinski en Re=2300.
- La zona 2300-4000 usa mezcla smoothstep configurable.
- Se agregaron eta_optical_abs_pct, eta_balance_pct, Qstorage_est_W y transition_weight.
- El dashboard muestra la eficiencia óptica como referencia para interpretar la eficiencia térmica del HTF.

## 2026-08-24 — Presets de referencia

Se añadió `presets.py` para evitar mezclar parámetros de Bhambare con el prototipo de Fiamonzini/Rea Quille. Los presets documentales cargan toda la configuración de una sola vez y guardan la referencia y los parámetros no publicados en `config["preset_meta"]`.
