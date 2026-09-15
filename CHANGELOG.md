# CHANGELOG consolidado

Historial único del proyecto PTC Python Interactivo. Los antiguos `CHANGELOG_V*.txt` fueron integrados aquí en V14.9 para reducir archivos redundantes dentro del ZIP.


---

## V8

PTC Python Interactivo — V8

Implementación solicitada: comparación automática de solvers para el caso Bhambare/Sukhatme.

Novedades:
- Botón en Validación: "Comparar RK45 · Radau · BDF — Bhambare/Sukhatme".
- Ejecuta exactamente el mismo preset con los tres integradores.
- Tabla final contra Sukhatme y contra los valores reportados por Bhambare.
- Tabla de resultados por solver con errores relativos frente a Sukhatme.
- Gráfico superpuesto de Tout, Tabs, Tvid, Qloss, Qutil y eta durante toda la simulación.
- Diagnóstico de costo numérico: tiempo CPU, nfev, njev y nlu.
- Residuo térmico final max|dT/dt| para evaluar aproximación a régimen estacionario.
- Métricas de dispersión entre solvers y exportación CSV.

Se conserva la factorización exacta de los términos radiativos T1^4 - T2^4 introducida en V7.


---

## V9

V9 — Diagnóstico de sensibilidad y convergencia

- Nueva pestaña "Sensibilidad".
- Análisis automático de convergencia numérica Bhambare/Sukhatme:
  * independencia de malla N = 1..48;
  * sensibilidad a max_step;
  * sensibilidad a rtol/atol;
  * sensibilidad al tiempo de calentamiento / residuo estacionario.
- Sensibilidad física OAT ±10 %:
  * óptica: reflectividad, interceptación, transmitancia, absortancia;
  * pérdidas: emisividades;
  * ambiente: viento y Tsky;
  * entradas publicadas: mdot y DNI como controles;
  * propiedades Paratherm NF: Cp, mu, k, rho.
- Score RMS frente a Tout, Tabs, Tvid y Qloss de Sukhatme.
- Ranking tipo tornado de parámetros que más reducen la discrepancia.
- Exportación CSV de ambos análisis.


---

## V10

V10 — Identificación multiparámetro / modelo inverso

- Nueva sección 3 en la pestaña Sensibilidad.
- Selector de referencia:
  * Bhambare / Sukhatme — Table 4.
  * Rea Quille — Foz do Iguaçu — Tabela 10.
  * Rea Quille — Alvorada do Norte — Tabela 11.
  * Rea Quille / Fiamonzini — prototipo — Tabela 8 (exploratorio).
- Optimización acotada con scipy.optimize.least_squares.
- Factor óptico efectivo agrupado para evitar intentar separar rho/gamma/tau/alpha sin identificabilidad suficiente.
- Parámetros opcionales: emisividades, viento, Tdp / Tsky, pérdidas de soportes y multiplicadores de propiedades HTF.
- Para Rea mensual: opción 4 meses de calibración (Ene/Abr/Jul/Oct) + 8 meses hold-out, o los 12 meses para ajuste global.
- Diagnóstico de identificabilidad mediante rango y número de condición del Jacobiano.
- Flags de parámetros próximos a límites físicos.
- Revalidación final a N completo después de una búsqueda acelerada a N=6.
- Exportación XLSX con parámetros identificados y comparación antes/después.


---

## V11

V11 — Rea Quille/Fiamonzini: doble validación + template CSV

- Se separa la validación del prototipo en dos modos:
  1) TRNSYS publicado: DNI=905 W/m², θ=0, IAM=1, sin tracking.
  2) Físico corregido: colector fijo con apertura horizontal/eje N-S, incidencia solar, IAM y EndLoss variables.
- El modo físico permite mantener DNI nominal=905 para aislar la geometría o usar el modelo de cielo claro A·exp(-B/cos z). El TCC no publica DNI horario medido.
- Se incorpora 2026-09-14T12-26_export.csv como benchmark/template exacto de salida.
- La validación puede comparar contra: Tabela 8 experimental, Tabela 8 TRNSYS, CSV modelo inicial o CSV modelo identificado.
- Se añade botón para aplicar cualquiera de los dos modos al simulador principal y ejecutarlo desde la pestaña Simulación.
- El template CSV se puede visualizar y descargar desde la app.
- No se inventan parámetros identificados: el CSV suministrado no contiene esos valores, solo las curvas de salida.
- Nueva rama solar `fijo_horizontal` para modelar un colector sin tracking mediante ángulo cenital.


---

## V11_1

PTC Python Interactivo V11.1

Corrección de Streamlit Session State:
- Corrige StreamlitWidgetAlreadyInstantiatedError al usar "Aplicar este modo al simulador".
- Las claves de widgets del preset ya no se modifican después de crear los widgets.
- Se usa un estado pendiente (_pending_widget_state) que se consume al inicio del siguiente rerun.
- Mantiene sin cambios la lógica física, validaciones, presets y template CSV de V11.


---

## V11_2

V11.2 — templates identificados y diagnóstico de generalización

- El objetivo "Template CSV · modelo identificado" aplica ahora realmente los parámetros encontrados por el modelo inverso del 14/09/2026, en vez de cambiar solo la curva objetivo.
- Rea/Fiamonzini Tabela 8: eta_opt_eff=0.5583394106143328 y viento=6.32658029702888 m/s.
- La curva identificada exportada se reproduce con RMSE < 1e-6 pp.
- Se incorporaron también como metadatos los templates identificados de Rea Foz y Rea Alvorada para trazabilidad.
- La validación por modo permite seleccionar parámetros nominales o identificados; para csv_identified la aplicación es automática.
- "Aplicar este modo al simulador" conserva los parámetros identificados cuando corresponde.
- El modelo inverso mensual muestra cuántos meses hold-out mejoran/empeoran y advierte cuando la generalización es mixta.
- Suite: 8 tests OK.


---

## V12

V12 — irradiación temporal + colector fijo sin tracking

- Nuevo DNI temporal para Rea/Fiamonzini: cielo claro A·exp(-B/cos z) reescalado para alcanzar 905 W/m² al mediodía solar.
- Conversión opcional de hora civil a hora solar aparente usando longitud, UTC local y ecuación del tiempo.
- El colector permanece fijo: theta, cos(theta), irradiancia proyectada, IAM y EndLoss varían con la hora.
- Se separan DNI y irradiancia realmente proyectada sobre la apertura: G_apertura = DNI*cos(theta).
- Nueva eficiencia de comparación Rea Eq.(10): eta_DNI = Qutil/(Aa*DNI). Esta métrica incluye la penalización geométrica de un colector fijo y se usa en la validación Rea/Fiamonzini.
- Se conserva eta_pct histórica, referida a la potencia ya proyectada sobre la apertura, para compatibilidad y diagnóstico.
- Export de validación ampliado con DNI, G_apertura, hora solar aparente, theta, IAM, EndLoss y ambas eficiencias.
- 10/10 tests OK.


---

## V13

V13 — Validación Bhambare con métricas de error

- Nueva sección en Validación: “Bhambare / Sukhatme — validación por objetivo”.
- Objetivos seleccionables:
  * Sukhatme & Nayak (referencia)
  * Bhambare (modelo publicado)
  * XLSX 14/09 — modelo inicial
  * XLSX 14/09 — modelo identificado
- Parámetros seleccionables: nominales o identificados del modelo inverso.
- Template identificado Bhambare incorporado:
  eta_opt_eff = 0.7671242147795794
  eps_abs     = 0.989999998086766
  eps_glass   = 0.9899999947177927
- Métricas globales normalizadas: MAPE multivariable, RMSRE, bias relativo medio y error máximo.
- Tabla por magnitud con valor objetivo, Python, diferencia, error absoluto y error relativo.
- Botón para aplicar parámetros nominales/identificados al simulador.
- XLSX original incorporado en templates/ptc_modelo_inverso_bhambare.xlsx.
- Los targets XLSX fuerzan BDF para reproducir el integrador usado durante la identificación.
- 14 tests aprobados.


---

## V13_1

V13.1 — Robustez frente a valores opcionales None en validaciones

- Corrige AttributeError: 'NoneType' object has no attribute 'get'.
- Bhambare: applied_parameters puede ser None en modo nominal; la UI ahora lo normaliza a {} antes de consultar near_bounds.
- Rea/Fiamonzini: preset_meta se normaliza con `or {}` para tolerar sesiones/configuraciones antiguas donde la clave exista con valor None.
- No se modifica ninguna ecuación, parámetro físico, preset, métrica o resultado numérico.


---

## V14

PTC Python Interactivo — V14
============================

Reorganización mayor de interfaz y flujo de validación.

1. Navegación principal reducida a 4 secciones:
   - Simulación
   - Propiedades
   - Validación
   - Sensibilidad

2. Simulación
   - Botón "Ejecutar simulación" movido al encabezado superior derecho.
   - Eliminada la comparación rápida con la referencia del preset.
   - Si la fuente solar absorbida es constante, no se muestra el dashboard transitorio; se presentan directamente estado representativo/perfiles axiales y, para barridos, una tabla comparativa estacionaria.
   - Reporte y exportación pasan a ser una subpestaña de Simulación.

3. Propiedades
   - Irradiación, cielo, perfil horario y propiedades del HTF quedan en una sección propia.

4. Validación
   - Flujo principal cambiado a calibración + hold-out.
   - Enero, abril, julio y octubre calibran parámetros.
   - Los otros 8 meses quedan completamente fuera del ajuste y se usan para validación.
   - Botón para aplicar al simulador los parámetros recién calibrados.
   - Métricas hold-out: score relativo, RMSE/MAE/MAPE/bias y error relativo máximo.
   - Benchmarks Bhambare/Sukhatme y Rea/Fiamonzini quedan como diagnóstico documental secundario, no como validación fuera de muestra.

5. Sensibilidad
   - Se conserva convergencia numérica y sensibilidad física local.
   - Se retira el modelo inverso de esta sección.
   - Texto corregido: sensibilidad = cuánto cambian las salidas al perturbar parámetros; no es calibración.

6. Reporte y exportación
   - Añadidos indicadores de error de la última validación hold-out.
   - El XLSX agrega hojas Errores_validacion, Detalle_validacion y Parametros_calibrados cuando están disponibles.

Pruebas: 15/15 tests automáticos aprobados.


---

## V14_1

PTC Python Interactivo — V14.1

Correcciones de compatibilidad Streamlit 1.56+:
- Se asignan keys explícitas y únicas a todos los st.plotly_chart.
  Esto evita StreamlitDuplicateElementId cuando dos pestañas renderizan axial_profiles en el mismo rerun.
- Se sustituye use_container_width=True por width="stretch".
- Se sustituye components.html por st.iframe para los visuales HTML/JS del PTC y circuito térmico.
- requirements.txt eleva Streamlit mínimo a 1.56.0.

No se modificaron ecuaciones, correlaciones, calibraciones ni resultados físicos.


---

## V14_2

V14.2 — Ventana Gráficos de validación

- Nueva sección principal Gráficos.
- Curvas mensuales referencia / inicial / calibrado para eficiencia y Tout.
- Bandas visuales identifican meses usados para calibrar; el resto son hold-out.
- Barras de error relativo absoluto antes/después sólo para hold-out.
- Scatter predicción vs referencia con línea ideal y=x.
- Residuos firmados para detectar sesgo.
- Tabla exacta detrás de los gráficos y descarga CSV.
- Botón Abrir gráficos desde la sección Validación.
- Todos los plotly_chart nuevos usan keys únicas.


---

## V14_3

V14.3 · Prueba simple Eq. (10) desde Tout

- Añade en Validación una prueba diagnóstica sin optimización para Rea Quille Foz/Alvorada.
- Para cada mes usa Tin, Tamb, DNI y mdot publicados, ejecuta el simulador y toma sólo Tout.
- Reconstruye η con la Ec. (10): mdot*Cp*(Tout-Tin)/(Aa*DNI).
- Recalcula también η de referencia desde Tout publicado para auditar consistencia de Tablas 10/11.
- Compara η reconstruida con el KPI interno del modelo para determinar si la definición de eficiencia altera la curva.
- Incluye dos gráficos, métricas, tabla completa y exportación CSV.
- Puede ejecutarse con parámetros nominales o con la última calibración guardada de la misma ciudad, sin recalibrar.


---

## V14_3_1

V14.3.1 — Hotfix de importación Ec. (10)

- Corrige ImportError: validate_rea_eq10_from_tout.
- La prueba Ec. (10) ahora vive en rea_eq10_validation.py y app.py la importa desde ese módulo independiente.
- Evita que una actualización parcial/stale de validations.py impida arrancar toda la aplicación.
- No cambia ecuaciones, resultados, calibraciones ni gráficos.


---

## V14_4

V14.4 — Auditoría mensual del balance energético

- Se retira de la interfaz la prueba simple de la Ec. (10), porque ya confirmó que la definición del KPI no era la causa principal de la discrepancia estacional.
- Se añade una nueva prueba diagnóstica sin optimización para Rea Quille (Foz/Alvorada).
- La prueba ejecuta los 12 meses y descompone Qsolar absorbida, Qútil, convección externa, radiación al cielo, soportes y almacenamiento.
- Se calculan dos factores contrafactuales: F_opt y F_loss.
- F_opt pregunta cuánto debería escalarse la potencia solar absorbida si las pérdidas actuales fueran correctas.
- F_loss pregunta cuánto deberían escalarse las pérdidas si la potencia solar absorbida actual fuera correcta.
- Se muestran tres gráficos: potencias principales, desglose de pérdidas y factores requeridos mes a mes.
- Se incluyen métricas de variación mensual, meses ópticamente insuficientes, cierre de balance y exportación CSV.
- La prueba puede usar parámetros nominales o aplicar una calibración ya existente, pero nunca recalibra.


---

## V14_5

V14.5 — Auditoría de pérdidas por mecanismo

- Retira de la interfaz la prueba global F_opt/F_loss de V14.4.
- Añade diagnóstico mensual convección externa vs radiación al cielo.
- Calcula k_conv y k_rad requeridos mes a mes manteniendo el otro mecanismo fijo.
- Calcula el mejor multiplicador global LS para cada mecanismo por separado, sin aplicarlo al modelo.
- Compara RMSE de eficiencia: modelo actual, corrección convectiva global y corrección radiativa global.
- Muestra participación media de Qconv y Qrad, coeficientes de variación e imposibilidad física (factor < 0).
- Añade tres gráficos: magnitud de pérdidas, factores requeridos y contrafactual de eficiencia.
- No cambia ecuaciones, correlaciones ni parámetros del simulador.


---

## V14_5_1

V14.5.1 — Hotfix de importación de gráficos de auditoría

- Los tres gráficos de la auditoría convección vs radiación fueron movidos a
  `rea_loss_component_charts.py`.
- `app.py` ya no depende de que `visualizations.py` contenga esas funciones.
- Esto evita el ImportError observado cuando Streamlit/GitHub despliega una copia
  anterior o cacheada de `visualizations.py`.
- No cambia ninguna ecuación, correlación ni resultado físico de V14.5.


---

## V14_6

V14.6 — Auditoría de flujo externo

- Retira de la interfaz la auditoría previa convección vs radiación de V14.5.
- Añade `rea_external_flow_audit.py` para abrir el término convectivo externo mes a mes.
- Reporta T de superficie, T de película, propiedades del aire, Re, Pr, Nu y h.
- Calcula h requerido para cerrar el balance de Rea Quille con el resto de mecanismos congelados.
- Calcula el viento implícito requerido usando exactamente la correlación Churchill–Bernstein del modelo.
- Añade cuatro gráficos independientes en `rea_external_flow_charts.py`.
- No modifica la correlación, el viento del preset ni ningún parámetro físico; es una etapa puramente diagnóstica.


---

## V14_7

V14.7 — Hipótesis de viento meteorológico mensual

- Se retira de la interfaz la auditoría de flujo externo V14.6.
- Nueva prueba diagnóstica: viento fijo de 1 m/s vs viento mensual independiente.
- Foz incluye template WeatherSpark / NASA MERRA-2 de velocidad media horaria a 10 m:
  3.7, 3.6, 3.6, 3.9, 4.0, 4.1, 4.3, 4.4, 4.4, 4.3, 4.0, 3.8 mph.
- Opción de consultar NASA POWER WS10M climatológico online para Foz o Alvorada.
- Serie mensual siempre editable manualmente.
- Opción de trasladar v10 a altura efectiva del receptor mediante ley de potencia
  v(z)=v10*(z/10)^alpha, con z y alpha explícitos y editables.
- La prueba compara RMSE, correlación, Tout y Qconv sin calibrar ninguna constante.
- El viento se elimina de los parámetros calibrables de las validaciones mensuales Rea.
- La prueba usa N=6 y BDF para acelerar el diagnóstico; no altera el modelo guardado.


---

## V14_8

V14.8 — Prueba diagnóstica de agregación temporal de irradiancia

- Se retira de la interfaz la prueba V14.7 de viento meteorológico mensual.
- El viento vuelve a quedar congelado en esta etapa: la hipótesis nueva modifica únicamente la distribución temporal del DNI.
- Nueva prueba controlada f(DNI medio) vs promedio[f(DNI(t))].
- Para cada mes se construye un día solar representativo y se escala el perfil para conservar exactamente el DNI medio publicado por Rea Quille.
- Tin, Tamb, caudal, viento, óptica e incidencia permanecen invariantes entre escenarios.
- Cada punto del día se resuelve cuasiestacionariamente con N=6; la eficiencia temporal se calcula a partir de la potencia útil media y la misma energía solar media.
- Nuevos gráficos: perfil de DNI, eficiencia mensual, Tout mensual y gap de agregación/Jensen.
- Exportación CSV del resumen mensual y de todos los puntos temporales.


---

## V14.9 — Hipótesis de apantallamiento aerodinámico

- Se retira de la interfaz la prueba V14.8 de agregación temporal del DNI, cuyo efecto resultó despreciable.
- Se añade una prueba controlada con `v_eff = S_v * v_amb`.
- `S_v` se identifica UNA sola vez usando exclusivamente Ene/Abr/Jul/Oct de Foz do Iguaçu.
- El factor queda congelado para los 8 meses hold-out de Foz y para los 12 meses de Alvorada do Norte; no existe recalibración por ciudad.
- La función objetivo utiliza eficiencia térmica mensual; `Tout` se conserva como comprobación independiente de magnitud.
- Se muestran curva objetivo de `S_v`, RMSE antes/después, curvas mensuales de Foz y Alvorada y exportación CSV.
- El ensayo conserva `v_amb = 1 m/s` para aislar sólo el posible apantallamiento geométrico del receiver. No reconstruye meteorología real.
- Todos los changelogs históricos se consolidan en este único `CHANGELOG.md`.

### Limpieza del paquete V14.9

Se retiraron del ZIP completo los módulos y tests de las pruebas diagnósticas ya descartadas (auditoría de pérdidas por componente, auditoría de flujo externo y agregación temporal). El historial y las conclusiones de esas etapas permanecen en este changelog consolidado.
## V14.10 — Auditoría radial Bhambare/Sukhatme

- Se cerró la comprobación de solver: la comparación externa MATLAB (ode45/ode15s/ode23t) y Python converge al mismo estado, por lo que el integrador deja de tratarse como hipótesis activa.
- Se retiró de la interfaz la prueba diagnóstica de apantallamiento aerodinámico V14.9.
- Nueva prueba única en Validación: **cierre radial Bhambare/Sukhatme**.
- La prueba impone las temperaturas de vidrio publicadas y recalcula Qloss con exactamente el bloque externo actual (Churchill–Bernstein + radiación al cielo).
- También calcula la Tglass que necesitaría el bloque externo para reproducir Qloss de Bhambare y Sukhatme.
- Se audita el balance aproximado del vidrio con los pares publicados (Tabs, Tglass) y el intercambio radiativo absorbedor→vidrio.
- No se calibra ningún parámetro: la prueba sirve para decidir si la discrepancia está en el bloque externo o aguas arriba del vidrio.
- Resultado de control con el preset nominal: Tglass requerida ≈332.18 K para Qloss de Bhambare (publicada 331.40 K) y ≈333.66 K para Qloss de Sukhatme (publicada 333.39 K). Esto indica que el bloque externo puede reproducir las pérdidas publicadas si se alcanza la Tglass correspondiente.
- El par Bhambare cierra el balance del vidrio dentro de ~3.3 %, mientras el par Sukhatme deja un residual aproximado de -29.5 % con las ecuaciones de Bhambare implementadas, señal de incompatibilidad estructural entre ambos benchmarks bajo el mismo circuito radial.
- Se mantienen un único `CHANGELOG.md` y se eliminan del paquete los módulos de la hipótesis diagnóstica anterior.


---

## V14.11 — Auditoría absorbedor → HTF · Bhambare/Sukhatme

- Se retira de la interfaz la auditoría radial V14.10 una vez comprobado que el bloque externo puede reproducir los Qloss publicados al imponer Tglass.
- Nueva prueba diagnóstica centrada en la transferencia interna absorbedor→HTF.
- Se ejecuta el mismo caso Bhambare con dos ramas controladas:
  - Dittus–Boelter forzado, tal como está configurado el preset documental;
  - selección automática de régimen, que con las propiedades actuales cae en Nu=4.36.
- Se reportan perfiles axiales de Re, Pr, Nu, h, Qfluid y propiedades del Paratherm NF.
- Se infiere un h/Nu global equivalente para Bhambare y Sukhatme a partir de Tabs y Tout publicados mediante LMTD, sin calibrar el modelo.
- La interfaz advierte explícitamente que Bhambare publica Cp pero no tabula rho, mu ni k del Paratherm NF usado en la referencia [15], por lo que los equivalentes se calculan con la base de propiedades actual y no representan una reconstrucción exacta del aceite del artículo.
- Diagnóstico esperado del caso nominal actual: Re medio ≈ 2180; Dittus–Boelter forzado produce h ≈ 134 W/m²K; la rama automática laminar ≈ 14 W/m²K; Bhambare exige un h equivalente intermedio y Sukhatme uno superior al Dittus actual.
- Se eliminan del paquete los módulos/test de la auditoría radial V14.10.
- El historial continúa concentrado exclusivamente en `CHANGELOG.md`.

---

## V14.12 — Auditoría directa de régimen/correlación interna

- Se retira de la interfaz y del paquete la auditoría V14.11 absorbedor→HTF basada en h equivalentes de Bhambare/Sukhatme.
- La hipótesis activa pasa a ser exclusivamente **régimen/correlación interna**, sin recalibrar propiedades de ningún fluido.
- La prueba incluye simultáneamente agua (Rea Quille: Foz y Alvorada) y Paratherm NF (Bhambare), de modo que el diagnóstico no dependa de propiedades desconocidas del aceite.
- Se añaden ramas diagnósticas a `internal_convection`: Nu=4.36 plenamente desarrollado, Hausen laminar con corrección de entrada, Sieder–Tate laminar de entrada y Dittus–Boelter forzado como control documental.
- Se añade el número de Graetz y estimaciones de longitud térmica/hidrodinámica de entrada.
- Se ejecutan las 12 condiciones mensuales de Foz y Alvorada con cada correlación, manteniendo idénticas propiedades y entradas; se reportan RMSE de eficiencia, RMSE de Tout, bias y correlación temporal.
- Se repite el caso Bhambare con las mismas cuatro ramas para comparar el efecto del régimen sin atribuirlo a cambios de propiedades.
- La prueba es diagnóstica: ninguna correlación se adopta automáticamente como definitiva.
- Se mantiene un único `CHANGELOG.md`.
