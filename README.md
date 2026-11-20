# PTC nodal en Python

Aplicación interactiva para simular un colector solar cilindro-parabólico con discretización axial y red térmica radial. Incluye:

- balance transitorio de HTF, absorbedor y cubierta de vidrio en cada nodo;
- propiedades `rho(T)`, `mu(T)`, `Cp(T)` y `k(T)`;
- convección interna laminar, Dittus-Boelter y Gnielinski-Forristall;
- radiación absorbedor-vidrio y superficie-cielo;
- convección exterior y tres opciones para el espacio anular;
- solver seleccionable SciPy `RK45`, `Radau` o `BDF`;
- edición de geometría, óptica, ambiente, irradiación y propiedades de fluidos;
- visualización nodo por nodo, esquema del PTC y red de resistencias con flujos térmicos;
- validaciones Bhambare/Sukhatme, TCC/TRNSYS y Tabla 8 del prototipo;
- comparación automática RK45/Radau/BDF para Bhambare/Sukhatme, incluyendo error frente a la referencia, costo numérico y residuo térmico final.

Las ecuaciones y parámetros predeterminados se basan en:

1. `A_SOLAR_PARABOLIC_TROUGH_CONCENTRATOR_PT.pdf`.
2. `Análise e comparação do desempenho térmico de coletores solares planos e parabólicos no TRNSYS.pdf`.
3. Forristall (2003), NREL/TP-550-34169.

## 1. Abrir el proyecto en VSCode

1. Descomprima la carpeta `PTC_Python_Interactivo`.
2. Abra VSCode.
3. Seleccione **File > Open Folder** y elija la carpeta descomprimida.
4. Instale la extensión oficial **Python** de Microsoft si todavía no la tiene.

## 2. Crear el entorno virtual en Windows

Abra **Terminal > New Terminal** en VSCode y ejecute:

```powershell
py -3.11 -m venv .venv
```

Active el entorno:

```powershell
.\.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea la activación, habilítela solamente para esa terminal:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

Instale las dependencias:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Después, presione `Ctrl+Shift+P`, busque **Python: Select Interpreter** y seleccione el Python de `.venv`.

## 3. Ejecutar la interfaz

En la terminal activada:

```powershell
python -m streamlit run app.py
```

Streamlit abrirá la aplicación en el navegador. Después de instalar las dependencias una vez, también puede iniciar con doble clic en:

```text
run_app.bat
```

Desde VSCode también puede abrir **Run and Debug** y seleccionar **Ejecutar interfaz PTC (Streamlit)**.

## 4. Flujo de uso

La interfaz V14.12 se organiza en cinco secciones principales:

1. **Simulación**: resultados, análisis nodo por nodo y reporte/exportación. El botón **Ejecutar simulación** permanece en el encabezado superior derecho. Si la potencia solar absorbida es constante, la app omite el dashboard transitorio y muestra directamente el estado representativo/perfiles axiales.
2. **Propiedades**: diagnóstico de irradiación, temperatura efectiva del cielo, perfil horario editable y propiedades termofísicas del HTF.
3. **Validación**: calibración con enero/abril/julio/octubre y validación fuera de muestra con los ocho meses restantes. Los parámetros calibrados pueden aplicarse al simulador sin recalibrar contra el hold-out.
4. **Gráficos**: auditoría visual de la última calibración/validación hold-out.
5. **Sensibilidad**: convergencia numérica y perturbación física local de parámetros. Esta sección mide qué parámetros cambian más las salidas; no realiza calibración.

Los benchmarks Bhambare/Sukhatme y Rea/Fiamonzini se mantienen como diagnósticos documentales secundarios dentro de Validación, diferenciados de la validación predictiva fuera de muestra.

## 5. Guardar parámetros y resultados

En **Reporte y exportación**:

- **Guardar proyecto JSON** conserva configuración y propiedades editadas.
- **Cargar proyecto JSON** restaura esos valores en otra sesión.
- **Descargar reporte TXT** genera el reporte técnico en texto plano.
- **Exportar resultados XLSX** guarda las series globales, el estado nodal final y, cuando existe una validación hold-out, sus indicadores de error, detalle y parámetros calibrados.

Los cambios efectuados directamente en los archivos `.py` se guardan normalmente con `Ctrl+S` en VSCode.

## 6. Estructura del proyecto

```text
PTC_Python_Interactivo/
├── app.py                  Interfaz Streamlit
├── ptc_model.py            Solvers RK45/Radau/BDF, balances y correlaciones
├── fluid_properties.py     Propiedades termofísicas
├── defaults.py             Parámetros y tablas de referencia
├── visualizations.py       Gráficos, PTC y red de resistencias
├── validations.py          Casos de validación
├── technical_report.py     Reporte técnico de consola/TXT
├── requirements.txt        Dependencias
├── run_app.bat             Inicio rápido en Windows
├── run_app.sh              Inicio rápido en Linux/macOS
├── .vscode/                Configuración de ejecución en VSCode
└── tests/test_smoke.py      Prueba rápida del núcleo numérico
```

## 7. Probar solamente el modelo numérico

```powershell
python tests\test_smoke.py
```

La prueba ejecuta un caso corto sin abrir Streamlit.

## Nota sobre equivalencia numérica

El modelo dinámico se integra con `solve_ivp` y permite seleccionar `RK45`, `Radau` o `BDF`, manteniendo propiedades termofísicas dependientes de la temperatura. Los términos radiativos `T_1^4-T_2^4` se evalúan mediante la factorización algebraicamente exacta `(T_1-T_2)(T_1+T_2)(T_1^2+T_2^2)`.


## Revisión de régimen para agua

La versión actual evita una conmutación abrupta de Nusselt en Re=2300. En modo automático se usa Nu=4.36 en laminar, Gnielinski-Forristall en turbulento y una transición smoothstep continua entre los Reynolds configurables (2300 y 4000 por defecto). Esto elimina picos artificiales de h, temperatura y eficiencia cuando la viscosidad del agua hace cruzar el umbral durante el día.

La interfaz muestra además η térmica del HTF, η óptica al absorbedor y η integrada del período. Una η térmica instantánea cercana a 60 % puede ser físicamente válida cuando el producto óptico ronda 65 % y las pérdidas son pequeñas.

---

## Presets documentales (versión 2026-08-24)

La barra lateral comienza ahora con **Preset documental**. Al pulsar **Aplicar preset completo** se sustituyen en conjunto geometría, materiales, fluido, caudal, temperaturas, irradiación, condiciones ambientales, modelo y solver.

Presets disponibles:

- **Rea Quille — prototipo Foz 23/10/2021**: geometría del prototipo de Fiamonzini, agua, `mdot=0.0192 kg/s`, `Tamb=25 °C`, DNI nominal `905 W/m²`, latitud `-25.43816°`, receptor sin vidrio. La Tabela 8 no publica `Tin`/`Tout` horarios; el preset deja `Tin=25 °C` como hipótesis explícita y editable y la validación Python se marca como exploratoria.
- **Rea Quille — Foz do Iguaçu (Tabela 10)**: selector por mes o promedio anual. Cada fila carga `Tin`, `Tout_ref`, `Tamb`, DNI, caudal y eficiencia de referencia de la tabla.
- **Rea Quille — Alvorada do Norte (Tabela 11)**: selector por mes o promedio anual con los valores tabulados y latitud `-14.600°`.
- **Bhambare / Sukhatme — Pune 15/04**: geometría y óptica del artículo, Paratherm NF, `mdot=0.0986 kg/s`, `Tin=150 °C`, `Tamb=31.9 °C`, viento `5.3 m/s`, haz `705 W/m²`, receptor con vidrio y vacío ideal.

### Importante: valores publicados vs supuestos

Algunas magnitudes necesarias para nuestro balance nodal no están publicadas por Rea Quille (por ejemplo, reflectividad efectiva, factor de interceptación, viento mensual y temperatura efectiva del cielo). El programa **no las presenta como datos de la fuente**. Cada preset contiene un bloque `preset_meta.assumptions` y la interfaz los muestra en **Ver parámetros fijados y supuestos**.

La pestaña **Validación** permite:

1. validar el preset activo;
2. ejecutar los 12 casos mensuales de Foz do Iguaçu;
3. ejecutar los 12 casos mensuales de Alvorada do Norte;
4. cargar la Tabela 8 experimental/TRNSYS del prototipo.

En la pestaña **Simulación**, cuando el caso activo proviene de un preset, aparece también **Comparación rápida con la referencia del preset**.

## Modelo de temperatura efectiva del cielo — Rea Quille / Martin-Berdahl

La aplicación ya no necesita asumir por defecto `Tsky = Tamb - 6 K` para los presets de Rea Quille.
Se implementaron las Ecs. (12)-(14) descritas en la sección 3.2.7 del TCC:

- emisividad de cielo claro en función de `Tdp`, hora del día y presión;
- corrección opcional por nubosidad;
- `Tsky_K = eps_sky**0.25 * Tamb_K`.

En `Operación y ambiente` se puede escoger entre el modelo de Rea Quille y el modo legado de diferencia constante.
Las Tablas 8, 10 y 11 no publican punto de rocío ni parámetros de nubosidad, por lo que los presets de Rea usan `Tdp = 15 °C` y cielo claro como hipótesis explícita editable. El modo de nubosidad queda desactivado hasta disponer de esos datos.

**Advertencia documental:** la Ec. (13) impresa en el TCC contiene `(1 + eps0)` y el texto indica a la vez que `f_nuvem = 0` representa cielo totalmente nublado. Esas dos afirmaciones no son consistentes entre sí. La app conserva la ecuación impresa como opción literal y ofrece una variante `(1 - eps0)` únicamente para sensibilidad, claramente identificada como tal.

## Mejora visual 27-08-2026

- Sección transversal del PTC con dos modos: **Mapa de calor** y **Seguidor de rayo**.
- Selector axial con **deslizador clickeable** y gráfico auxiliar con nodos seleccionables por clic (si está instalada la dependencia `streamlit-plotly-events`).
- Circuito térmico mejorado con **flechas de flujo**, entrada de **radiación solar**, y nodos nombrados como `Tagua`, `Tabs,int`, `Tabs,ext`, `Tvid,int`, `Tvid,ext`, `Tamb` y `Tsky`.
- Se añadió `requirements.txt` para facilitar el despliegue en Streamlit Cloud / GitHub.

## Interfaz óptica HTML/CSS/JavaScript (27-08-2026)

La sección transversal del PTC ya no depende de Plotly para la interacción óptica. `interactive_visuals.py` genera un componente autocontenido que Streamlit incrusta mediante `streamlit.components.v1.html`. Incluye controles internos para ángulo de incidencia y posición del rayo, animación continua del rayo, modo de seguidor de rayo y mapa de calor óptico alrededor de la circunferencia del absorbedor. No requiere paquetes JavaScript externos ni npm.

El mapa de calor representa distribución óptica estimada de la potencia absorbida, no una solución CFD de temperatura circunferencial.

## Ajuste visual 27-08-2026 · circuito y selector axial

- Corregido el sentido visual del flujo `absorbedor -> HTF`: en el solver `Qfluid > 0` representa calor que sale del absorbedor y entra al fluido.
- Circuito térmico con estética de esquema eléctrico: cables rectos, resistencias compactas y flechas pequeñas al costado de cada resistencia.
- Selector axial sin slider: únicamente volúmenes grises grandes y clickeables, con `ΔH` y la flecha de transporte dentro de cada volumen.
- Mapa de calor óptico reforzado visualmente con una banda térmica más gruesa sobre el absorbedor.

## V6 - selector axial estable
El selector axial ya no depende de Plotly ni de `streamlit-plotly-events`. Cada volumen axial es un botón nativo de Streamlit estilizado con CSS, por lo que el clic actualiza directamente `st.session_state` y cambia el nodo activo, el circuito térmico, los KPIs y las tablas. Los bloques permanecen grandes, grises y contiguos, con la dirección y el valor de ΔH dentro de cada volumen.

## V10 — Identificación multiparámetro / modelo inverso

La pestaña **Sensibilidad** incorpora una tercera etapa: **Identificación multiparámetro (modelo inverso)**. El ajuste usa `scipy.optimize.least_squares` con límites físicos explícitos y reejecuta las referencias después de identificar los parámetros.

Casos disponibles:

- **Bhambare / Sukhatme — Table 4**: ajuste contra `Tout`, `Tabs`, `Tvid` y `Qloss`. Es calibración sobre el mismo caso y se marca como tal; no se presenta como validación independiente.
- **Rea Quille — Foz do Iguaçu — Tabela 10** y **Alvorada do Norte — Tabela 11**: puede usarse una separación estacional de 4 meses para calibración (Ene/Abr/Jul/Oct) y 8 meses `hold-out` que no participan del ajuste, o usar los 12 meses para ajuste global.
- **Rea Quille / Fiamonzini — Tabela 8**: identificación exploratoria contra la curva horaria de eficiencia. Se mantiene explícita la limitación de que Tin/Tout horarios no fueron publicados y el preset usa `Tin=25 °C` como hipótesis.

Para evitar falsa identificabilidad de factores ópticos multiplicativos, el inverso identifica por defecto un **factor óptico efectivo** `η_opt,ef = ρ·γ·τ·α·...` y lo impone variando una reflectividad equivalente mientras los demás factores permanecen fijos. También pueden seleccionarse emisividades, viento, cielo, pérdidas de soportes y multiplicadores de propiedades del HTF cuando corresponda.

El optimizador reporta:

- score inicial y final;
- parámetros nominales, identificados y límites;
- aviso si un parámetro queda cerca del límite físico;
- rango y número de condición del Jacobiano para diagnosticar identificabilidad;
- reejecución de las referencias antes/después;
- score `hold-out` para los casos mensuales cuando se usa la separación 4+8;
- exportación XLSX de parámetros y comparación.

Durante la búsqueda se usa una malla reducida `N=6` y `max_step=600 s` para acelerar la evaluación, apoyándose en el estudio previo de independencia de malla. La revalidación final vuelve a la malla completa del preset (normalmente `N=12`).

## V11 — validación del prototipo Rea Quille/Fiamonzini

La aplicación diferencia ahora entre la reproducción de la idealización TRNSYS publicada (`DNI=905 W/m²`, `IAM=1`) y una exploración física con colector fijo, incidencia horaria e IAM/EndLoss variables. La segunda opción puede mantener el DNI nominal o usar el modelo de cielo claro; como el trabajo de Rea Quille no publica una serie DNI horaria medida para la Tabela 8, esa rama se presenta explícitamente como exploratoria.

También se incorpora `templates/rea_fiamonzini_export_2026-09-14.csv`, copia exacta del CSV entregado por el usuario. Puede seleccionarse su curva `Modelo_inicial` o `Modelo_identificado` como benchmark en la pestaña Validación. El archivo no contiene los parámetros del ajuste inverso, por lo que V11 no inventa un preset de parámetros identificados.

## V12 — irradiación temporal y PTC fijo

Para el prototipo Rea Quille/Fiamonzini se incorpora un modo físico temporal sin tracking. El DNI puede modelarse con un perfil de cielo claro normalizado para alcanzar 905 W/m² al mediodía solar. El colector no sigue al Sol, por lo que la potencia sobre su apertura se calcula como `G_apertura = DNI*cos(theta)` y además se aplican `IAM(theta)` y `EndLoss(theta)`.

Para comparar con la ecuación experimental (10) de Rea Quille se usa `eta_DNI = Qutil/(Aa*DNI)`. La eficiencia histórica `eta_pct = Qutil/Qincidente_proyectada` se conserva como diagnóstico. Esta separación evita cancelar artificialmente la pérdida por coseno al evaluar un colector fijo.

## V13 — validación Bhambare por errores
La pestaña **Validación** incluye ahora una comparación multivariable para Bhambare/Sukhatme con selección de referencia documental o de los benchmarks del XLSX de identificación del 14/09/2026. Las métricas globales se calculan sobre residuos relativos normalizados porque las salidas comparadas tienen unidades distintas.


## V14.1 — compatibilidad Streamlit

Se corrigieron IDs duplicados de gráficos Plotly y APIs deprecadas de Streamlit. Requiere Streamlit >= 1.56.0. No cambia el modelo físico.


## V14.2 · Gráficos de validación
La navegación principal incluye ahora **Gráficos**, una vista de auditoría visual de la última validación hold-out: comparativas mensuales, errores antes/después, predicción vs referencia, residuos y datos descargables.


## V14.3 · Prueba simple Eq. (10) desde Tout

La sección **Validación** incorpora una prueba diagnóstica que no optimiza parámetros. Para cada mes de Rea Quille toma exactamente `Tin`, `Tamb`, `DNI` y `mdot` publicados, ejecuta el simulador, toma únicamente `Tout` y reconstruye posteriormente la eficiencia con `eta = mdot*Cp*(Tout-Tin)/(Aa*DNI)`. La interfaz grafica `Tout` y eficiencia, recalcula la eficiencia de la propia tabla para comprobar consistencia documental y compara la eficiencia Eq. (10) con el KPI interno del modelo.

La prueba puede ejecutarse con los parámetros nominales o con la última calibración ya guardada de esa misma ciudad, pero nunca vuelve a calibrar. Su objetivo es distinguir un error de definición de KPI de un error de entradas o estructura física.


## V14.4 · Auditoría mensual del balance energético

La prueba simple de la Ec. (10) se retira de la interfaz después de confirmar que el KPI de eficiencia no era la causa principal de la discrepancia estacional. En su lugar, **Validación** incorpora una auditoría energética mensual que no optimiza parámetros: ejecuta cada mes con las entradas documentales de Rea Quille, separa potencia solar absorbida, potencia útil y pérdidas externas, y calcula dos factores contrafactuales `F_opt` y `F_loss`.

La prueba está pensada para decidir qué bloque debe revisarse después: una corrección óptica global sólo es plausible si `F_opt` se mantiene aproximadamente constante entre meses; una corrección global de pérdidas sólo es plausible si `F_loss` es aproximadamente constante. Si alguno cruza 1 y cambia fuertemente con la estación, el problema es de dependencia funcional/ambiental y no de una única constante.


## V14.5 · Auditoría de pérdidas por mecanismo

La auditoría global `F_opt/F_loss` de V14.4 se retira de la interfaz y se reemplaza por una prueba más específica. Para cada mes de Rea Quille, la app mantiene fija la potencia solar absorbida y calcula cuánto tendría que cambiar **solamente la convección externa** (`k_conv`) o **solamente la radiación al cielo** (`k_rad`) para alcanzar la potencia útil de referencia.

La prueba muestra la participación de cada mecanismo, la estabilidad mensual de los factores requeridos y un contrafactual con el mejor multiplicador global por mínimos cuadrados aplicado a cada mecanismo por separado. Estos multiplicadores son exclusivamente diagnósticos: no se guardan ni se aplican al simulador como parámetros calibrados.

### V14.5.1 — hotfix de despliegue
Los gráficos de la auditoría de pérdidas se cargan desde `rea_loss_component_charts.py`, desacoplados de `visualizations.py`, para evitar fallos por actualizaciones parciales/cacheadas del despliegue.

## V14.6 · Auditoría de flujo externo

La auditoría convección/radiación de V14.5 se retira de la interfaz una vez identificado el bloque convectivo externo como candidato prioritario. La nueva prueba abre explícitamente la correlación **Churchill–Bernstein** usada por el modelo para flujo cruzado sobre el receptor y reporta, para cada mes de Rea Quille:

- temperatura de superficie y temperatura de película;
- propiedades del aire evaluadas a la temperatura de película;
- Reynolds, Prandtl, Nusselt y coeficiente `h` externo;
- `h` requerido para cerrar el balance manteniendo solar, radiación al cielo y soportes fijos;
- velocidad de viento implícita que produciría esa pérdida con la misma correlación y con el campo de temperaturas congelado.

El viento implícito requerido es sólo un diagnóstico local; no se interpreta como medición meteorológica ni se aplica automáticamente al modelo. Esto permite distinguir si el exceso de convección proviene principalmente de la hipótesis `viento = 1 m/s`, de la dependencia de Churchill–Bernstein con Reynolds/temperatura de película o de un mecanismo que el viento por sí solo no puede explicar.

## V14.7 — prueba de viento meteorológico mensual

La validación Rea mensual incluye una prueba diagnóstica que mantiene congeladas las constantes del colector y cambia únicamente la entrada de viento. El baseline de 1 m/s se compara contra una serie mensual independiente a 10 m, con corrección opcional a la altura efectiva del receptor mediante una ley de potencia. El viento ya no se ofrece como parámetro de calibración en Rea mensual: se considera una entrada meteorológica.

## V14.8 — hipótesis de agregación temporal del DNI

La prueba de viento meteorológico de V14.7 se retira de la interfaz tras mostrar que, para Foz, aumenta las pérdidas convectivas y empeora el ajuste sin corregir la forma anual. La nueva prueba mantiene viento, Tin, Tamb, caudal, óptica e incidencia congelados y modifica exclusivamente la distribución temporal de la irradiancia.

Para cada mes se construye un día solar representativo en hora solar. El perfil relativo de DNI sigue una forma de cielo claro `exp(-B/cos(z))` y se reescala para que su media durante las horas solares sea exactamente igual al DNI mensual publicado. La app compara el caso histórico `f(DNI_medio)` contra el promedio de varios estados cuasiestacionarios `promedio[f(DNI(t))]`, calcula la eficiencia integrada por potencia útil media y muestra el gap de agregación mes a mes.

La prueba no calibra parámetros y no pretende reconstruir el TRNSYS hora a hora. Su objetivo es comprobar de forma aislada si la no linealidad del PTC hace que trabajar directamente con promedios mensuales distorsione la curva de eficiencia.

## V14.9 · Apantallamiento aerodinámico y changelog único

La prueba de agregación temporal de V14.8 se retira de la interfaz después de mostrar un efecto despreciable sobre la curva mensual. La nueva prueba diagnóstica introduce un único factor estructural `S_v` mediante `v_eff = S_v · v_amb`.

`S_v` se identifica solamente con enero, abril, julio y octubre de Foz do Iguaçu. Después queda congelado y se evalúa contra los ocho meses hold-out de Foz y contra los doce meses de Alvorada do Norte sin recalibración. El ensayo mantiene el viento ambiente baseline de 1 m/s para aislar el posible apantallamiento de la calha; por tanto no se presenta como reconstrucción meteorológica.

Desde V14.9 todo el historial de versiones está concentrado en `CHANGELOG.md`. Los antiguos archivos `CHANGELOG_V*.txt` fueron eliminados del paquete.

## V14.10 · Auditoría radial Bhambare/Sukhatme

La prueba diagnóstica activa ya no modifica el viento ni el solver. Ejecuta el caso Bhambare nominal y comprueba:

1. si el bloque externo `convección + radiación al cielo` puede reproducir los Qloss publicados cuando se impone la Tglass de la referencia;
2. qué Tglass necesita ese mismo bloque para alcanzar Qloss de Bhambare y Sukhatme;
3. si los pares publicados `(Tabs, Tglass)` cierran el balance radial del vidrio con las ecuaciones actuales;
4. cómo se reparte aproximadamente la potencia en el absorbedor en régimen cuasiestacionario.

La prueba **no calibra parámetros**. Su objetivo es localizar el siguiente bloque físico que debe revisarse.



## V14.12 · Auditoría directa de régimen/correlación interna

La prueba diagnóstica activa ya no intenta inferir propiedades de Paratherm. Mantiene la misma base de propiedades y cambia **únicamente la correlación interna** en tres conjuntos: agua de Foz, agua de Alvorada y Paratherm NF de Bhambare.

Se comparan cuatro ramas:

- `Nu=4.36`: límite laminar plenamente desarrollado con flujo de calor uniforme;
- Hausen con corrección de entrada laminar y asíntota 4.36;
- Sieder–Tate laminar de entrada;
- Dittus–Boelter forzado como control documental, sin asumir validez cuando Reynolds es bajo.

La interfaz reporta Reynolds, Prandtl, Graetz, longitud térmica/hidrodinámica de entrada, Nu, h, RMSE de eficiencia y RMSE de Tout para Foz y Alvorada, además del efecto sobre Bhambare/Sukhatme. Esto permite responder directamente si la discrepancia aparece también con agua y, por tanto, no puede atribuirse solo a las propiedades del aceite.

La auditoría V14.11 fue retirada de la interfaz y del paquete. El historial continúa concentrado en un único `CHANGELOG.md`.
