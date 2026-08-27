# PTC nodal en Python

Aplicación interactiva para simular un colector solar cilindro-parabólico con discretización axial y red térmica radial. Incluye:

- balance transitorio de HTF, absorbedor y cubierta de vidrio en cada nodo;
- propiedades `rho(T)`, `mu(T)`, `Cp(T)` y `k(T)`;
- convección interna laminar, Dittus-Boelter y Gnielinski-Forristall;
- radiación absorbedor-vidrio y superficie-cielo;
- convección exterior y tres opciones para el espacio anular;
- solver rígido BDF de SciPy, equivalente funcional de `ode15s`;
- edición de geometría, óptica, ambiente, irradiación y propiedades de fluidos;
- visualización nodo por nodo, esquema del PTC y red de resistencias con flujos térmicos;
- validaciones Bhambare/Sukhatme, TCC/TRNSYS y Tabla 8 del prototipo.

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

1. Ajuste geometría, operación, ambiente, óptica y solver en la barra lateral.
2. Seleccione el modelo de irradiación:
   - **Parishwad / cielo claro**;
   - **DNI y ángulo constantes**;
   - **Perfil horario editable**.
3. En **Propiedades e irradiación** puede comparar el modelo solar seleccionado con:
   - histograma horario del DNI medio;
   - curva DNI·cos(theta) sobre la apertura;
   - tarjeta de horas de sol del modelo (DNI > 1 W/m²);
   - DNI máximo, energía DNI diaria y ventana solar LAT.
   En la misma pestaña puede editar la base del fluido mediante correlación original, tabla completa interpolada con PCHIP o propiedades constantes.
4. Pulse **Ejecutar simulación**.
5. En **Nodo por nodo**, seleccione el tiempo y el volumen de control. La aplicación muestra:
   - temperaturas locales;
   - derivadas `dT/dt`;
   - Reynolds, Prandtl, Nusselt y coeficientes convectivos;
   - esquema geométrico del PTC;
   - red térmica y resistencias efectivas;
   - flujos de energía que entran y salen del nodo;
   - tabla axial completa.

El modo **Barrido comparativo de caudales** reproduce el enfoque del MATLAB con los valores predeterminados `0.015, 0.045, 0.090 kg/s`.

## 5. Guardar parámetros y resultados

En **Reporte y exportación**:

- **Guardar proyecto JSON** conserva configuración y propiedades editadas.
- **Cargar proyecto JSON** restaura esos valores en otra sesión.
- **Descargar reporte TXT** genera el reporte técnico en texto plano.
- **Exportar resultados XLSX** guarda las series globales y el estado nodal final.

Los cambios efectuados directamente en los archivos `.py` se guardan normalmente con `Ctrl+S` en VSCode.

## 6. Estructura del proyecto

```text
PTC_Python_Interactivo/
├── app.py                  Interfaz Streamlit
├── ptc_model.py            Solver BDF, balances y correlaciones
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

El modelo dinámico se integra con `solve_ivp(method="BDF")` y mantiene propiedades termofísicas dependientes de la temperatura.


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
