# Preset experimental de Foz do Iguaçu — termopares (16/11/2026)

## Compatibilidad

Esta extensión parte del paquete **PTC V14.12** que estaba disponible en la conversación. Se incorpora sin sustituir los presets documentales de Rea Quille, Bhambare/Sukhatme ni la operación genérica. No se ha aplicado ni verificado directamente sobre V14.14.

## Conjunto de datos incorporado

- Fuente: `dia 16_1_3420.xlsx`, entregado por el usuario.
- 495 instantes del **16/11/2026**, entre **10:04:48 y 14:12:20** (hora civil local de Foz).
- Intervalos: 493 diferencias consecutivas de 30 s y un intervalo de 62 s.
- **Únicas columnas empleadas**: `DATE`, `TIME`, `termopar_1` (entrada T1) y `termopar_2` (salida T2).
- CSV depurado: `data/foz_2026_11_16_termopares.csv`. No contiene radiación, caudal, luminosidad ni otras columnas.
- La columna `temopar_1` del XLSX no se usa: está vacía; la columna correcta es `termopar_1`.
- En 12 instantes la medida T2 es menor que T1. Se conservan las lecturas originales sin alterar el signo ni invertir sensores.

## Procedimiento numérico

1. Se ejecuta el modelo nodal PTC existente, con las mismas ecuaciones energéticas.
2. La temperatura de entrada deja de ser constante y se interpola linealmente a partir de T1(t).
3. Se realiza la integración temporal BDF (`rtol=1e-6`, `atol=1e-7 K`, `max_step=20 s`).
4. Las salidas se registran **exactamente** en los instantes medidos, incluso cuando el intervalo no es uniforme.
5. T2 se reserva para la comparación: RMSE, MAE, sesgo y máximo error absoluto; no se usa para ajustar parámetros ni inicializar el estado.
6. La radiación procede exclusivamente del modelo de cielo claro implementado: `DNI = A·exp(−B/cos z)`, con posición solar determinada por fecha, hora civil, latitud y longitud, y con corrección de ecuación del tiempo. No se utiliza ninguna lectura del sensor de irradiancia.
7. Se acepta importar una campaña CSV/XLSX distinta desde la pestaña **Validación experimental**, reutilizando la misma función y permitiendo varias fechas.

## Parámetros iniciales EDITABLES (no medidos en esta campaña)

| Elemento | Valor inicial | Procedencia |
|---|---|---|
| Geometría | L=2,0 m; W=1,10 m; f=0,16 m; D2=0,039 m; D3=0,042 m; N=12; sin vidrio | Geometría de Rea Quille (2025), confirmada para este preset |
| Fluido | Agua, sin vidrio | Preset existente; confirmar montaje real |
| Caudal másico | 0,0192 kg/s | Referencia del prototipo; **NO** medido en este archivo |
| Temperatura ambiente | 25 °C | Supuesto del preset; **NO** medido en este archivo |
| Viento | 1 m/s | Supuesto del preset; **NO** medido en este archivo |
| Latitud, longitud | −25,516°, −54,585° | Coordenadas aproximadas de Foz do Iguaçu; ajustables |
| Fuso horário | UTC−03 | Hora civil local; corregida a hora solar aparente |
| Modelo de DNI | Cielo claro A·exp(−B/cos z) | Correlación ya existente en PTC V14.12 |
| Coeficientes A, B | 713,35 W/m²; 0,131 | Valores heredados del modelo; **NO** calibrados para el ensayo |
| Orientación | Apertura fija horizontal/eje N-S | Hipótesis del modelo; confirmar orientación real |
| Correlación interna | Automática | Modelo V14.12 |

**Limitación crucial:** la posición solar permite estimar el ángulo de incidencia, pero la DNI real depende de atmósfera y nubosidad. Por eso el resultado es una comparación experimental condicionada a la hipótesis de cielo claro, y no una validación independiente de las condiciones meteorológicas. El flujo y montaje también deben confirmarse para interpretar los errores físicamente.

**Condición inicial:** al comienzo, el HTF se inicializa a T1 del primer registro y los sólidos al ambiente supuesto. No se utiliza T2 para identificar el estado inicial. Por ello, la discrepancia al comienzo puede reflejar la condición térmica previa del colector.

## Uso de la aplicación

1. Instalar dependencias con `pip install -r requirements.txt` y ejecutar `streamlit run app.py`.
2. En la barra lateral, seleccionar **Experimental · Foz · termopares 16/11/2026** y pulsar **Aplicar preset completo**.
3. Ajustar el caudal real, la geometría, la orientación solar, el ambiente y los parámetros atmosféricos si están disponibles de fuentes independientes.
4. Entrar en **Validación experimental** y pulsar **Ejecutar validación temporal T2 vs modelo**.
5. Comparar T1, T2, la salida calculada y el DNI sintético. Descargar la tabla de 495 filas en CSV.
6. Para futuras campañas, usar **Importar una nueva campaña experimental (CSV o XLSX)**. Se admiten las columnas `DATE`, `TIME`, `termopar_1` y `termopar_2`. Las otras columnas se ignoran.

## Ejecución sin Streamlit

```python
from presets import build_preset
from experimental_validation import run_experimental_validation

cfg, fluidos = build_preset("experimental_foz_2026_11_16")
resultado = run_experimental_validation(cfg, fluidos)
print(resultado["metrics"])
resultado["table"].to_csv("validacion_experimental.csv", index=False)
```

## Tests

Ejecutar `PYTHONPATH=. pytest -q` desde la carpeta del paquete. Se comprobaron 22 tests (incluyendo el nuevo preset, la serie 495, la exclusión de sensores no autorizados, la interpolación de T1 y la transición de fecha). La pantalla de Streamlit no se ejecutó automáticamente en este entorno por no estar instalada esa dependencia local, aunque figura en `requirements.txt`.

## Despliegue en GitHub / Streamlit Cloud (corrección de FileNotFoundError)

La versión original requiere `data/foz_2026_11_16_termopares.csv` en la raíz del repositorio, junto a `app.py` y `presets.py`. Si GitHub contiene solamente los archivos `.py` pero no la subcarpeta `data/`, el botón **Aplicar preset completo** puede fallar con `FileNotFoundError`.

**Esta revisión resuelve ambos casos:** carga el CSV cuando está disponible; si no está, recupera la misma serie de 495 lecturas de T1/T2 de una copia compacta incorporada en `experimental_validation.py`. La radiación del sensor original no está incluida en esa copia. Las demás campañas siguen importándose con CSV o XLSX desde la interfaz.

Para desplegar el proyecto completo, subir **el contenido** del ZIP descomprimido a la raíz del repositorio, conservar `data/foz_2026_11_16_termopares.csv`, y comprobar que la aplicación arranque con `app.py` desde la raíz. Subir el ZIP como un único archivo a GitHub no lo descomprime.

Esta corrección modifica solo el mecanismo de carga de datos del preset y aclara su geometría. No modifica los balances térmicos ni calibra el modelo con T2.
