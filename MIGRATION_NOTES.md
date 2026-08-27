# Cambio a visualización HTML + CSS + JavaScript — 27/08/2026

## Por qué
La visualización de la sección transversal del PTC pasó de Plotly a un componente web autocontenido para conseguir una interacción más limpia y dinámica, similar a los microexperimentos de Circuitos Eléctricos.

## Nuevo archivo
- `interactive_visuals.py`: genera el HTML completo que se incrusta en Streamlit.

## Funciones nuevas
- Slider de ángulo de incidencia en tiempo real, sin rerun de Streamlit.
- Slider de posición del rayo sobre la abertura.
- Animación continua del recorrido óptico.
- Modo **Seguidor de rayo**: rayo incidente → reflexión especular → impacto en receptor.
- Modo **Mapa de calor**: trazado de 180 rayos y distribución relativa de impactos sobre la circunferencia externa del absorbedor.
- El mapa usa `Qsolar_abs_node_W` ya calculado por el modelo para expresar un pico óptico estimado por sector.

## Integración
`app.py` importa `streamlit.components.v1` y renderiza el componente con `components.html(...)`.

No se requieren librerías JavaScript externas, npm ni un proceso de build frontend; por ello el despliegue sigue siendo compatible con Streamlit Cloud desde GitHub.
