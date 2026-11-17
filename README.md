# ¿Lloverá mañana en Cartagena?

Proyecto final de Inteligencia Artificial (UTB). Ruta C, aprendizaje automático supervisado.

Con las mediciones de hoy en la estación del Aeropuerto Rafael Núñez (IDEAM), el sistema predice si mañana lloverá (≥ 1 mm) en Cartagena. Se compara contra dos líneas base: **mayoritario** («nunca llueve») y **persistencia** («mañana será igual que hoy»).

**Integrantes:** David Peláez Mejía (T00079273) · Jose Alejandro Pereira (T00079768)

## Cómo ejecutarlo

Requiere Python 3.10 o superior.

```bash
git clone https://github.com/Davidpel31/lluvia-cartagena.git
cd lluvia-cartagena
pip install -r requirements.txt
python src/lineas_base.py
```

El último comando limpia los datos crudos, construye `data/processed/dataset.csv`, evalúa las líneas base en validación (2021–2022) y guarda los resultados en `experiments/`.

## Datos

| | |
| --- | --- |
| Fuente | IDEAM, sistema DHIME (dhime.ideam.gov.co/atencionciudadano) |
| Estación | 14015080, Aeropuerto Rafael Núñez, Cartagena (Bolívar) |
| Período | 01/01/2010 – 31/12/2025, diario (5.844 días) |
| Variables | Día pluviométrico (precipitación, mm), temperatura máxima y mínima diaria (°C), humedad relativa calculada máxima y mínima diaria (%) |
| Condiciones de uso | Uso bajo responsabilidad del usuario; prohibida la comercialización. Proyecto académico sin fines comerciales. Capturas de los términos en `docs/evidencia/`. |
| Nivel de aprobación | Mezcla de datos «Definitivo» y «Preliminar» (dato crudo sujeto a cambios); se declara como limitación. |

Los archivos crudos están en `data/raw/` tal como los entrega DHIME.

**Partición temporal** (no aleatoria, para evitar fuga de datos): entrenamiento 2010–2020, validación 2021–2022, prueba 2023–2025. La prueba solo se evalúa una vez, al final del proyecto.

**Limpieza** (`src/preparar_datos.py`): humedad > 100 % se recorta a 100; temperatura máxima > 39 °C se marca como faltante; se descartan días con tmin > tmax. Los días sin todas las variables se excluyen al entrenar y evaluar, y todos los modelos se comparan sobre los mismos días.

## Estructura

```
data/raw/          CSV originales de DHIME
data/processed/    dataset diario generado por los scripts
src/               preparar_datos.py, lineas_base.py
notebooks/         exploración de los datos
experiments/       resultados (CSV y JSON) con la semilla usada
docs/              bitácora humano-IA y evidencia de la licencia
```

## Métricas y umbral de éxito (declarados en el Hito 1)

- Métrica principal: F1 de la clase «llueve». Secundaria: AUC.
- Éxito: el mejor modelo supera el F1 de la persistencia en al menos 0,10, medido en prueba (2023–2025).

## Uso de IA

Todo uso de asistentes de IA está registrado en `docs/bitacora_ia.md`.
