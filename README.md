# XAI-project

Análisis exploratorio del registro histórico de incendios forestales de **CONAF** (2002-2020) y pipeline de enriquecimiento puntual con datos climáticos de **ERA5-Land** (Copernicus). Insumo para futuros modelos predictivos con XAI (SHAP/LIME).

## Estructura

```
.
├── data/                     # gitignored
│   ├── raw/                  # CONAF + NetCDF ERA5
│   ├── interim/              # CONAF limpio
│   ├── processed/            # dataset enriquecido
│   └── runs/                 # estado/eventos de ejecuciones monitoreadas
├── src/                      # módulos reutilizables
│   ├── config.py
│   ├── conaf_loader.py
│   ├── era5_downloader.py
│   ├── era5_extractor.py
│   ├── derived_features.py
│   ├── enrichment.py
│   ├── pipeline.py
│   └── run_store.py
├── eda/01_eda_conaf.ipynb
└── scripts/
    ├── preprocess.py         # CLI
    └── monitor.py            # dashboard local
```

## Setup

> **Nota Python**: probado en 3.11 / 3.12. Si tienes 3.14, crea un venv con una versión soportada (ej: `python3.12 -m venv .venv`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Nota: requirements.txt instala pytrend desde git@ItrendCL/pytrend (no PyPI)

cp .env.example .env
chmod 600 .env
# Editar .env con tus credenciales de itrend y CDS (ver más abajo)
```

### Configurar credenciales de itrend (CONAF)

La plataforma usa **Dataverse** (`https://datospararesiliencia.cl`) y se accede vía **pyDataverse 0.3.3**.

1. Regístrate en https://datospararesiliencia.cl/loginpage.xhtml
2. Ve a tu perfil: https://datospararesiliencia.cl/dataverseuser.xhtml
3. Sección **API Token** → "Create Token" → copia el UUID.
4. Pégalo en `.env`:
   ```
   ITREND_API_KEY=tu-uuid
   ```

### Configurar Copernicus CDS

1. Crea cuenta en https://cds.climate.copernicus.eu/user/register
2. Acepta los términos de **ERA5-Land** en la página del dataset: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download#manage-licences
3. Copia tu **API key** (UUID) desde https://cds.climate.copernicus.eu/profile
4. Crea `~/.cdsapirc` con este contenido:
   ```
   url: https://cds.climate.copernicus.eu/api
   key: TU-API-KEY-AQUI
   ```
   `chmod 600 ~/.cdsapirc` para proteger el archivo.

Alternativa: definir `CDSAPI_KEY` en `.env` (este repo lo lee y sobrescribe a `~/.cdsapirc`). Si queda vacío, se ignora y se usa `~/.cdsapirc`.

Opcionalmente puedes limitar el comportamiento de retry de la API CDS:

```bash
CDS_TIMEOUT_SECONDS=90
CDS_RETRY_MAX=3
CDS_SLEEP_MAX_SECONDS=15
```

Esto evita que un `500 Internal Server Error` de Copernicus deje el pipeline reintentando durante horas.

## Uso

### EDA exploratorio

```bash
jupyter notebook eda/01_eda_conaf.ipynb
```

### Pipeline de enriquecimiento

```bash
# Smoke test: un solo año
python scripts/preprocess.py --years 2002-2002 --skip-download

# Producción: todos los años. Requiere licencia ERA5-Land aceptada en CDS.
python scripts/preprocess.py --years 2002-2020

# Si ERA5 ya está descargado
python scripts/preprocess.py --years 2002-2020 --skip-download
```

Outputs principales:

```text
data/processed/conaf_enriched_YYYY_YYYY.parquet
data/processed/conaf_enriched_latest.parquet
data/processed/conaf_enriched_YYYY_YYYY.attribution.json
data/processed/conaf_enriched_YYYY_YYYY.features.json
data/processed/features_report.md
data/processed/features_report.json
```

El parquet versionado contiene cada incendio del rango solicitado + columnas climáticas en el mismo timestamp y ubicación. `conaf_enriched_latest.parquet` se actualiza con el último run exitoso. El archivo histórico `conaf_enriched.parquet` puede existir en instalaciones previas, pero ya no es el output principal.

El pipeline interpreta `hora_inicio` de CONAF como hora civil `America/Santiago`. Para el cruce con ERA5-Land conserva `fecha_hora_inicio` como timestamp local y agrega `fecha_hora_inicio_utc` convertido a UTC, que es la columna usada para extraer clima.

Para forzar una ruta de salida:

```bash
python scripts/preprocess.py --years 2002-2002 --skip-download --out data/processed/mi_run.parquet
```

### Dashboard local de monitoreo

El proyecto incluye un panel web local para lanzar y monitorear ejecuciones del pipeline.

```bash
source .venv/bin/activate
python scripts/monitor.py --port 8877
```

Abrir en el navegador:

```text
http://127.0.0.1:8877
```

El dashboard permite:

- Lanzar runs con rango de años, `skip_download` y `refresh_conaf`.
- Ver etapa actual: CONAF, ERA5, enriquecimiento y output.
- Revisar inventario ERA5 por año.
- Ver eventos y warnings sin leer logs crudos.
- Confirmar cuántas filas quedaron con `era5_match_quality = missing`.
- Ver la ruta exacta del parquet versionado, latest y reporte de features.

Cada ejecución queda registrada en:

```text
data/runs/{run_id}/status.json
data/runs/{run_id}/events.jsonl
```

`status.json` contiene el estado consolidado del run. `events.jsonl` contiene eventos cronológicos emitidos por el pipeline.

## Licencias y atribución de datos

Este proyecto genera datasets derivados a partir de fuentes externas. Al compartir outputs, reportes, notebooks o modelos entrenados con estos datos, conserva la atribución de las fuentes.

### ERA5-Land

ERA5-Land está sujeto a **Creative Commons Attribution 4.0 International (CC BY 4.0)**:

```text
https://creativecommons.org/licenses/by/4.0/
```

Resumen práctico para este proyecto:

- Puedes copiar, redistribuir, adaptar y usar el material, incluso comercialmente.
- Debes dar crédito apropiado a la fuente.
- Debes incluir un enlace a la licencia.
- Debes indicar si hiciste cambios.
- No debes sugerir respaldo de Copernicus, C3S o ECMWF.
- No debes aplicar restricciones adicionales que impidan a otros ejercer lo permitido por CC BY 4.0.

Atribución recomendada:

```text
Contains modified Copernicus Climate Change Service information:
ERA5-Land hourly data from 1981 to present.
Muñoz Sabater, J. (2019). Copernicus Climate Change Service (C3S) Climate Data Store (CDS).
DOI: 10.24381/cds.e2161bac.
Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0):
https://creativecommons.org/licenses/by/4.0/
Changes: point extraction by fire-event location/time and derived weather features.
```

El pipeline escribe un archivo de atribución junto al parquet versionado:

```text
data/processed/conaf_enriched_YYYY_YYYY.attribution.json
```

Ese archivo resume fuentes, licencia ERA5-Land, cambios realizados y aviso de no respaldo.

### CONAF / itrend

Fuente:

```text
Registro histórico de incendios forestales
CONAF / itrend - Datos para Resiliencia
DOI: 10.71578/UXAUN5
https://datospararesiliencia.cl
```

Verifica los términos específicos del dataset en la plataforma de origen antes de redistribuir datos crudos o derivados.

## Estado actual de datos locales

Al último smoke test local:

| Dataset | Estado |
|---|---:|
| CSV CONAF crudos | 18 archivos |
| Filas CONAF crudas | 109.985 |
| CONAF limpio | 109.947 filas, 28 columnas |
| Rango CONAF local | 2002-2020 |
| Temporadas CONAF locales | 2002-2003 a 2019-2020 |
| NetCDF ERA5 locales | 0 archivos |
| Dataset enriquecido actual | smoke `2002-2002`: 1.371 filas, 42 columnas |

Como no hay NetCDF ERA5 descargados, los runs con `--skip-download` dejan las filas del rango solicitado con `era5_match_quality = missing`.

Faltan, al menos:

- NetCDF ERA5 por año en `data/raw/era5/`, por ejemplo `era5_land_2002.nc`.

## Vacíos conocidos y riesgos pendientes

- **ERA5 local ausente**: no hay archivos NetCDF en `data/raw/era5/`. El smoke `2002-2002` genera 1.371 filas, pero todas quedan con `era5_match_quality = missing`; todavía no sirve para entrenar o explicar clima.
- **Meses UTC adicionales**: el cruce climático usa `fecha_hora_inicio_utc`. Un rango local puede requerir meses del año siguiente en UTC; el smoke `2002-2002` necesita también `2003-01`.
- **Horarios ambiguos por DST**: 6 registros CONAF quedan con `fecha_hora_inicio_utc = null` por horas ambiguas en cambios de horario de Chile. Requieren revisión manual o una regla explícita si se quieren enriquecer.
- **Cobertura geográfica continental**: 209 registros locales están fuera del bbox continental usado para ERA5, principalmente territorios insulares. Con la configuración actual se marcarán `out_of_coverage`.
- **Validación limitada**: existen tests unitarios para parsing temporal, inventario UTC y extracción sintética de ERA5, más smoke test sin descarga. Falta una prueba end-to-end con NetCDF real.
- **Datos y runs locales**: `data/` está ignorado por Git. Los outputs, caches y eventos del monitor son estado local reproducible, no artefactos versionados.

## Informe de features

Cada run genera:

```text
data/processed/features_report.md
data/processed/features_report.json
data/processed/conaf_enriched_YYYY_YYYY.features.json
```

El informe cubre:

- CSV CONAF por temporada.
- `- Archivo Indice.csv`.
- `data/interim/conaf_clean.parquet`.
- NetCDF ERA5 presentes en `data/raw/era5/` o nota explícita si no hay.
- Parquet enriquecido del run.

Para cada feature tabular incluye nombre, dtype, nulos, porcentaje de nulos, ejemplos, mínimo/máximo cuando aplica, fuente estimada y rol (`conaf_context`, `candidate_target`, `era5_raw`, `era5_derived`, `join_quality`, etc.).

## Columnas climáticas añadidas

| Columna | Unidad | Origen |
|---|---|---|
| `t2m`, `t2m_celsius` | K, °C | ERA5 puntual |
| `d2m`, `d2m_celsius` | K, °C | ERA5 puntual |
| `u10`, `v10` | m/s | ERA5 puntual |
| `tp`, `tp_mm` | m, mm | ERA5 puntual |
| `ssrd` | J/m² | ERA5 puntual |
| `relative_humidity` | % | derivado (Magnus) |
| `wind_speed` | m/s | derivado |
| `wind_direction` | ° | derivado |
| `era5_match_quality` | str | flag `'good'`/`'poor'` del nearest match |

Columnas temporales relevantes:

- `fecha_hora_inicio`: timestamp CONAF local (`America/Santiago`).
- `fecha_hora_inicio_utc`: timestamp UTC usado para buscar ERA5.

Si una API key real quedó expuesta fuera de tu equipo, rótala en la plataforma de origen y actualiza `.env`.

## Fuentes

- CONAF / itrend: https://datospararesiliencia.cl (DOI: `10.71578/UXAUN5`)
- ERA5-Land: Muñoz Sabater, J., (2019). ERA5-Land hourly data from 1981 to present. Copernicus Climate Change Service (C3S) CDS. https://doi.org/10.24381/cds.e2161bac
