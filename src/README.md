# `src/` — pipeline de datos CONAF + ERA5

Convierte CONAF crudo + ERA5-Land en `data/processed/conaf_enriched_latest.parquet` con sus dos sidecars (`.attribution.json`, `.features.json`) y un reporte Markdown global (`data/processed/features_report.md`).

## Archivos

| Archivo | Rol | LOC |
|---|---|---:|
| `config.py` | Paths del proyecto, bbox de estudio (`CHILE_BBOX`), listas `ERA5_VARIABLES` / `ERA5_INVARIANTS`, carga de `.env` (credenciales CDS e itrend) | 94 |
| `conaf_loader.py` | Descarga CONAF desde Dataverse (itrend), normaliza columnas a snake_case ASCII, parsea fechas, convierte timezone `America/Santiago` → UTC. Cachea en `data/interim/conaf_clean.parquet` | 294 |
| `era5.py` | (1) Descarga ERA5-Land desde Copernicus CDS: batches por límite de variables, manejo de ZIP vs NetCDF, normalización y alineación de longitudes (redondeo a `GRID_DECIMALS` antes de fusionar lotes, para no duplicar columnas), descarga de invariantes. (2) Extracción puntual nearest-neighbor `(lat, lon, ts)` con métricas de calidad (`era5_dist_km`, `era5_dt_hours`, `era5_land_snap_km`, `era5_match_quality`). (3) Salto a tierra (`build_land_index` + `extract_point`): ERA5-Land es solo-tierra, así que un incendio costero cuya celda más cercana cae en mar (NaN) recupera la met de la celda de tierra más cercana si está a ≤ `MAX_LAND_SNAP_KM` (6 km). (4) Deduplicación de grilla (`deduplicate_lon`): repara in-place NetCDF cuya longitud quedó duplicada por el merge de lotes (columna real + gemela all-NaN), lossless e idempotente con backup | 758 |
| `enrichment.py` | Une CONAF con ERA5 puntual agrupando por `(año, mes)` para mantener solo 1 NetCDF abierto a la vez en memoria. Marca registros fuera de cobertura, sin NetCDF disponible o con coordenadas inválidas | 198 |
| `derived_features.py` | Features derivadas: Kelvin → Celsius, humedad relativa (Magnus-Tetens), VPD, velocidad y dirección del viento, precipitación en mm | 107 |
| `reporting.py` | (1) Atribución CC-BY de fuentes (`attribution_payload`, `write_attribution_sidecar`) → `.attribution.json` por artefacto. (2) Perfil de columnas con clasificación de roles (`generate_feature_report`) → `.features.json` + `features_report.md` | 505 |
| `pipeline.py` | Orquesta CONAF → ERA5 → enriquecimiento → escritura de parquet + sidecars. Calcula bbox dinámico ajustado a eventos reales, días/meses necesarios e inventario antes/después de descarga. `backfill_era5_water_cells`: relleno no destructivo de celdas de mar saltando a tierra (o re-extracción ERA5 completa con `fill_all=True`, p. ej. tras deduplicar la grilla) | 425 |

> **`modeling_features.py`** no pertenece al pipeline de datos: es la fuente única de la configuración de modeling —la whitelist de 44 features ex-ante (incluye `evavt`), `XGB_PARAMS`, el umbral L1 y las semillas— que importan el notebook baseline y los scripts `modeling/02`–`03`. Vive en `src/` para reusar el patrón de import del repo (`from src.modeling_features import ...`) y evitar que esas constantes se dupliquen y desincronicen entre experimentos.

## Grafo de dependencias

```
config ──► conaf_loader
config ──► era5 ──┐
config ──► reporting ◄─── usa EXPECTED_KEYS / INVARIANT_KEYS de era5
                  ▼
            enrichment ──► derived_features (features post-join)
                  │
                  ▼
              pipeline ──► reporting (sidecars + feature report)
```

## Uso programático

```python
from src.pipeline import run_pipeline

summary = run_pipeline(
    start_year=2016,
    end_year=2017,
    skip_download=True,  # asume ERA5 ya local
)
print(summary["output"]["rows"], "filas enriquecidas")
print(summary["output"]["era5_match_quality"])
```

## Convenciones

- **Timezone**: CONAF llega en hora local `America/Santiago`. Se preserva `fecha_hora_inicio` (local) y se añade `fecha_hora_inicio_utc` (UTC naive) para indexar ERA5.
- **Bbox dinámico**: el pipeline calcula el bbox mínimo que contiene los eventos del rango solicitado, acotado por `CHILE_BBOX`. Reduce dramáticamente el tamaño de las descargas ERA5.
- **Schema estable**: `extract_point` y `extract_invariant_point` siempre retornan dicts con las claves de `EXPECTED_KEYS` / `INVARIANT_KEYS`, aunque la variable no exista en el NetCDF (con `None`).
- **Match quality**: cada fila enriquecida incluye `era5_match_quality ∈ {good, land_snapped, water, poor, missing, out_of_coverage}` para filtrar downstream sin perder trazabilidad. `good` = celda con dato directo; `land_snapped` = recuperado saltando a la celda de tierra más cercana (≤ `MAX_LAND_SNAP_KM`), con la distancia en `era5_land_snap_km`; `water` = celda de mar sin tierra dentro del tope (queda NaN); `out_of_coverage` = fuera del bbox.
- **Backfill no destructivo**: `pipeline.backfill_era5_water_cells` (CLI `preprocess.py --backfill`) rellena in-place las celdas `water`/NaN de un parquet ya consolidado saltando a tierra, **preservando todas las demás columnas** (`label_l2`, `modis_*`) y reusando los NetCDF en disco. Idempotente: salta las filas ya cargadas; re-ataca solo las que siguen sin dato.
- **Deduplicación de grilla ERA5**: `era5.deduplicate_lon` (CLI `preprocess.py --dedup`) corrige NetCDF cuya coordenada de longitud quedó duplicada por el `xr.merge` de lotes descargados —cada longitud aparecía 2× (una con datos + una gemela 100% NaN), inflando `land_snapped` con falsos positivos costeros—. Lossless e idempotente, con backup en `_backup_pre_dedup/`. Tras deduplicar, `preprocess.py --backfill --fill-all` re-extrae ERA5 de todas las filas con cobertura para recomputar `era5_match_quality` desde las celdas reales.
