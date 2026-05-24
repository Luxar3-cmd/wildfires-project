# `src/` — pipeline de datos CONAF + ERA5

Convierte CONAF crudo + ERA5-Land en `data/processed/conaf_enriched_latest.parquet` con sus dos sidecars (`.attribution.json`, `.features.json`) y un reporte Markdown global (`data/processed/features_report.md`).

## Archivos

| Archivo | Rol | LOC |
|---|---|---:|
| `config.py` | Paths del proyecto, bbox de estudio (`CHILE_BBOX`), listas `ERA5_VARIABLES` / `ERA5_INVARIANTS`, carga de `.env` (credenciales CDS e itrend) | 94 |
| `conaf_loader.py` | Descarga CONAF desde Dataverse (itrend), normaliza columnas a snake_case ASCII, parsea fechas, convierte timezone `America/Santiago` → UTC. Cachea en `data/interim/conaf_clean.parquet` | 294 |
| `era5.py` | (1) Descarga ERA5-Land desde Copernicus CDS: batches por límite de variables, manejo de ZIP vs NetCDF, normalización de longitudes 0-360 ↔ -180/180, descarga de invariantes. (2) Extracción puntual nearest-neighbor `(lat, lon, ts)` con métricas de calidad (`era5_dist_km`, `era5_dt_hours`, `era5_match_quality`) | 590 |
| `enrichment.py` | Une CONAF con ERA5 puntual agrupando por `(año, mes)` para mantener solo 1 NetCDF abierto a la vez en memoria. Marca registros fuera de cobertura, sin NetCDF disponible o con coordenadas inválidas | 184 |
| `derived_features.py` | Features derivadas: Kelvin → Celsius, humedad relativa (Magnus-Tetens), VPD, velocidad y dirección del viento, precipitación en mm | 107 |
| `reporting.py` | (1) Atribución CC-BY de fuentes (`attribution_payload`, `write_attribution_sidecar`) → `.attribution.json` por artefacto. (2) Perfil de columnas con clasificación de roles (`generate_feature_report`) → `.features.json` + `features_report.md` | 505 |
| `pipeline.py` | Orquesta CONAF → ERA5 → enriquecimiento → escritura de parquet + sidecars. Calcula bbox dinámico ajustado a eventos reales, días/meses necesarios e inventario antes/después de descarga | 256 |

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
- **Match quality**: cada fila enriquecida incluye `era5_match_quality ∈ {good, poor, missing, out_of_coverage}` para filtrar downstream sin perder trazabilidad.
