# `scripts/` — entrypoints CLI

## Archivos

| Archivo | Rol |
|---|---|
| `preprocess.py` | Pipeline completo CONAF + ERA5 → parquet enriquecido + sidecars. Con `--download-only` baja ERA5 acotado a los días con eventos CONAF sin enriquecer. Con `--backfill` rellena (no destructivo, idempotente) las celdas ERA5 que cayeron en mar saltando a la celda de tierra más cercana, preservando `label_l2`/`modis_*`. |

## Ejemplos

```bash
# Pipeline completo: descarga + enriquecimiento (años acotados)
python scripts/preprocess.py --years 2016-2017

# Solo descarga ERA5 acotada a días con eventos CONAF, a un directorio separado
python scripts/preprocess.py --years 2018-2018 --download-only --era5-dir data/raw/era5_conaf_days

# Re-enriquecer asumiendo que los NetCDF ya están en disco
python scripts/preprocess.py --years 2016-2017 --skip-download

# Rellenar celdas ERA5 sobre mar (salto a tierra) en un parquet ya consolidado, sin re-descargar
python scripts/preprocess.py --years 2012-2018 --backfill --max-snap-km 6

# Forzar re-descarga del CSV CONAF (ignora cache .parquet)
python scripts/preprocess.py --years 2016-2017 --refresh-conaf

```

## Outputs típicos

`preprocess.py` produce, dentro de `data/processed/`:

- `conaf_enriched_<start>_<end>.parquet` — output versionado.
- `conaf_enriched_latest.parquet` — copia simbólica del último run.
- `<archivo>.attribution.json` — sidecar de atribución CC-BY (fuentes + parámetros del run).
- `<archivo>.features.json` — perfil de columnas en JSON.
- `features_report.md` + `features_report.json` — reporte global con artefactos, inventario ERA5 y perfil por columna.

`megafire_thresholds.py` produce:

## Notas

- Las credenciales CDS (Copernicus) e itrend (Dataverse) se leen desde `.env` en raíz. Ver `.env.example`.
- El pipeline usa logging estándar — sin dashboards ni callbacks externos. Para output más verboso: `PYTHONLOGLEVEL=DEBUG python scripts/preprocess.py …`.
