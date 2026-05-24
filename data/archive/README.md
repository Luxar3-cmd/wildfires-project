# data/archive/

Datasets superados que se conservan para trazabilidad. **No usar como entrada de pipeline activa.**

| Archivo | Tamaño | Origen | Rol histórico | Reemplazo vigente | Archivado el |
|---------|--------|--------|---------------|-------------------|--------------|
| `conaf_clean.parquet` | 5.8 MB | CONAF/itrend, normalizado por `src/conaf_loader.py` | Dataset CONAF completo 2002-2020, 109.947 eventos, 28 columnas. Fuente del análisis original de umbrales de megaincendio. | `data/processed/conaf_enriched_latest.parquet` (12.381 eventos 2016-2017, 66 columnas con ERA5). | 2026-05-24 |
| `conaf_enriched.parquet` (+ `.attribution.json`) | 70 KB | Pipeline ERA5 prototipo | Primera versión del enriched, alcance muy reducido. | `conaf_enriched_latest.parquet`. | 2026-05-24 |
| `conaf_enriched_2002_2002.parquet` (+ `.attribution.json`, `.features.json`) | 198 KB + 200 KB | Pipeline ERA5 acotado a 2002 | Prueba de concepto de enriquecimiento para un solo año. | `conaf_enriched_latest.parquet` (la cobertura ERA5 actual cubre 2016-2017). | 2026-05-24 |

## Por qué archivado y no borrado

- Los scripts `scripts/megafire_thresholds.py`, `src/feature_report.py`, `src/conaf_loader.py` y el notebook `eda/01_eda_conaf.ipynb` aún apuntan a `data/interim/conaf_clean.parquet`. Mantener el archivo aquí permite re-ejecutarlos sin perder la referencia histórica.
- Las cifras de la documentación original (`docs/megafire_labeling_pipeline.html` v1.0 antes de 2026-05-24) se calcularon con estos parquets — preservarlos asegura reproducibilidad de citas.

## Cómo restaurar uno temporalmente

```bash
ln -sf ../archive/conaf_clean.parquet data/interim/conaf_clean.parquet
# ... ejecutar el script legacy ...
rm data/interim/conaf_clean.parquet
```
