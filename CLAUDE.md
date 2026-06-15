# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Proyecto de curso INF-473 (XAI, UTFSM). Predice —**condicional a que un incendio ya se inició**— si
escalará a *megaincendio*, y explica los drivers con **Tree SHAP**. Fusiona registros CONAF + reanálisis
ERA5-Land + detecciones MODIS/FIRMS. La doc de usuario está en `README.md`, `src/README.md` (spec del
pipeline) y `scripts/README.md` (CLI); este archivo resume lo que cruza varios módulos.

## Comandos

```bash
make setup            # crea .venv (Python 3.12), instala requirements.txt, copia .env.example → .env
make test             # pytest tests/ -v
make lint             # ruff check src/ --select E,W,F
make notebook         # jupyter lab eda/
make readme           # regenera README.html con pandoc (NO editar el .html a mano)

# Un solo test
.venv/bin/python -m pytest tests/test_era5_extractor.py -v
.venv/bin/python -m pytest tests/test_pipeline.py::nombre_del_test -v

# Smoke test del pipeline (sin descargas, verifica el wiring)
python scripts/preprocess.py --years 2002-2002 --skip-download

# Pipeline completo (descarga CONAF + ERA5-Land + MODIS, luego enriquece)
python scripts/preprocess.py --years 2012-2018
#   flags: --skip-download | --download-only | --skip-modis | --refresh-conaf | --out PATH
#   --backfill  : rellena celdas ERA5 sobre mar saltando a tierra (no destructivo, idempotente)
#   --dedup     : corrige .nc con longitud duplicada por el merge de lotes (ver "Gotchas")

# Experimentos de modeling (escriben reportes HTML en eda/)
python modeling/02_l1_vs_l2_experiment.py    # L1 vs L2 + Quantus faithfulness
python modeling/03_l2_robust_eval.py         # eval robusta de L2 (clase rara) + proxy L1→L2

# Visualizador del pipeline (artefacto HTML autocontenido)
.venv/bin/python viz/export_viz_data.py      # genera viz/viz_data.json (requiere red)
.venv/bin/python viz/build_viz.py            # ensambla viz/pipeline_viz.html
```

Las credenciales (`ITREND_API_KEY`, `CDSAPI_KEY`, `FIRMS_MAP_KEY`) se leen de `.env` en la raíz.

## Arquitectura del pipeline de datos (`src/`)

Flujo orquestado por `pipeline.run_pipeline()` → escribe `data/processed/conaf_enriched_*.parquet`:

```
conaf_loader  ──► enrichment ──► derived_features ──► [modis] ──► parquet + sidecars (reporting)
(itrend/         (nearest-NN     (RH, VPD, viento,    (FRP→FLI,
 Dataverse,       lat/lon/ts       precip mm, K→°C)     label L2)
 tz→UTC)          con ERA5)
      era5 ──────────┘
(CDS, bbox dinámico,
 salto a tierra)
```

- **`config.py`** centraliza paths, `CHILE_BBOX` (macrozona centro-sur), y las listas `ERA5_VARIABLES` /
  `ERA5_INVARIANTS`. Importar paths desde aquí, no hardcodear.
- **`era5.py`** (el módulo más grande) hace 4 cosas: descarga por lotes desde CDS, extracción puntual
  nearest-neighbor con métricas de calidad, *salto a tierra* (ERA5-Land es solo-tierra; un incendio costero
  recupera la celda de tierra más cercana ≤ `MAX_LAND_SNAP_KM`=6 km), y `deduplicate_lon` (repara la grilla,
  ver Gotchas).
- **`enrichment.py`** agrupa por `(año, mes)` para mantener 1 solo NetCDF abierto a la vez.
- **`reporting.py`** emite los sidecars `.attribution.json` (CC-BY) y `.features.json` + `features_report.md`.

`scripts/preprocess.py` (CLI Typer) y `scripts/megafire_thresholds.py` son los entrypoints. El uso
programático es `from src.pipeline import run_pipeline`.

## Las dos etiquetas (la pregunta de investigación)

- **L1** — megaincendio por área (`superficie_quemada_total_ha ≥ 1000`). Directo pero grueso. 78 positivos.
- **L2** — aproximación física de un Extreme Wildfire Event (EWE): FRP→FLI ≥ 10.000 kW/m, computado en
  `src/modis.py`. **Clase muy rara: ~11 positivos** en 2012-2018. Limitación: la definición EWE completa
  exige *spread rate* y *spot distance* (solo in-situ); aquí solo hay FRP.
- La pregunta es si L1 (fácil de computar) es proxy válido de L2 (intensidad). `modeling/02`–`03` lo
  contrastan; hallazgo preliminar: las explicaciones **no siempre son consistentes**. Detalles en
  `docs/l2_robust_eval_findings.md`.

## Modeling: fuente única + caveats de datos

- **`src/modeling_features.py`** es la única fuente de verdad para los experimentos: la whitelist de **44
  features ex-ante** (`FEATURE_COLS`), `XGB_PARAMS`, `MEGAFIRE_HA_THRESHOLD`, semillas y folds. El notebook
  `modeling/01` y los scripts `02`/`03` importan de aquí — **no redefinir estas constantes localmente**
  (antes estaban triplicadas y se desincronizaron). No pertenece al pipeline de datos pese a vivir en `src/`.
- **Las 44 features están pobladas (post-fix 2026-06-14).** Antes, `evavt`
  (`evaporation_from_vegetation_transpiration`) y los 6 invariantes ERA5 (`slt, lsm, cvh, cvl, tvh, tvl`)
  salían **all-NaN** en el parquet 2012-2018 —por un bug en `deduplicate_lon` (ver Gotchas) y por no haberse
  descargado el NetCDF de invariantes—, así que `load()` los descartaba y `02`/`03` entrenaban con ~37 features
  efectivas. Se corrigió el código y se regeneró el dataset (no destructivo, backup `.bak_pre_evavt_inv`): hoy
  son **44 features efectivas**.
- Métricas honestas: CV estratificada (repetida 20× en `03`) con intervalos de confianza y LOPO, dado el
  conteo mínimo de positivos. El intervalo ancho de L2 es la firma esperada de 11 positivos, no un error.
- El dataset canónico de modeling es `data/processed/conaf_enriched_2012_2018.parquet` (gitignored; viene
  comprimido como `.tar.gz` en la raíz). Los scripts lo hardcodean.

## Gotchas que cruzan archivos

- **Timezone.** CONAF llega en hora local `America/Santiago`. Se preserva `fecha_hora_inicio` (local) y se
  añade `fecha_hora_inicio_utc` (UTC *naive*). **Todos los joins con ERA5 son en UTC.**
- **`era5_match_quality`** ∈ `{good, land_snapped, water, poor, missing, out_of_coverage}` viaja en cada
  fila para filtrar downstream sin perder trazabilidad. `good` = dato directo; `land_snapped` = saltó a
  tierra (distancia en `era5_land_snap_km`); `water` = mar sin tierra dentro del tope (NaN).
- **Artefacto de grilla duplicada (resuelto).** El `xr.merge` por lotes duplicaba cada longitud (una real +
  una gemela), lo que disparaba `land_snapped` falsos en la costa. Ahora el merge realinea los lotes a una
  grilla común (`_align_to_ref`) para no duplicar, y `era5.deduplicate_lon` (CLI `--dedup`) **combina** las
  gemelas celda a celda (lossless + idempotente, con backup) en vez de descartar una. Antes descartaba la
  gemela que traía `evavt` (llega en un batch aparte con grilla desfasada) y lo dejaba all-NaN — esa era la
  causa del punto anterior. El baseline honesto actual es **post-fix** (32.162 filas usables). Si ves un
  `land_snapped` masivo en datos nuevos, sospecha de esto.
- **Inmutabilidad.** NUNCA modificar los `.parquet` de `data/processed/` ni `data/raw/` en sitio. Toda
  limpieza/filtrado/transformación va en memoria dentro de los scripts. (`backfill`/`dedup` son la excepción
  pactada: in-place no destructivo con backup.)
- `pyDataverse` está pineado a `0.3.3` y `quantus` a `0.6.0` (monkey-patch de internals en `modeling/02`).
  No actualizar.

## Estilo de código

- **Indentación por directorio**: `src/` y `scripts/` usan **tabs**; `modeling/` y `viz/` usan **4 espacios**.
  Respetar el del archivo que tocas.
- **Docstrings Google** (`Args:`/`Returns:`/`Raises:`) con **prosa en español**; identificadores y términos
  técnicos en inglés. Convención consistente en `src/`, `scripts/`, `modeling/`, `viz/`.
- CLI con **Typer**; logging estándar (sin dashboards). `ruff` para estilo.
- Los headers de banner (`# ===…` con curso/autores/archivo) van en los módulos de `src/`, `modeling/`,
  `viz/` — mantenerlos al crear archivos nuevos en esas carpetas.
