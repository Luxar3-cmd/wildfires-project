# XAI-project — Interpretable Prediction of Mega-Fires in Chile

**From Spark to Catastrophe: Interpretable Prediction of Mega-Fires in Chile via XGBoost and Tree SHAP.**

This repository contains the data pipeline, exploratory analysis, baseline model, and paper for an
INF-473 (Explainable AI) course project. Rather than predicting *whether* a fire will ignite, the project
predicts — **conditional on ignition having occurred** — whether a fire will escalate into a *mega-fire*,
and uses **Tree SHAP** to explain which meteorological and geospatial drivers push an ordinary fire over
that threshold. The dataset fuses CONAF historical fire records with ERA5-Land atmospheric reanalysis and
MODIS/FIRMS active-fire detections.

The full write-up is in [`latex/main.pdf`](latex/main.pdf).

---

## Authors & course

| | |
|---|---|
| **Course** | INF-473 — Explainable AI |
| **Institution** | Universidad Técnica Federico Santa María (UTFSM) |
| **Instructor** | Prof. Raquel Pezoa Rivera |
| **Authors** | Eduardo Morales · Octavia Jara · Benjamín Reyes |

---

## Repository structure

```
XAI-project/
├── src/                      # Core pipeline (reusable modules) — see src/README.md
│   ├── config.py             # Paths, bbox, ERA5 variable lists, .env loading
│   ├── conaf_loader.py       # CONAF registry download + normalization (Dataverse/itrend)
│   ├── era5.py               # ERA5-Land download + nearest-neighbor point extraction
│   ├── enrichment.py         # Join CONAF events with point-matched ERA5
│   ├── derived_features.py   # Derived meteorology (RH, VPD, wind, precip)
│   ├── modis.py              # MODIS/FIRMS FRP + CONAF matching + L2 label
│   ├── pipeline.py           # Orchestration: run_pipeline()
│   └── reporting.py          # CC-BY attribution + feature-profile reports
├── scripts/                  # Command-line entry points — see scripts/README.md
│   ├── preprocess.py         # CLI: end-to-end CONAF+ERA5(+MODIS) preprocessing
│   └── megafire_thresholds.py# Per-region statistical thresholds (P95/99, log-normal, Pareto)
├── eda/                      # Exploratory analysis (Jupyter)
│   ├── 01_conaf_eda.ipynb            # CONAF EDA 2002–2020 (heavy-tail, maps, seasonality)
│   └── 02_frontier_sensitivity_l2.ipynb  # L2 (EWE) label sensitivity analysis
├── modeling/
│   └── 01_xgboost_baseline.ipynb     # L1 baseline: XGBoost + Tree SHAP (preliminary results)
├── tests/                    # Pytest suite (loader, era5, modis, pipeline)
├── data/                     # gitignored (datasets) — see data/spec.md for ERA5 variables
│   ├── raw/ · interim/ · processed/ · models/ · archive/
├── latex/                    # IEEE paper (Overleaf source + main.pdf)
├── references/               # Reference PDFs
├── requirements.txt · Makefile · .env.example
```

Supplementary docs (Spanish): [`src/README.md`](src/README.md) (pipeline spec), [`scripts/README.md`](scripts/README.md)
(CLI), [`data/spec.md`](data/spec.md) (ERA5-Land variables and derived-feature formulas).

---

## Data pipeline architecture

```
  CONAF registry            ERA5-Land (CDS)          MODIS/FIRMS (NASA)
  (Dataverse / itrend)      hourly + invariants      active-fire FRP
        │                         │                        │
        ▼                         │                        │
  conaf_loader.py                 │                        │
  normalize → UTC                 │                        │
  data/interim/conaf_clean.parquet│                        │
        │                         ▼                        │
        │                   era5.py download               │
        │                   (dynamic bbox per events)      │
        └──────────────┬──────────┘                        │
                       ▼                                    │
                 enrichment.py                              │
                 nearest-neighbor (lat, lon, ts)            │
                 + era5_match_quality flag                  │
                       │                                    │
                       ▼                                    ▼
              derived_features.py                    modis.py
              K→°C · RH · VPD · wind · precip         FRP→FLI · match · label L2 (Tedim 2018)
                       └───────────────┬────────────────────┘
                                       ▼
                          data/processed/conaf_enriched_*.parquet
                          + .attribution.json (CC-BY) + features_report.md   (reporting.py)
                                       │
                       ┌───────────────┴───────────────┐
                       ▼                                ▼
                 eda/ notebooks                modeling/01_xgboost_baseline.ipynb
                 (heavy-tail, labels)          XGBoost + Tree SHAP (L1)
```

**Labels.** *L1* — area-based mega-fire (`superficie_quemada_total_ha ≥ 1,000`), the target of the current
baseline. *L2* — Extreme Wildfire Event via FRP→FLI conversion (≥ 10,000 kW/m, Tedim 2018) with a MODIS
detection guard; still in progress.

---

## Installation & credentials

Python 3.12 recommended (Fedora/RHEL: `dnf install python3.12`; macOS: `brew install python@3.12`).

```bash
git clone <repo-url> && cd XAI-project
make setup            # creates .venv, installs requirements.txt, copies .env.example → .env
```

Edit `.env` with your credentials (see `.env.example`):

| Variable | Where to get it |
|---|---|
| `ITREND_API_KEY` | https://datospararesiliencia.cl → profile → API Token |
| `CDSAPI_KEY` | https://cds.climate.copernicus.eu → profile (also accept the ERA5-Land license) |
| `FIRMS_MAP_KEY` | https://firms.modaps.eosdis.nasa.gov/api/map_key/ |

For ERA5 you may instead create `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api
key: YOUR-API-KEY
```

Run `make help` for all targets (`setup`, `notebook`, `test`, `lint`, `readme`, `clean`).

---

## Reproduction (end-to-end)

```bash
# 1. Smoke test (no downloads, single year) — verifies the pipeline wiring
python scripts/preprocess.py --years 2002-2002 --skip-download

# 2. Full preprocessing (downloads CONAF + ERA5-Land + MODIS, then enriches)
python scripts/preprocess.py --years 2016-2017          # subset used by the baseline
#    flags: --skip-download | --download-only | --refresh-conaf | --skip-modis | --out PATH

# 3. Per-region mega-fire thresholds (writes data/processed/megafire_thresholds.md)
python scripts/megafire_thresholds.py

# 4. Exploratory analysis
make notebook                                            # jupyter lab eda/

# 5. Baseline model + explanations
jupyter lab modeling/01_xgboost_baseline.ipynb
```

The enriched dataset is also shipped compressed at the repo root (`conaf_enriched_latest.tar.gz`) so the
notebooks can run without re-downloading the source APIs.

---

## Results summary (preliminary)

L1 baseline (XGBoost, stratified 5-fold CV) on the 2016–2017 subset of four south-central regions
(Maule, Biobío, Araucanía, O'Higgins): 8,650 events with valid ERA5 coverage, 42 mega-fires
(`≥ 1,000` ha, prevalence 0.49%).

| Metric | Value |
|---|---|
| ROC-AUC | 0.947 ± 0.025 |
| PR-AUC (average precision) | 0.269 ± 0.076 |
| PR-AUC baseline (prevalence) | 0.0049 |

Top Tree SHAP drivers: soil temperature (`stl2`), seasonality (`day_of_year`), and total evaporation
(`e`) — a physically coherent, environment-dominated signal. Full analysis, figures, and discussion are
in [`latex/main.pdf`](latex/main.pdf).

---

## AI usage

This project was developed with the assistance of an AI coding assistant (**Claude Code**, Anthropic). We
disclose its use by area for transparency. In every case the authors defined the problem, reviewed and
validated the output, made the modeling and scientific decisions, and are responsible for the final content.

| Area | How AI assisted | Human responsibility |
|---|---|---|
| **Data pipeline** (`src/`, `scripts/`) | Implementation of modules (download, enrichment, reporting), CLI scaffolding, refactors | Architecture, data-source choices, validation against source APIs |
| **EDA** (notebooks) | Drafting analysis code and visualizations (heavy-tail, maps, seasonality) | Interpretation, framing, conclusions |
| **Model + XAI** | XGBoost baseline, cross-validation, Tree SHAP, figure generation | Threshold/feature choices, leakage control, reading of explanations |
| **Paper + documentation** | LaTeX drafting/editing, this README, formatting | Authorship, claims, review, citations |
| **SOTA / literature research** | Searching and summarizing related work (incl. Google Scholar) | Source selection, critical reading, citation decisions |

The use of AI assistance does not transfer authorship or accountability: all results were reviewed by the
authors and are reported honestly, including the preliminary and high-variance nature of the current
baseline.

---

## Troubleshooting

- **ERA5 download fails / 403.** You must accept the ERA5-Land license in the CDS web UI once, and have a
  valid `CDSAPI_KEY` (or `~/.cdsapirc`). New CDS accounts use the `cds.climate.copernicus.eu/api` endpoint.
- **`ITREND_API_KEY` missing.** CONAF download via Dataverse requires the token from
  datospararesiliencia.cl. Use `--skip-download` to run against already-downloaded data.
- **`pyDataverse` errors.** The Dataverse client is pinned to `pyDataverse==0.3.3` (the version the source
  portal documents); do not upgrade it.
- **Timezone confusion.** CONAF timestamps are local Chilean time and are converted to **UTC (naive)**
  before matching ERA5; all joins are in UTC.
- **FIRMS quota / empty MODIS.** The FIRMS Area API is rate-limited; use `--skip-modis` to build the
  dataset without the L2 label.

---

## Data sources & licenses

| Source | Reference | License |
|---|---|---|
| CONAF wildfire records | itrend / Plataforma de Datos para la Resiliencia (DOI `10.71578/UXAUN5`) | per portal terms |
| ERA5-Land reanalysis | Muñoz Sabater, J. (2019), Copernicus C3S CDS — DOI `10.24381/cds.e2161bac` | CC BY 4.0 |
| MODIS / FIRMS active fire | NASA FIRMS Area API (MODIS Thermal Anomalies/Fire) | NASA open data |

Per-dataset provenance is also emitted as machine-readable sidecars
(`data/processed/*.attribution.json`) by `src/reporting.py`.
