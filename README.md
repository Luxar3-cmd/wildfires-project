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
├── modeling/                          # restricted to the 4 study regions; config in src/modeling_features.py
│   ├── 01_xgboost_baseline.ipynb        # L1 baseline: XGBoost + Tree SHAP
│   ├── 02_l1_vs_l2_experiment.py        # L1 vs L2 contrast: Tree SHAP + Quantus faithfulness → eda/L1_vs_L2_Experiment_Report.html
│   ├── 03_l2_robust_eval.py             # robust L2 eval (repeated CV + LOPO) + L1→L2 proxy → eda/L2_Robust_Eval_Report.html
│   ├── 04_l2_threshold_sensitivity.py   # L2 label robustness to the FLI threshold and η_r → latex/images/
│   └── 05_operational_triage.py         # operational utility: recall/lift per inspection budget → latex/images/
├── tests/                    # Pytest suite (loader, era5, modis, pipeline)
├── data/                     # gitignored (datasets)
│   ├── raw/ · interim/ · processed/ · models/ · archive/
├── latex/                    # IEEE paper (Overleaf source + main.pdf)
├── references/               # Reference PDFs
├── requirements.txt · Makefile · .env.example
```

Supplementary docs (Spanish): [`src/README.md`](src/README.md) (pipeline spec), [`scripts/README.md`](scripts/README.md) (CLI).

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

**Labels.** *L1* — area-based mega-fire (`superficie_quemada_total_ha ≥ 1,000`): a coarse but direct label.
*L2* — a physically grounded approximation of an Extreme Wildfire Event (EWE), via FRP→FLI conversion
(Wooster et al. 2003, 2004; threshold ≥ 10,000 kW/m) with a MODIS-detection / area ≥ 50 ha guard, computed in
`src/modis.py`. **Limitation:** the full EWE definition (the standard CONAF adheres to) also requires *spread
rate* and *spot distance*, which are only measurable in situ; here only FRP is available. **Research question:**
is the easy-to-compute L1 (area) a valid proxy for L2 (intensity)? The modeling experiments (`modeling/02`–`05`,
restricted to the four study regions) find the two **complementary, not interchangeable**: the area score ranks
intensity events well (L1→L2 AUC ≈ 0.89) but structurally misses small, high-intensity fires, and their Tree
SHAP drivers diverge.

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

Run `make help` for all targets (`setup`, `notebook`, `test`, `lint`, `readme`, `report`, `clean`).

---

## Reproduction (end-to-end)

```bash
# 1. Smoke test (no downloads, single year) — verifies the pipeline wiring
python scripts/preprocess.py --years 2002-2002 --skip-download

# 2. Full preprocessing (downloads CONAF + ERA5-Land + MODIS, then enriches)
python scripts/preprocess.py --years 2016-2017          # subset used by the baseline
#    flags: --skip-download | --download-only | --refresh-conaf | --skip-modis | --out PATH
#    --backfill: rellena celdas ERA5 sobre mar (salto a tierra ≤6 km) en un parquet ya hecho, no destructivo

# 3. Per-region mega-fire thresholds (writes data/processed/megafire_thresholds.md)
python scripts/megafire_thresholds.py

# 4. Exploratory analysis
make notebook                                            # jupyter lab eda/

# 5. Baseline model + explanations
jupyter lab modeling/01_xgboost_baseline.ipynb

# 6. Modeling experiments (2012–2018, restricted to the 4 study regions)
python modeling/02_l1_vs_l2_experiment.py      # L1 vs L2 contrast + Quantus faithfulness
python modeling/03_l2_robust_eval.py           # robust L2 eval (repeated CV + LOPO) + L1→L2 proxy
python modeling/04_l2_threshold_sensitivity.py # L2 label robustness to FLI threshold / η_r
python modeling/05_operational_triage.py       # operational utility (recall/lift per budget)

# 7. Session report (docs/reporte_e3.html)
make report
```

The enriched dataset is also shipped compressed at the repo root (`conaf_enriched_latest.tar.gz`) so the
notebooks can run without re-downloading the source APIs.

---

## Results summary (work in progress)

Modeling uses the **2012–2018** dataset restricted to the four study regions (Maule, Biobío, Araucanía,
O'Higgins): **30,511 modelable events, L1 = 76 positives, L2 = 11**. All metrics are honest out-of-fold
(stratified CV, repeated 20× for L2 to attach confidence intervals).

- **Discrimination** (`modeling/03`): L1 ROC-AUC 0.914 [0.899, 0.930]; L2 ROC-AUC 0.854 [0.703, 0.916] — the
  wide L2 interval is the honest signature of 11 positives.
- **Research question** — is L1 (area) a proxy for L2 (intensity)? The area-trained score ranks the EWE events
  with AUC ≈ 0.89 and, under leave-one-positive-out, all 11 rank above the 68th risk percentile (it generalizes),
  yet only 6/11 EWEs are also area mega-fires and the Tree SHAP drivers diverge. **L1 is a partial proxy, not a
  substitute** — area and intensity are complementary.
- **Robustness** (`modeling/04`): the conclusion holds across the plausible range of the FLI threshold and η_r.
- **Operational utility** (`modeling/05`): as a triage ranker, the top-10% risk captures ~80% of mega-fires and
  82% of EWEs (8× over random); the bottleneck is precision (class prevalence), not recall.
- **Faithfulness** (`modeling/02`): Tree SHAP is exact by construction; Faithfulness Correlation (Bhatt et al.,
  2020) corroborates it, while Faithfulness Estimate (Alvarez-Melis & Jaakkola, 2018) is unreliable on
  non-linear models (Miró-Nicolau et al., 2025).

The central limitation remains the EWE label: only FRP is available (not in-situ spread rate / spot distance),
so the intensity model is capable but high-variance.

Honest performance metrics are reported out-of-fold (stratified CV); full analysis, figures, and discussion
go in [`latex/main.pdf`](latex/main.pdf).

---

## AI usage

This project was developed with the assistance of an AI coding assistant (**Claude Code**, Anthropic). We
disclose its use by area for transparency. In every case the authors defined the problem, reviewed and
validated the output, made the modeling and scientific decisions, and are responsible for the final content.

| Area | How AI assisted | Human responsibility |
|---|---|---|
| **Data pipeline** (`src/`, `scripts/`) | Implementation of modules (download, enrichment, reporting), CLI scaffolding, refactors | Architecture, data-source choices, validation against source APIs |
| **EDA** (notebooks) | Drafting analysis code and visualizations (heavy-tail, maps, seasonality) | Interpretation, framing, conclusions |
| **Model + XAI** | XGBoost baseline, cross-validation, Tree SHAP, Quantus faithfulness, figure generation | Threshold/feature choices, leakage control, reading of explanations |
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
