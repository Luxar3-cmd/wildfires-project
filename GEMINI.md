# Project Overview: XAI-project — Interpretable Prediction of Mega-Fires in Chile

This project aims to predict whether a wildfire in Chile will become a "mega-fire" (≥ 1,000 hectares), conditional on ignition having occurred. It uses **XGBoost** for prediction and **Tree SHAP** for interpretability, explaining the meteorological and geospatial drivers of these events. The project integrates historical fire records from CONAF with meteorological data from ERA5-Land and active-fire detections from MODIS/FIRMS.

## Core Technologies
- **Language:** Python 3.12
- **Data Processing:** Pandas, GeoPandas, xarray, pyarrow
- **Machine Learning:** XGBoost, SHAP, Quantus, scikit-learn
- **API Clients:** cdsapi (ERA5), pyDataverse (CONAF), FIRMS API (MODIS)
- **Visualization:** Matplotlib, Seaborn, Plotly, Folium
- **CLI/Orchestration:** Typer, Makefile

## Repository Structure
- `src/`: Core pipeline logic (loading, enrichment, feature engineering).
- `scripts/`: CLI tools for preprocessing and threshold calculation.
- `eda/`: Jupyter notebooks for exploratory data analysis.
- `modeling/`: Model training and evaluation — `01_xgboost_baseline.ipynb` (L1 baseline) and `02_l1_vs_l2_experiment.py` (L1 vs L2 contrast + Quantus faithfulness).
- `tests/`: Automated test suite using `pytest`.
- `data/`: Data storage (gitignored). Includes `raw`, `interim`, `processed`, and `models`.
- `latex/`: IEEE paper source and PDF report.
- `references/`: Relevant research papers and documentation.

# Building and Running

## Setup
To initialize the environment (requires Python 3.12):
```bash
make setup
```
This creates a `.venv`, installs dependencies, and initializes the `.env` file from `.env.example`.

## Environment Variables
Configure the following in `.env`:
- `ITREND_API_KEY`: CONAF data access.
- `CDSAPI_KEY`: ERA5-Land data access.
- `FIRMS_MAP_KEY`: MODIS/FIRMS data access.

## Data Preprocessing
Run the end-to-end pipeline (download + enrichment):
```bash
python scripts/preprocess.py --years 2016-2017
```
Flags: `--skip-download`, `--download-only`, `--refresh-conaf`, `--skip-modis`, `--out PATH`.

## Analysis & Modeling
- **EDA:** `make notebook` to launch Jupyter Lab in the `eda/` directory.
- **Thresholds:** `python scripts/megafire_thresholds.py` to calculate per-region thresholds.
- **Modeling:** Explore `modeling/01_xgboost_baseline.ipynb` (L1 baseline); run `python modeling/02_l1_vs_l2_experiment.py` for the L1 vs L2 contrast + Quantus faithfulness report.

## Testing & Quality
- **Run tests:** `make test` (uses `pytest`).
- **Linting:** `make lint` (uses `ruff`).

# Development Conventions

## Code Style
- **Python Version:** 3.12 is recommended and used in the Makefile.
- **Linting:** `ruff` is used for style checking.
- **CLI Framework:** `typer` is preferred for all scripts.
- **Error Handling:** Use logging (configured in `scripts/preprocess.py`) and standard Python exceptions.

## Data Handling & Integrity
- **Immutability:** NEVER modify the original `.parquet` files in `data/processed/` or `data/raw/`. All cleaning, filtering, or transformations must be done in-memory within the execution scripts.
- **Formats:** Prefers `.parquet` for processed data and `.json` for sidecar metadata.
- **Timezones:** CONAF timestamps are converted from local Chilean time to **UTC (naive)** before enrichment.
- **Coordinate Systems:** GPS coordinates (lat/lon) are used for spatial matching.

## Testing
- Tests are located in the `tests/` directory and use `pytest`.
- Add tests for any new pipeline modules or logic changes in `src/`.
- Use the smoke test (`python scripts/preprocess.py --years 2002-2002 --skip-download`) to verify pipeline wiring.

## Documentation
- `README.md`: Main project documentation.
- `src/README.md`: Pipeline technical specification.
- `scripts/README.md`: CLI usage guide.
- `GEMINI.md`: (This file) Instructional context for AI agents.
