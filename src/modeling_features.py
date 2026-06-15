# =============================================================================
# XAI-project — Interpretable Prediction of Mega-Fires in Chile (XGBoost + Tree SHAP)
# Course:  INF-473 Explainable AI · UTFSM · Prof. Raquel Pezoa Rivera
# Authors: Eduardo Morales · Octavia Jara · Benjamín Reyes
# File:    src/modeling_features.py — Fuente única del feature set ex-ante y la config de modeling
# =============================================================================
"""Fuente única de verdad para los experimentos de modeling.

Centraliza el conjunto de predictores ex-ante (la whitelist de 44 columnas que
documenta la Tabla de features del paper) y los hiperparámetros/constantes
compartidos por el baseline L1 y los experimentos L1-vs-L2 / L2-robusto. Antes
estas definiciones estaban triplicadas (notebook + dos scripts) y se habían
desincronizado (`evavt` faltaba en los scripts); importarlas desde aquí evita
que vuelvan a divergir.

Consumidores: ``modeling/01_xgboost_baseline.ipynb``,
``modeling/02_l1_vs_l2_experiment.py`` y ``modeling/03_l2_robust_eval.py``.

Los nombres son los short-names que xarray expone tras leer el NetCDF de ERA5
(mapeados desde los long-names del API de CDS en ``src/era5.py``).
"""
from __future__ import annotations

# Predictores ex-ante, agrupados por origen (== whitelist de la tabla del paper).
LOCATION = ["latitud", "longitud", "region", "provincia", "comuna"]
IGNITION_TIME = ["month", "hour", "day_of_year"]
ERA5_TEMPORAL = ["t2m", "d2m", "u10", "v10", "tp", "ssrd",
                 "stl1", "stl2", "stl3", "stl4",
                 "swvl1", "swvl2", "swvl3", "swvl4",
                 "pev", "e", "evavt", "lai_hv", "lai_lv"]
ERA5_STATIC = ["slt", "lsm", "cvh", "cvl", "tvh", "tvl"]
DERIVED = ["t2m_celsius", "d2m_celsius",
           "stl1_celsius", "stl2_celsius", "stl3_celsius", "stl4_celsius",
           "relative_humidity", "vpd_hpa", "wind_speed", "wind_direction", "tp_mm"]

# 44 columnas: location (5) + ignition (3) + ERA5 temporal (19) + estático (6) + derivado (11).
FEATURE_COLS = LOCATION + IGNITION_TIME + ERA5_TEMPORAL + ERA5_STATIC + DERIVED

# Umbral L1 (megaincendio por área) y semillas/folds de la validación cruzada.
MEGAFIRE_HA_THRESHOLD = 1000
RANDOM_STATE = 42
N_SPLITS = 5

# Hiperparámetros del clasificador XGBoost compartidos por todos los experimentos.
# `scale_pos_weight` se calcula por experimento/fold (clase positiva muy rara).
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="aucpr",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
