# =============================================================================
# XAI-project — Interpretable Prediction of Mega-Fires in Chile (XGBoost + Tree SHAP)
# Course:  INF-473 Explainable AI · UTFSM · Prof. Raquel Pezoa Rivera
# Authors: Eduardo Morales · Octavia Jara · Benjamín Reyes
# File:    src/enrichment.py — Join CONAF events with point-matched ERA5 by (year, month)
# =============================================================================
"""Orquestación: une CONAF con ERA5 puntual (mismo timestamp y ubicación).

Optimización de memoria: agrupa los incendios por (año, mes) y abre solo el
NetCDF de ese período, en vez de cargar todos los archivos a la vez.
Cadena de fallback para encontrar el NetCDF: primero busca el mensual, luego
el anual, y si ninguno existe marca los registros como "missing".
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import xarray as xr
from tqdm import tqdm

from src.config import CHILE_BBOX, DATA_PROCESSED, ERA5_RAW_DIR
from src.derived_features import add_all
from src.era5 import (
	EXPECTED_KEYS,
	INVARIANT_KEYS,
	build_land_index,
	era5_invariants_path,
	era5_month_path,
	era5_year_path,
	extract_invariant_point,
	extract_point,
)

logger = logging.getLogger(__name__)

ENRICHED_PARQUET = DATA_PROCESSED / "conaf_enriched.parquet"


def _resolve_timestamp_col(df: pd.DataFrame) -> str:
	"""Devuelve el nombre de la primera columna de timestamp presente en el DataFrame."""
	for cand in ("fecha_hora_inicio_utc", "fecha_hora_inicio", "fecha_inicio", "inicio", "fecha"):
		if cand in df.columns:
			return cand
	raise KeyError("No se encontró columna de timestamp (fecha_hora_inicio_utc / fecha_hora_inicio / fecha_inicio).")


def _resolve_lat_lon_cols(df: pd.DataFrame) -> tuple[str, str]:
	"""Devuelve los nombres de las columnas de latitud y longitud presentes en el DataFrame."""
	lat = next((c for c in df.columns if c in {"latitud", "latitude", "lat"}), None)
	lon = next((c for c in df.columns if c in {"longitud", "longitude", "lon", "lng"}), None)
	if not (lat and lon):
		raise KeyError("No se encontraron columnas de latitud/longitud.")
	return lat, lon


def _bbox_mask(df: pd.DataFrame, lat_col: str, lon_col: str, bbox: dict) -> pd.Series:
	"""Máscara booleana de los eventos que caen dentro del bounding box dado."""
	return (
		df[lat_col].between(bbox["south"], bbox["north"])
		& df[lon_col].between(bbox["west"], bbox["east"])
	)


def enrich_conaf_with_era5(
	conaf: gpd.GeoDataFrame | pd.DataFrame,
	era5_dir: Path = ERA5_RAW_DIR,
	out_path: Optional[Path] = None,
	save: bool = True,
	bbox: dict | None = None,
) -> pd.DataFrame:
	"""Enriquece cada incendio con las variables ERA5 del mismo (lat, lon, ts).

	Flujo:
	1. Filtra registros con lat/lon/ts válidos y dentro del bbox.
	2. Para cada (año, mes) único, abre el NetCDF correspondiente y extrae
	   el grid point más cercano en tiempo y espacio para cada incendio.
	3. Calcula features derivadas (temp Celsius, VPD, velocidad de viento, etc.).
	4. Si existe el NetCDF de invariantes, añade variables estáticas por coordenada.
	"""
	out_path = out_path or ENRICHED_PARQUET
	bbox = bbox or CHILE_BBOX

	ts_col = _resolve_timestamp_col(conaf)
	lat_col, lon_col = _resolve_lat_lon_cols(conaf)

	df = pd.DataFrame(conaf.drop(columns="geometry", errors="ignore")).copy().reset_index(drop=True)
	df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

	# Separa registros enriquecibles (tienen coords + ts y caen dentro del bbox)
	# de los que están fuera de cobertura o tienen datos faltantes
	valid_point_mask = df[[ts_col, lat_col, lon_col]].notna().all(axis=1)
	coverage_mask = _bbox_mask(df, lat_col, lon_col, bbox)
	enrich_mask = valid_point_mask & coverage_mask
	out_of_coverage_mask = valid_point_mask & ~coverage_mask
	logger.info(
		"Registros enriquecibles: %d / %d (descartados por NaN en lat/lon/ts: %d, fuera de cobertura: %d)",
		enrich_mask.sum(),
		len(df),
		(~valid_point_mask).sum(),
		out_of_coverage_mask.sum(),
	)

	df["_year"] = df[ts_col].dt.year
	df["_month"] = df[ts_col].dt.month

	results: list[dict] = [None] * len(df)  # type: ignore[list-item]

	if out_of_coverage_mask.any():
		for idx in df[out_of_coverage_mask].index:
			results[idx] = {k: None for k in EXPECTED_KEYS} | {
				"era5_dist_km": None,
				"era5_dt_hours": None,
				"era5_land_snap_km": None,
				"era5_match_quality": "out_of_coverage",
			}
		logger.warning(
			"Registros fuera de cobertura ERA5 continental: %d (bbox=%s)",
			int(out_of_coverage_mask.sum()),
			bbox,
		)

	# Itera por (año, mes) para mantener solo 1 NetCDF abierto a la vez en memoria
	for (year, month), group in df[enrich_mask].groupby(["_year", "_month"]):
		year = int(year)
		month = int(month)
		# Fallback: primero busca el NetCDF mensual, si no existe busca el anual
		nc_path = era5_month_path(year, month, era5_dir)
		if not nc_path.exists():
			nc_path = era5_year_path(year, era5_dir)
		if not nc_path.exists():
			logger.warning(
				"ERA5 NetCDF no encontrado para %04d-%02d (%s) — %d registros marcados missing",
				year, month, nc_path, len(group),
			)
			for idx in group.index:
				results[idx] = {k: None for k in EXPECTED_KEYS} | {
					"era5_dist_km": None,
					"era5_dt_hours": None,
					"era5_land_snap_km": None,
					"era5_match_quality": "missing",
				}
			continue

		logger.info("Abriendo %s y enriqueciendo %d incendios de %04d-%02d", nc_path.name, len(group), year, month)
		with xr.open_dataset(nc_path, chunks={"time": 24}) as ds:
			land_index = build_land_index(ds)
			for idx, row in tqdm(group.iterrows(), total=len(group), desc=f"ERA5 {year}-{month:02d}"):
				results[idx] = extract_point(ds, row[lat_col], row[lon_col], row[ts_col], land_index=land_index)

	# Rellena cualquier registro que haya quedado sin resultado
	missing_template = {k: None for k in EXPECTED_KEYS} | {
		"era5_dist_km": None,
		"era5_dt_hours": None,
		"era5_land_snap_km": None,
		"era5_match_quality": "missing",
	}
	for idx in df.index:
		if results[idx] is None:
			results[idx] = missing_template

	era5_cols = EXPECTED_KEYS + ["era5_dist_km", "era5_dt_hours", "era5_land_snap_km", "era5_match_quality"]
	era5_df = pd.DataFrame(results, index=df.index).reindex(columns=era5_cols)
	enriched = pd.concat([df.drop(columns=["_year", "_month"]), era5_df], axis=1)

	# Features derivadas (temp Celsius, VPD, velocidad de viento, etc.)
	# Se calculan antes de añadir invariantes porque no dependen de ellas
	enriched = add_all(enriched)

	# Añade variables invariantes (tipo de suelo, cobertura vegetal, etc.)
	# Se hace join por coordenada, sin dimensión temporal
	inv_path = era5_invariants_path(out_dir=era5_dir)
	if inv_path.exists():
		with xr.open_dataset(inv_path) as ds_inv:
			inv_rows = [
				extract_invariant_point(ds_inv, row[lat_col], row[lon_col])
				for _, row in enriched[[lat_col, lon_col]].iterrows()
			]
		inv_df = pd.DataFrame(inv_rows, index=enriched.index)
		enriched = pd.concat([enriched, inv_df], axis=1)
		logger.info("Invariantes ERA5 añadidas: %s", INVARIANT_KEYS)
	else:
		logger.warning("Archivo de invariantes ERA5 no encontrado (%s) — columnas omitidas", inv_path)
		for k in INVARIANT_KEYS:
			enriched[k] = None

	if save:
		out_path.parent.mkdir(parents=True, exist_ok=True)
		enriched.to_parquet(out_path)
		logger.info("Dataset enriquecido guardado en %s (%d filas, %d cols)", out_path, len(enriched), enriched.shape[1])

	return enriched


__all__ = ["enrich_conaf_with_era5", "ENRICHED_PARQUET"]
