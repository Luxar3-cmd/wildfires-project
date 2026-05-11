"""Orquestación: une CONAF con ERA5 puntual (mismo timestamp y ubicación).

Optimización: agrupa los incendios por año y abre solo el NetCDF de ese año
(en vez de cargar 19 años a la vez).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

import geopandas as gpd
import pandas as pd
import xarray as xr
from tqdm import tqdm

from src.config import CHILE_BBOX, DATA_PROCESSED, ERA5_RAW_DIR
from src.derived_features import add_all
from src.era5_downloader import era5_month_path, era5_year_path
from src.era5_extractor import EXPECTED_KEYS, extract_point

logger = logging.getLogger(__name__)

ENRICHED_PARQUET = DATA_PROCESSED / "conaf_enriched.parquet"
Reporter = Callable[[str, str, str, dict[str, Any] | None], None]


def _emit(reporter: Reporter | None, message: str, level: str = "info", **data: Any) -> None:
	if reporter:
		reporter("enrichment", message, level, data or None)


def _resolve_timestamp_col(df: pd.DataFrame) -> str:
	for cand in ("fecha_hora_inicio_utc", "fecha_hora_inicio", "fecha_inicio", "inicio", "fecha"):
		if cand in df.columns:
			return cand
	raise KeyError("No se encontró columna de timestamp (fecha_hora_inicio_utc / fecha_hora_inicio / fecha_inicio).")


def _resolve_lat_lon_cols(df: pd.DataFrame) -> tuple[str, str]:
	lat = next((c for c in df.columns if c in {"latitud", "latitude", "lat"}), None)
	lon = next((c for c in df.columns if c in {"longitud", "longitude", "lon", "lng"}), None)
	if not (lat and lon):
		raise KeyError("No se encontraron columnas de latitud/longitud.")
	return lat, lon


def _bbox_mask(df: pd.DataFrame, lat_col: str, lon_col: str, bbox: dict) -> pd.Series:
	return (
		df[lat_col].between(bbox["south"], bbox["north"])
		& df[lon_col].between(bbox["west"], bbox["east"])
	)


def enrich_conaf_with_era5(
	conaf: gpd.GeoDataFrame | pd.DataFrame,
	era5_dir: Path = ERA5_RAW_DIR,
	out_path: Optional[Path] = None,
	save: bool = True,
	reporter: Reporter | None = None,
	bbox: dict | None = None,
) -> pd.DataFrame:
	"""Enriquece cada incendio con las variables ERA5 del mismo (lat, lon, ts)."""
	out_path = out_path or ENRICHED_PARQUET
	bbox = bbox or CHILE_BBOX

	ts_col = _resolve_timestamp_col(conaf)
	lat_col, lon_col = _resolve_lat_lon_cols(conaf)

	df = pd.DataFrame(conaf.drop(columns="geometry", errors="ignore")).copy().reset_index(drop=True)
	df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

	# Filtra registros enriquecibles
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
	_emit(
		reporter,
		f"Registros enriquecibles: {int(enrich_mask.sum())} / {len(df)}",
		enrichible=int(enrich_mask.sum()),
		total=int(len(df)),
		not_enrichible=int((~valid_point_mask).sum()),
		out_of_coverage=int(out_of_coverage_mask.sum()),
	)

	df["_year"] = df[ts_col].dt.year
	df["_month"] = df[ts_col].dt.month

	results: list[dict] = [None] * len(df)  # type: ignore[list-item]

	if out_of_coverage_mask.any():
		for idx in df[out_of_coverage_mask].index:
			results[idx] = {k: None for k in EXPECTED_KEYS} | {
				"era5_dist_km": None,
				"era5_dt_hours": None,
				"era5_match_quality": "out_of_coverage",
			}
		_emit(
			reporter,
			f"Registros fuera de cobertura ERA5 continental: {int(out_of_coverage_mask.sum())}",
			level="warning",
			rows=int(out_of_coverage_mask.sum()),
			bbox=bbox,
		)

	for (year, month), group in df[enrich_mask].groupby(["_year", "_month"]):
		year = int(year)
		month = int(month)
		nc_path = era5_month_path(year, month, era5_dir)
		if not nc_path.exists():
			nc_path = era5_year_path(year, era5_dir)
		if not nc_path.exists():
			logger.warning("ERA5 NetCDF no encontrado para %04d-%02d (%s) — registros marcados missing", year, month, nc_path)
			_emit(
				reporter,
				f"ERA5 NetCDF no encontrado para {year}-{month:02d}",
				level="warning",
				year=year,
				month=month,
				rows=int(len(group)),
				path=str(nc_path),
			)
			for idx in group.index:
				results[idx] = {k: None for k in EXPECTED_KEYS} | {
					"era5_dist_km": None,
					"era5_dt_hours": None,
					"era5_match_quality": "missing",
				}
			continue

		logger.info("Abriendo %s y enriqueciendo %d incendios de %04d-%02d", nc_path.name, len(group), year, month)
		_emit(
			reporter,
			f"Enriqueciendo {len(group)} incendios de {year}-{month:02d}",
			year=year,
			month=month,
			rows=int(len(group)),
			path=str(nc_path),
		)
		with xr.open_dataset(nc_path, chunks={"time": 24}) as ds:
			for idx, row in tqdm(group.iterrows(), total=len(group), desc=f"ERA5 {year}-{month:02d}"):
				results[idx] = extract_point(ds, row[lat_col], row[lon_col], row[ts_col])
		_emit(reporter, f"Mes {year}-{month:02d} enriquecido", year=year, month=month, rows=int(len(group)))

	# Llena los registros no enriquecibles con missing
	missing_template = {k: None for k in EXPECTED_KEYS} | {
		"era5_dist_km": None,
		"era5_dt_hours": None,
		"era5_match_quality": "missing",
	}
	for idx in df.index:
		if results[idx] is None:
			results[idx] = missing_template

	era5_cols = EXPECTED_KEYS + ["era5_dist_km", "era5_dt_hours", "era5_match_quality"]
	era5_df = pd.DataFrame(results, index=df.index).reindex(columns=era5_cols)
	enriched = pd.concat([df.drop(columns=["_year", "_month"]), era5_df], axis=1)
	enriched = add_all(enriched)

	if save:
		out_path.parent.mkdir(parents=True, exist_ok=True)
		enriched.to_parquet(out_path)
		logger.info("Dataset enriquecido guardado en %s (%d filas, %d cols)", out_path, len(enriched), enriched.shape[1])
	_emit(
		reporter,
		"Enriquecimiento finalizado",
		rows=int(len(enriched)),
		columns=int(enriched.shape[1]),
	)

	return enriched


__all__ = ["enrich_conaf_with_era5", "ENRICHED_PARQUET"]
