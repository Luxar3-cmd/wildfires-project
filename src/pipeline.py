# =============================================================================
# XAI-project — Interpretable Prediction of Mega-Fires in Chile (XGBoost + Tree SHAP)
# Course:  INF-473 Explainable AI · UTFSM · Prof. Raquel Pezoa Rivera
# Authors: Eduardo Morales · Octavia Jara · Benjamín Reyes
# File:    src/pipeline.py — Pipeline orchestration: download, enrich, write final parquet
# =============================================================================
"""Pipeline CONAF + ERA5: descarga, enriquece y produce parquet final."""
from __future__ import annotations

import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import xarray as xr

from src.conaf_loader import CONAF_RAW_DIR, load_conaf
from src.config import CHILE_BBOX, DATA_PROCESSED, ERA5_RAW_DIR
from src.derived_features import add_all
from src.enrichment import enrich_conaf_with_era5
from src.era5 import (
	EXPECTED_KEYS,
	MAX_LAND_SNAP_KM,
	build_land_index,
	download_era5_months,
	era5_month_path,
	era5_year_path,
	extract_point,
)
from src.reporting import attribution_payload, generate_feature_report, write_attribution_sidecar

logger = logging.getLogger(__name__)

LATEST_PARQUET = DATA_PROCESSED / "conaf_enriched_latest.parquet"


def versioned_output_path(start_year: int, end_year: int) -> Path:
	return DATA_PROCESSED / f"conaf_enriched_{start_year}_{end_year}.parquet"


def parse_year_range(spec: str) -> tuple[int, int]:
	m = re.fullmatch(r"\s*(\d{4})\s*-\s*(\d{4})\s*", spec)
	if not m:
		raise ValueError("Formato esperado: YYYY-YYYY (ej: 2002-2020)")
	start_year, end_year = int(m.group(1)), int(m.group(2))
	if start_year > end_year:
		raise ValueError("El primer año no puede ser mayor que el segundo")
	return start_year, end_year


def filter_conaf_years(conaf: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
	ts_col = next((c for c in ("fecha_hora_inicio", "fecha_inicio", "inicio", "fecha") if c in conaf.columns), None)
	if ts_col is None:
		raise KeyError("No se encontró columna de timestamp (fecha_hora_inicio / fecha_inicio).")

	df = conaf.copy()
	ts = pd.to_datetime(df[ts_col], errors="coerce")
	mask = ts.dt.year.between(start_year, end_year)
	result = df.loc[mask].copy()
	if result.empty:
		logger.warning(
			"filter_conaf_years: ningún registro en %d-%d (columna '%s'). "
			"¿Formato de fecha incorrecto o años fuera del dataset?",
			start_year, end_year, ts_col,
		)
	return result


def _year_counts(df: pd.DataFrame) -> dict[str, int]:
	if "fecha_hora_inicio" not in df.columns:
		return {}
	years = pd.to_datetime(df["fecha_hora_inicio"], errors="coerce").dt.year
	counts = years.value_counts().sort_index()
	return {str(int(year)): int(count) for year, count in counts.items()}


def _available_conaf_files() -> list[str]:
	return sorted(path.name for path in CONAF_RAW_DIR.glob("*.csv"))


def _era5_timestamp_col(df: pd.DataFrame) -> str | None:
	# Busca la columna UTC primero; si no existe, cae a la local.
	# Esto garantiza que el agrupamiento por (año, mes) use el tiempo correcto
	# para decidir qué NetCDF de ERA5 abrir (los archivos están en UTC).
	return next((c for c in ("fecha_hora_inicio_utc", "fecha_hora_inicio", "fecha_inicio", "inicio", "fecha") if c in df.columns), None)


def _needed_year_months(df: pd.DataFrame) -> list[tuple[int, int, tuple[int, ...]]]:
	"""Retorna la lista de (año, mes, días) únicos necesarios para descargar ERA5.

	Usa la columna UTC para que un incendio a las 23:00 hora local del 31 de diciembre
	no genere una descarga innecesaria de enero del año siguiente.
	"""
	ts_col = _era5_timestamp_col(df)
	if ts_col is None:
		return []
	ts = pd.to_datetime(df[ts_col], errors="coerce").dropna()
	needed = {}
	for value in ts:
		key = (int(value.year), int(value.month))
		needed.setdefault(key, set()).add(int(value.day))
	if not needed:
		logger.warning("_needed_year_months: ningún timestamp válido en '%s' — no se descargará ERA5", ts_col)
	return sorted((year, month, tuple(sorted(days))) for (year, month), days in needed.items())


def _era5_inventory(year_months: list[tuple[int, int, tuple[int, ...]]], era5_dir: Path = ERA5_RAW_DIR) -> dict[str, bool]:
	"""Mapea cada (año, mes) a si existe algún NetCDF ERA5 local (mensual o anual)."""
	inventory = {}
	for year, month, _days in year_months:
		month_path = era5_month_path(year, month, era5_dir)
		year_path = era5_year_path(year, era5_dir)
		inventory[f"{year}-{month:02d}"] = month_path.exists() or year_path.exists()
	return inventory


def _era5_sizes(year_months: list[tuple[int, int, tuple[int, ...]]], era5_dir: Path = ERA5_RAW_DIR) -> dict[str, int | None]:
	"""Mapea cada (año, mes) al tamaño en bytes del NetCDF local correspondiente."""
	sizes = {}
	for year, month, _days in year_months:
		month_path = era5_month_path(year, month, era5_dir)
		year_path = era5_year_path(year, era5_dir)
		path = month_path if month_path.exists() else year_path
		sizes[f"{year}-{month:02d}"] = path.stat().st_size if path.exists() else None
	return sizes


def _download_bbox(df: pd.DataFrame, margin: float = 0.5) -> dict[str, float]:
	"""Calcula el bbox mínimo que contiene todos los eventos CONAF, con margen.

	Se usa el bbox dinámico (en vez del CHILE_BBOX fijo) para minimizar el tamaño
	de los archivos ERA5 descargados. Si el dataset está filtrado a 4 regiones,
	el bbox resultante es mucho más pequeño que Chile completo.
	CHILE_BBOX actúa como límite máximo: el resultado nunca lo excede.
	"""
	mask = (
		df["latitud"].between(CHILE_BBOX["south"], CHILE_BBOX["north"])
		& df["longitud"].between(CHILE_BBOX["west"], CHILE_BBOX["east"])
	)
	points = df.loc[mask, ["latitud", "longitud"]].dropna()
	if points.empty:
		return CHILE_BBOX

	return {
		"north": min(CHILE_BBOX["north"], float(points["latitud"].max()) + margin),
		"south": max(CHILE_BBOX["south"], float(points["latitud"].min()) - margin),
		"west": max(CHILE_BBOX["west"], float(points["longitud"].min()) - margin),
		"east": min(CHILE_BBOX["east"], float(points["longitud"].max()) + margin),
	}


def _prepare_conaf(start_year: int, end_year: int, refresh_conaf: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
	"""Carga CONAF, filtra por rango de años y retorna (df, resumen)."""
	logger.info("Cargando CONAF (refresh=%s); archivos disponibles: %s", refresh_conaf, _available_conaf_files())
	conaf = load_conaf(refresh=refresh_conaf)
	conaf_year_counts = _year_counts(conaf)
	requested_years = [str(year) for year in range(start_year, end_year + 1)]
	missing_conaf_years = [year for year in requested_years if year not in conaf_year_counts]
	conaf_summary: dict[str, Any] = {
		"rows_total": int(len(conaf)),
		"columns": int(conaf.shape[1]),
		"year_counts": conaf_year_counts,
		"missing_requested_years": missing_conaf_years,
	}
	logger.info("CONAF cargado: %d filas", len(conaf))

	conaf = filter_conaf_years(conaf, start_year, end_year)
	conaf_summary["rows_filtered"] = int(len(conaf))
	if conaf.empty:
		logger.warning("No hay registros CONAF locales para %d-%d", start_year, end_year)
	else:
		logger.info("Filtrado %d-%d: %d filas", start_year, end_year, len(conaf))
	return conaf, conaf_summary


def _write_outputs(enriched: pd.DataFrame, versioned_path: Path, params: dict[str, Any]) -> dict[str, Any]:
	"""Copia al latest, escribe sidecars de atribución y genera feature report."""
	LATEST_PARQUET.parent.mkdir(parents=True, exist_ok=True)
	if versioned_path.resolve() != LATEST_PARQUET.resolve():
		shutil.copy2(versioned_path, LATEST_PARQUET)

	attribution_path = write_attribution_sidecar(versioned_path, extra={"params": params})
	latest_attribution_path = write_attribution_sidecar(LATEST_PARQUET, extra={"params": params})
	features_report = generate_feature_report(versioned_path)

	quality_counts = {}
	if "era5_match_quality" in enriched.columns:
		quality_counts = {
			str(key): int(value)
			for key, value in enriched["era5_match_quality"].value_counts(dropna=False).items()
		}

	output_summary: dict[str, Any] = {
		"path": str(versioned_path),
		"versioned_output": str(versioned_path),
		"latest_output": str(LATEST_PARQUET),
		"rows": int(len(enriched)),
		"columns": int(enriched.shape[1]),
		"era5_match_quality": quality_counts,
		"attribution_path": str(attribution_path),
		"latest_attribution_path": str(latest_attribution_path),
		"features_report_path": features_report["paths"]["markdown"],
		"features_report_json_path": features_report["paths"]["json"],
		"features_sidecar_path": features_report["paths"]["sidecar"],
	}
	logger.info("Dataset guardado: %s (%d filas, %d cols)", versioned_path, len(enriched), enriched.shape[1])
	return output_summary


def run_pipeline(
	start_year: int,
	end_year: int,
	skip_download: bool = False,
	refresh_conaf: bool = False,
	skip_modis: bool = False,
	out_path: Path | None = None,
	era5_dir: Path | None = None,
) -> dict[str, Any]:
	"""Ejecuta el pipeline completo y retorna un resumen serializable."""
	_era5_dir = era5_dir or ERA5_RAW_DIR
	versioned_path = out_path or versioned_output_path(start_year, end_year)
	params = {
		"start_year": start_year,
		"end_year": end_year,
		"skip_download": skip_download,
		"refresh_conaf": refresh_conaf,
		"skip_modis": skip_modis,
		"out_path": str(versioned_path),
		"era5_dir": str(_era5_dir),
	}
	summary: dict[str, Any] = {"params": params}

	conaf, conaf_summary = _prepare_conaf(start_year, end_year, refresh_conaf)
	summary["conaf"] = conaf_summary

	# Calcula qué meses de ERA5 se necesitan y el bbox ajustado a los datos reales
	year_months = _needed_year_months(conaf)
	download_bbox = _download_bbox(conaf)

	# Inventario antes de descarga: permite reportar qué archivos ya existían vs. nuevos
	era5_before = _era5_inventory(year_months, _era5_dir)
	summary["era5"] = {
		"before": era5_before,
		"sizes_before": _era5_sizes(year_months, _era5_dir),
		"needed_months": [f"{year}-{month:02d}" for year, month, _days in year_months],
		"needed_days": {f"{year}-{month:02d}": list(days) for year, month, days in year_months},
		"bbox": download_bbox,
	}

	if skip_download:
		logger.info("Skip de descarga ERA5 activo")
	else:
		logger.info("Descargando ERA5-Land para %d mes(es)", len(year_months))
		t0 = time.monotonic()
		download_era5_months(year_months, bbox=download_bbox, out_dir=_era5_dir)
		summary["era5"]["download_seconds"] = round(time.monotonic() - t0, 2)

	# Inventario después de descarga: permite detectar qué meses siguen faltando
	era5_after = _era5_inventory(year_months, _era5_dir)
	summary["era5"]["after"] = era5_after
	summary["era5"]["sizes_after"] = _era5_sizes(year_months, _era5_dir)
	summary["era5"]["missing_months"] = [month for month, exists in era5_after.items() if not exists]
	if summary["era5"]["missing_months"]:
		logger.warning("Faltan NetCDF ERA5: %s", ", ".join(summary["era5"]["missing_months"]))

	logger.info("Enriqueciendo CONAF con ERA5")
	enriched = enrich_conaf_with_era5(conaf, era5_dir=_era5_dir, out_path=versioned_path, save=True, bbox=download_bbox)

	# Label L2 (FLI ≥ 10.000 kW/m vía MODIS-FRP, Tedim 2018). Paso opcional y robusto:
	# si falta FIRMS_MAP_KEY o falla la descarga, se loguea y el pipeline continúa sin L2.
	if not skip_modis:
		try:
			from src.modis import (
				download_firms_for_conaf,
				l2_summary,
				label_l2,
				load_firms_csvs,
				log_l2_summary,
				match_modis_to_conaf,
			)

			firms_paths = download_firms_for_conaf(conaf, bbox=download_bbox)
			modis_df = load_firms_csvs(firms_paths)
			matches = match_modis_to_conaf(enriched, modis_df)
			enriched = label_l2(enriched, matches)
			# Re-escribir el versionado con las columnas L2 antes de copiar a latest / perfilar
			enriched.to_parquet(versioned_path)
			modis_summary = l2_summary(enriched)
			log_l2_summary(modis_summary)
			summary["modis"] = modis_summary
		except (RuntimeError, requests.RequestException, OSError) as exc:
			logger.warning("L2 (MODIS) falló: %s — pipeline continúa sin label_l2", exc)
			summary["modis"] = {"error": str(exc)}
	else:
		logger.info("Skip de MODIS/L2 activo")
		summary["modis"] = {"skipped": True}

	summary["output"] = _write_outputs(enriched, versioned_path, params)
	summary["attribution"] = attribution_payload()
	return summary


def backfill_era5_water_cells(
	parquet_path: Path,
	era5_dir: Path = ERA5_RAW_DIR,
	max_land_snap_km: float = MAX_LAND_SNAP_KM,
	allow_download: bool = True,
	fill_all: bool = False,
) -> dict[str, Any]:
	"""Rellena, in-place y de forma no destructiva, las filas cuyo ERA5 quedó NaN por
	caer en celda de mar, saltando a la celda de tierra más cercana (≤ max_land_snap_km).

	Idempotente: solo procesa filas con ERA5 ausente y dentro de cobertura; las que ya
	tienen datos o están `out_of_coverage` se saltan (no se rehace trabajo hecho). Preserva
	todas las demás filas y columnas (label_l2, modis_*, superficie_*, etc.) — no recomputa
	MODIS. Reusa los NetCDF en disco; descarga un mes solo si falta y `allow_download`.

	Con `fill_all=True` re-extrae ERA5 de TODAS las filas con cobertura (no solo las NaN):
	se usa tras deduplicar la grilla para recomputar flags y valores desde celdas reales.
	"""
	if not parquet_path.exists():
		raise FileNotFoundError(f"No existe el parquet a rellenar: {parquet_path}")

	df = pd.read_parquet(parquet_path)
	n_rows = len(df)

	# Backup no destructivo (no se borra nada)
	backup_dir = DATA_PROCESSED / ("_backup_pre_dedup_reextract" if fill_all else "_backup_pre_snapping")
	backup_dir.mkdir(parents=True, exist_ok=True)
	backup_path = backup_dir / parquet_path.name
	shutil.copy2(parquet_path, backup_path)
	logger.info("Backfill: backup creado en %s", backup_path)

	ts_col = _era5_timestamp_col(df)
	lat_col = next((c for c in ("latitud", "latitude", "lat") if c in df.columns), None)
	lon_col = next((c for c in ("longitud", "longitude", "lon", "lng") if c in df.columns), None)
	if not (ts_col and lat_col and lon_col):
		raise KeyError("El parquet no tiene columnas de timestamp/lat/lon reconocibles")
	ts = pd.to_datetime(df[ts_col], errors="coerce")

	# Fill set: ERA5 ausente (t2m NaN), NO out_of_coverage y con punto/fecha válidos.
	# Las filas ya cargadas (t2m presente) se saltan → idempotencia.
	t2m_nan = pd.to_numeric(df.get("t2m"), errors="coerce").isna()
	quality = df.get("era5_match_quality")
	in_coverage = quality.ne("out_of_coverage") if quality is not None else True
	valid_point = (
		ts.notna()
		& pd.to_numeric(df[lat_col], errors="coerce").notna()
		& pd.to_numeric(df[lon_col], errors="coerce").notna()
	)
	fill_mask = (in_coverage & valid_point) if fill_all else (t2m_nan & in_coverage & valid_point)
	logger.info(
		"Backfill%s: %d filas candidatas de %d",
		" (fill_all)" if fill_all else "", int(fill_mask.sum()), n_rows,
	)

	era5_value_cols = list(EXPECTED_KEYS) + ["era5_dist_km", "era5_dt_hours", "era5_land_snap_km", "era5_match_quality"]
	for col in era5_value_cols:
		if col not in df.columns:
			df[col] = None

	recovered = 0
	snapped = 0
	fill_idx = df.index[fill_mask]
	groups = ts.loc[fill_idx].groupby([ts.loc[fill_idx].dt.year, ts.loc[fill_idx].dt.month])

	for (year, month), idx_ts in groups:
		year, month = int(year), int(month)
		group_idx = idx_ts.index
		nc_path = era5_month_path(year, month, era5_dir)
		if not nc_path.exists():
			nc_path = era5_year_path(year, era5_dir)
		if not nc_path.exists() and allow_download:
			days = tuple(sorted(set(ts.loc[group_idx].dt.day.astype(int))))
			logger.info("Backfill: ERA5 faltante %04d-%02d — descargando %d día(s)", year, month, len(days))
			download_era5_months([(year, month, days)], bbox=_download_bbox(df), out_dir=era5_dir)
			nc_path = era5_month_path(year, month, era5_dir)
		if not nc_path.exists():
			logger.warning("Backfill: sin NetCDF para %04d-%02d — %d filas sin rellenar", year, month, len(group_idx))
			continue

		with xr.open_dataset(nc_path, chunks={"time": 24}) as ds:
			land_index = build_land_index(ds)
			for idx in group_idx:
				res = extract_point(
					ds, df.at[idx, lat_col], df.at[idx, lon_col], ts.at[idx],
					land_index=land_index, max_snap_km=max_land_snap_km,
				)
				for col in era5_value_cols:
					df.at[idx, col] = res.get(col)
				q = res.get("era5_match_quality")
				if q in ("good", "land_snapped"):
					recovered += 1
				if q == "land_snapped":
					snapped += 1

	# Recalcula derivadas (idempotente) para que las filas rellenadas tengan RH/VPD/etc.
	df = add_all(df)

	df.to_parquet(parquet_path)
	if parquet_path.resolve() != LATEST_PARQUET.resolve():
		shutil.copy2(parquet_path, LATEST_PARQUET)

	quality_counts = {
		str(key): int(value)
		for key, value in df["era5_match_quality"].value_counts(dropna=False).items()
	}
	summary = {
		"parquet": str(parquet_path),
		"backup": str(backup_path),
		"rows": n_rows,
		"candidates": int(fill_mask.sum()),
		"recovered": recovered,
		"land_snapped": snapped,
		"era5_match_quality": quality_counts,
	}
	logger.info(
		"Backfill listo: %d recuperadas (%d por salto a tierra) de %d candidatas",
		recovered, snapped, int(fill_mask.sum()),
	)
	return summary
