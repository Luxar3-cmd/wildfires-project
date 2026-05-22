"""Pipeline CONAF + ERA5 con eventos opcionales para monitoreo."""
from __future__ import annotations

import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.attribution import attribution_payload, write_attribution_sidecar
from src.conaf_loader import CONAF_RAW_DIR, load_conaf
from src.config import CHILE_BBOX, DATA_PROCESSED
from src.enrichment import enrich_conaf_with_era5
from src.era5_downloader import download_era5_months, era5_month_path, era5_year_path
from src.feature_report import generate_feature_report

logger = logging.getLogger(__name__)

PipelineEvent = Callable[[str, str, str, dict[str, Any] | None], None]
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
	return df.loc[mask].copy()


def _emit(
	reporter: PipelineEvent | None,
	stage: str,
	message: str,
	level: str = "info",
	data: dict[str, Any] | None = None,
	**extra: Any,
) -> None:
	"""Emite un evento al reporter (si hay) y logea simultáneamente.

	`data` y `**extra` son alternativos: usa `data={}` para dicts ya construidos
	o `**extra` para kwargs simples. Si ambos están presentes, `data` tiene prioridad.
	"""
	payload = data or extra or None
	if reporter:
		reporter(stage, message, level, payload)
	getattr(logger, level if level in {"debug", "info", "warning", "error"} else "info")(message)


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
	return sorted((year, month, tuple(sorted(days))) for (year, month), days in needed.items())


def _era5_inventory(year_months: list[tuple[int, int, tuple[int, ...]]]) -> dict[str, bool]:
	"""Mapea cada (año, mes) a si existe algún NetCDF ERA5 local (mensual o anual)."""
	inventory = {}
	for year, month, _days in year_months:
		month_path = era5_month_path(year, month)
		year_path = era5_year_path(year)
		inventory[f"{year}-{month:02d}"] = month_path.exists() or year_path.exists()
	return inventory


def _era5_sizes(year_months: list[tuple[int, int, tuple[int, ...]]]) -> dict[str, int | None]:
	"""Mapea cada (año, mes) al tamaño en bytes del NetCDF local correspondiente."""
	sizes = {}
	for year, month, _days in year_months:
		month_path = era5_month_path(year, month)
		year_path = era5_year_path(year)
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


def _prepare_conaf(
	start_year: int,
	end_year: int,
	refresh_conaf: bool,
	reporter: PipelineEvent | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
	"""Carga CONAF, filtra por rango de años y retorna (df, resumen)."""
	_emit(reporter, "conaf", f"Cargando CONAF (refresh={refresh_conaf})", data={"files": _available_conaf_files()})
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
	_emit(reporter, "conaf", f"CONAF cargado: {len(conaf)} filas", data=conaf_summary)

	conaf = filter_conaf_years(conaf, start_year, end_year)
	conaf_summary["rows_filtered"] = int(len(conaf))
	if conaf.empty:
		_emit(reporter, "conaf", f"No hay registros CONAF locales para {start_year}-{end_year}", level="warning", rows_filtered=0)
	else:
		_emit(reporter, "conaf", f"Filtrado {start_year}-{end_year}: {len(conaf)} filas", rows_filtered=int(len(conaf)))
	return conaf, conaf_summary


def _write_outputs(
	enriched: pd.DataFrame,
	versioned_path: Path,
	params: dict[str, Any],
	reporter: PipelineEvent | None,
) -> dict[str, Any]:
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
	_emit(reporter, "output", f"Dataset guardado: {versioned_path}", data=output_summary)
	return output_summary


def run_pipeline(
	start_year: int,
	end_year: int,
	skip_download: bool = False,
	refresh_conaf: bool = False,
	out_path: Path | None = None,
	reporter: PipelineEvent | None = None,
) -> dict[str, Any]:
	"""Ejecuta el pipeline completo y retorna un resumen serializable.

	El summary tiene la forma:
	  {
	    "params": {...},
	    "conaf": {"rows_total": int, "rows_filtered": int, "year_counts": {...}, ...},
	    "era5": {"before": {...}, "after": {...}, "missing_months": [...], ...},
	    "output": {"path": str, "rows": int, "era5_match_quality": {...}, ...},
	    "attribution": {...},
	  }
	"""
	versioned_path = out_path or versioned_output_path(start_year, end_year)
	params = {
		"start_year": start_year,
		"end_year": end_year,
		"skip_download": skip_download,
		"refresh_conaf": refresh_conaf,
		"out_path": str(versioned_path),
	}
	summary: dict[str, Any] = {"params": params}

	conaf, conaf_summary = _prepare_conaf(start_year, end_year, refresh_conaf, reporter)
	summary["conaf"] = conaf_summary

	# Calcula qué meses de ERA5 se necesitan y el bbox ajustado a los datos reales
	year_months = _needed_year_months(conaf)
	download_bbox = _download_bbox(conaf)

	# Inventario antes de descarga: permite reportar qué archivos ya existían vs. nuevos
	era5_before = _era5_inventory(year_months)
	summary["era5"] = {
		"before": era5_before,
		"sizes_before": _era5_sizes(year_months),
		"needed_months": [f"{year}-{month:02d}" for year, month, _days in year_months],
		"needed_days": {f"{year}-{month:02d}": list(days) for year, month, days in year_months},
		"bbox": download_bbox,
	}

	if skip_download:
		_emit(reporter, "era5", "Skip de descarga ERA5 activo", data={"inventory": era5_before})
	else:
		_emit(reporter, "era5", f"Descargando ERA5-Land para {len(year_months)} mes(es)", data={"inventory": era5_before})
		t0 = time.monotonic()
		download_era5_months(year_months, bbox=download_bbox)
		summary["era5"]["download_seconds"] = round(time.monotonic() - t0, 2)

	# Inventario después de descarga: permite detectar qué meses siguen faltando
	era5_after = _era5_inventory(year_months)
	summary["era5"]["after"] = era5_after
	summary["era5"]["sizes_after"] = _era5_sizes(year_months)
	summary["era5"]["missing_months"] = [month for month, exists in era5_after.items() if not exists]
	downloaded = {
		month: size for month, size in summary["era5"]["sizes_after"].items()
		if size and summary["era5"]["sizes_before"].get(month) != size
	}
	if downloaded:
		_emit(reporter, "era5", "ERA5 descargado", data={"sizes_bytes": downloaded})
	if summary["era5"]["missing_months"]:
		_emit(
			reporter,
			"era5",
			f"Faltan NetCDF ERA5: {', '.join(summary['era5']['missing_months'])}",
			level="warning",
			data={"missing_months": summary["era5"]["missing_months"]},
		)

	_emit(reporter, "enrichment", "Enriqueciendo CONAF con ERA5")
	enriched = enrich_conaf_with_era5(conaf, out_path=versioned_path, save=True, reporter=reporter, bbox=download_bbox)

	summary["output"] = _write_outputs(enriched, versioned_path, params, reporter)
	summary["attribution"] = attribution_payload()
	return summary
