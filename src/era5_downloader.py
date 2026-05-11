"""Descarga de ERA5-Land desde Copernicus CDS.

Estrategia: 1 request por año por bbox completo de Chile, NetCDF horario.
Variables: t2m, d2m, u10, v10, tp, ssrd. Dataset: reanalysis-era5-land.
"""
from __future__ import annotations

import logging
import os
from calendar import monthrange
from pathlib import Path
from typing import Iterable, Optional

import cdsapi
import requests

from src.config import (
	CDSAPI_KEY,
	CDSAPI_URL,
	CHILE_BBOX,
	ERA5_RAW_DIR,
	ERA5_VARIABLES,
)

logger = logging.getLogger(__name__)
ERA5_LICENCE_URL = "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download#manage-licences"
CDS_TIMEOUT_SECONDS = int(os.getenv("CDS_TIMEOUT_SECONDS", "90"))
CDS_RETRY_MAX = int(os.getenv("CDS_RETRY_MAX", "3"))
CDS_SLEEP_MAX_SECONDS = int(os.getenv("CDS_SLEEP_MAX_SECONDS", "15"))

ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]
ALL_HOURS = [f"{h:02d}:00" for h in range(24)]


def _client() -> cdsapi.Client:
	"""Construye el cliente CDS leyendo credenciales desde el .env."""
	options = {
		"timeout": CDS_TIMEOUT_SECONDS,
		"retry_max": CDS_RETRY_MAX,
		"sleep_max": CDS_SLEEP_MAX_SECONDS,
	}
	if CDSAPI_KEY:
		return cdsapi.Client(url=CDSAPI_URL, key=CDSAPI_KEY, **options)
	# Fallback a ~/.cdsapirc si no hay env vars
	return cdsapi.Client(url=CDSAPI_URL, **options)


def era5_year_path(year: int, out_dir: Path = ERA5_RAW_DIR) -> Path:
	return out_dir / f"era5_land_{year}.nc"


def era5_month_path(year: int, month: int, out_dir: Path = ERA5_RAW_DIR) -> Path:
	return out_dir / f"era5_land_{year}_{month:02d}.nc"


def _request(
	year: int,
	months: list[str],
	days: list[str],
	bbox: dict,
	variables: list[str],
) -> dict:
	return {
		"variable": variables,
		"year": str(year),
		"month": months,
		"day": days,
		"time": ALL_HOURS,
		"area": [bbox["north"], bbox["west"], bbox["south"], bbox["east"]],
		"data_format": "netcdf",
		"download_format": "unarchived",
	}


def _retrieve(request: dict, target: Path) -> None:
	try:
		_client().retrieve("reanalysis-era5-land", request, str(target))
	except requests.HTTPError as e:
		message = str(e)
		status_code = e.response.status_code if e.response is not None else None
		if "required licences not accepted" in message or "required licence" in message:
			raise RuntimeError(
				"No están aceptadas las licencias requeridas de ERA5-Land en Copernicus CDS.\n"
				f"Acepta la licencia aquí: {ERA5_LICENCE_URL}\n"
				"Luego reintenta el pipeline."
			) from e
		if status_code == 500:
			raise RuntimeError(
				"Copernicus CDS respondió 500 Internal Server Error al solicitar ERA5-Land. "
				"Es un fallo del servicio CDS o de su cola para esta solicitud; reintenta más tarde "
				"o reduce el rango temporal."
			) from e
		raise
	except requests.RequestException as e:
		raise RuntimeError(
			"No se pudo conectar correctamente a Copernicus CDS para descargar ERA5-Land. "
			f"Detalle: {e}"
		) from e


def download_era5_year(
	year: int,
	bbox: Optional[dict] = None,
	variables: Optional[Iterable[str]] = None,
	out_dir: Path = ERA5_RAW_DIR,
	overwrite: bool = False,
) -> Path:
	"""Descarga 1 año de ERA5-Land en NetCDF horario.

	Parameters
	----------
	year : año a descargar.
	bbox : dict con keys north, west, south, east. Default: Chile continental.
	variables : iterable de nombres de variables CDS. Default: las 6 estándar.
	out_dir : directorio de salida.
	overwrite : si True, descarga aunque el archivo ya exista.
	"""
	bbox = bbox or CHILE_BBOX
	variables = list(variables or ERA5_VARIABLES)
	out_dir.mkdir(parents=True, exist_ok=True)
	target = era5_year_path(year, out_dir)

	if target.exists() and not overwrite:
		logger.info("ERA5 %d ya existe, skip: %s", year, target)
		return target

	request = _request(year, ALL_MONTHS, ALL_DAYS, bbox, variables)

	logger.info("Solicitando ERA5-Land año %d (esto puede demorar varios minutos en cola CDS)", year)
	_retrieve(request, target)
	logger.info("Descargado: %s (%.1f MB)", target, target.stat().st_size / 1e6)
	return target


def download_era5_month(
	year: int,
	month: int,
	days: Optional[Iterable[int]] = None,
	bbox: Optional[dict] = None,
	variables: Optional[Iterable[str]] = None,
	out_dir: Path = ERA5_RAW_DIR,
	overwrite: bool = False,
) -> Path:
	"""Descarga 1 mes de ERA5-Land en NetCDF horario."""
	bbox = bbox or CHILE_BBOX
	variables = list(variables or ERA5_VARIABLES)
	out_dir.mkdir(parents=True, exist_ok=True)
	target = era5_month_path(year, month, out_dir)

	if target.exists() and not overwrite:
		logger.info("ERA5 %04d-%02d ya existe, skip: %s", year, month, target)
		return target

	if days is None:
		request_days = [f"{day:02d}" for day in range(1, monthrange(year, month)[1] + 1)]
	else:
		request_days = [f"{int(day):02d}" for day in sorted(set(days))]
	request = _request(year, [f"{month:02d}"], request_days, bbox, variables)

	logger.info("Solicitando ERA5-Land %04d-%02d (%d día(s))", year, month, len(request_days))
	_retrieve(request, target)
	logger.info("Descargado: %s (%.1f MB)", target, target.stat().st_size / 1e6)
	return target


def download_era5_months(
	year_months: Iterable[tuple[int, int] | tuple[int, int, tuple[int, ...]]],
	bbox: Optional[dict] = None,
	variables: Optional[Iterable[str]] = None,
	out_dir: Path = ERA5_RAW_DIR,
	overwrite: bool = False,
) -> list[Path]:
	"""Descarga ERA5-Land para una lista de meses (year, month)."""
	paths = []
	for item in sorted(set(year_months)):
		year, month = int(item[0]), int(item[1])
		days = item[2] if len(item) == 3 else None
		paths.append(download_era5_month(year, month, days, bbox, variables, out_dir, overwrite))
	return paths


def download_era5_range(
	start_year: int,
	end_year: int,
	bbox: Optional[dict] = None,
	variables: Optional[Iterable[str]] = None,
	out_dir: Path = ERA5_RAW_DIR,
	overwrite: bool = False,
) -> list[Path]:
	"""Descarga ERA5-Land para un rango de años (inclusivo)."""
	paths = []
	for year in range(start_year, end_year + 1):
		paths.append(download_era5_year(year, bbox, variables, out_dir, overwrite))
	return paths


__all__ = [
	"download_era5_year",
	"download_era5_range",
	"download_era5_month",
	"download_era5_months",
	"era5_year_path",
	"era5_month_path",
]
