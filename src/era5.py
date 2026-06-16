# =============================================================================
# XAI-project — Interpretable Prediction of Mega-Fires in Chile (XGBoost + Tree SHAP)
# Course:  INF-473 Explainable AI · UTFSM · Prof. Raquel Pezoa Rivera
# Authors: Eduardo Morales · Octavia Jara · Benjamín Reyes
# File:    src/era5.py — Download ERA5-Land and extract point values (nearest-neighbor)
# =============================================================================
"""Descarga y extracción puntual de ERA5-Land.

Dos responsabilidades cohesivas en un solo módulo:

1. Descarga desde Copernicus CDS (reanalysis-era5-land):
   - 1 request por mes (download_era5_month) o por año completo (download_era5_year)
   - batches por límite de variables × timesteps que impone CDS
   - manejo de ZIP cuando CDS lo entrega en vez de NetCDF directo
   - normalización de longitudes 0-360 ↔ -180/180
   - invariantes (un solo timestamp estático): download_era5_invariants

2. Extracción puntual nearest-neighbor (lat, lon, ts):
   - extract_point: variables temporales + métricas de calidad del match
   - extract_invariant_point: variables estáticas (sin dimensión temporal)
   - schema estable garantizado vía EXPECTED_KEYS / INVARIANT_KEYS
"""
from __future__ import annotations

import logging
import os
import shutil
import zipfile
from calendar import monthrange
from pathlib import Path
from typing import Iterable, Optional

import cdsapi
import numpy as np
import pandas as pd
import requests
import xarray as xr
from scipy.spatial import cKDTree

from src.config import (
	CDSAPI_KEY,
	CDSAPI_URL,
	CHILE_BBOX,
	ERA5_INVARIANTS,
	ERA5_RAW_DIR,
	ERA5_VARIABLES,
)

logger = logging.getLogger(__name__)
ERA5_LICENCE_URL = "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download#manage-licences"

# Parámetros de conexión al CDS. Sobreescribibles con variables de entorno.
# CDS_TIMEOUT_SECONDS: tiempo máximo de espera por respuesta HTTP (no incluye tiempo en cola).
# CDS_RETRY_MAX: reintentos automáticos ante errores 5xx o de red.
# CDS_SLEEP_MAX_SECONDS: espera máxima entre reintentos (backoff exponencial).
CDS_TIMEOUT_SECONDS = int(os.getenv("CDS_TIMEOUT_SECONDS", "90"))
CDS_RETRY_MAX = int(os.getenv("CDS_RETRY_MAX", "3"))
CDS_SLEEP_MAX_SECONDS = int(os.getenv("CDS_SLEEP_MAX_SECONDS", "15"))

ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]
ALL_HOURS = [f"{h:02d}:00" for h in range(24)]

# Umbral de distancia y delta temporal para clasificar un match como "good".
# MAX_DIST_KM=25 cubre la resolución nativa de ERA5-Land (~9 km) con margen amplio.
# MAX_TIME_HOURS=2 tolera desfases horarios típicos al redondear timestamps de incendio.
MAX_DIST_KM = 25.0
MAX_TIME_HOURS = 2.0

# Tope para el "salto a tierra" (land snapping): ERA5-Land es un producto solo-tierra,
# así que las celdas sobre mar son NaN. Un incendio costero cuya celda más cercana es de
# mar puede recuperar la met de la celda de tierra más cercana si está a ≤ este umbral.
MAX_LAND_SNAP_KM = 6.0

# Decimales a los que se redondea la grilla lat/lon antes de fusionar lotes (xr.merge).
# El step nativo de ERA5-Land es 0.1° ≫ 1e-5, así que 5 decimales eliminan las
# diferencias de ULP de float64 entre lotes —que de otro modo duplican columnas de
# longitud (una real + una gemela all-NaN)— sin fusionar celdas reales distintas.
GRID_DECIMALS = 5

# VAR_RENAMES mapea tanto los short-names de xarray (e.g. "t2m") como los long-names
# del API de CDS (e.g. "2m_temperature") al mismo identificador de columna.
# Esto es necesario porque xarray puede usar cualquiera de los dos dependiendo de cómo
# fue generado el NetCDF y la versión de las bibliotecas.
VAR_RENAMES = {
	"t2m": "t2m",
	"2m_temperature": "t2m",
	"d2m": "d2m",
	"2m_dewpoint_temperature": "d2m",
	"u10": "u10",
	"10m_u_component_of_wind": "u10",
	"v10": "v10",
	"10m_v_component_of_wind": "v10",
	"tp": "tp",
	"total_precipitation": "tp",
	"ssrd": "ssrd",
	"surface_solar_radiation_downwards": "ssrd",
	"stl1": "stl1",
	"soil_temperature_level_1": "stl1",
	"stl2": "stl2",
	"soil_temperature_level_2": "stl2",
	"stl3": "stl3",
	"soil_temperature_level_3": "stl3",
	"stl4": "stl4",
	"soil_temperature_level_4": "stl4",
	"swvl1": "swvl1",
	"volumetric_soil_water_layer_1": "swvl1",
	"swvl2": "swvl2",
	"volumetric_soil_water_layer_2": "swvl2",
	"swvl3": "swvl3",
	"volumetric_soil_water_layer_3": "swvl3",
	"swvl4": "swvl4",
	"volumetric_soil_water_layer_4": "swvl4",
	"evavt": "evavt",
	"evaporation_from_vegetation_transpiration": "evavt",
	"pev": "pev",
	"potential_evaporation": "pev",
	"e": "e",
	"total_evaporation": "e",
	"lai_hv": "lai_hv",
	"leaf_area_index_high_vegetation": "lai_hv",
	"lai_lv": "lai_lv",
	"leaf_area_index_low_vegetation": "lai_lv",
	# Invariantes — se incluyen aquí para reutilizar el mismo mapeo en extract_invariant_point
	"slt": "slt",
	"soil_type": "slt",
	"lsm": "lsm",
	"land_sea_mask": "lsm",
	"cvh": "cvh",
	"high_vegetation_cover": "cvh",
	"cvl": "cvl",
	"low_vegetation_cover": "cvl",
	"tvh": "tvh",
	"type_of_high_vegetation": "tvh",
	"tvl": "tvl",
	"type_of_low_vegetation": "tvl",
}

# EXPECTED_KEYS son las claves que deben aparecer en el dict de retorno de extract_point,
# aunque la variable no exista en el NetCDF. Esto garantiza un schema estable al
# construir el DataFrame de enriquecimiento (las columnas siempre están presentes, con None si faltan).
EXPECTED_KEYS = [
	"t2m", "d2m", "u10", "v10", "tp", "ssrd",
	"stl1", "stl2", "stl3", "stl4",
	"swvl1", "swvl2", "swvl3", "swvl4",
	"pev", "e", "evavt",
	"lai_hv", "lai_lv",
]

# INVARIANT_KEYS: variables estáticas extraídas del NetCDF de invariantes (sin dimensión temporal).
INVARIANT_KEYS = ["slt", "lsm", "cvh", "cvl", "tvh", "tvl"]


# ============================================================
# Sección 1 — Descarga desde Copernicus CDS
# ============================================================


def _client() -> cdsapi.Client:
	"""Construye el cliente CDS leyendo credenciales desde el .env.

	Usa CDSAPI_KEY/CDSAPI_URL si están definidas; si no, cae al fallback de
	~/.cdsapirc. Aplica los parámetros de conexión (timeout, retry_max, sleep_max).

	Returns:
		Cliente cdsapi.Client configurado.
	"""
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
	"""Devuelve la ruta del NetCDF anual de ERA5-Land para un año dado.

	Args:
		year: Año de los datos.
		out_dir: Directorio de salida.

	Returns:
		Ruta al archivo NetCDF del año.
	"""
	return out_dir / f"era5_land_{year}.nc"


def era5_month_path(year: int, month: int, out_dir: Path = ERA5_RAW_DIR) -> Path:
	"""Devuelve la ruta del NetCDF mensual de ERA5-Land para un (año, mes) dado.

	Args:
		year: Año de los datos.
		month: Mes de los datos.
		out_dir: Directorio de salida.

	Returns:
		Ruta al archivo NetCDF del mes.
	"""
	return out_dir / f"era5_land_{year}_{month:02d}.nc"


def era5_invariants_path(out_dir: Path = ERA5_RAW_DIR) -> Path:
	"""Devuelve la ruta del NetCDF de invariantes de ERA5-Land.

	Args:
		out_dir: Directorio de salida.

	Returns:
		Ruta al archivo NetCDF de invariantes.
	"""
	return out_dir / "era5_land_invariants.nc"


def _request(
	year: int,
	months: list[str],
	days: list[str],
	bbox: dict,
	variables: list[str],
) -> dict:
	"""Arma el dict de request para reanalysis-era5-land en formato NetCDF horario.

	Args:
		year: Año a solicitar.
		months: Meses a solicitar (lista de strings "MM").
		days: Días a solicitar (lista de strings "DD").
		bbox: Bounding box con claves north/west/south/east.
		variables: Variables a solicitar (long-names del API de CDS).

	Returns:
		Dict de request listo para client.retrieve().
	"""
	# Nota: el API de CDS espera el bbox como [north, west, south, east],
	# que es el orden inverso al convencional (min_lat, min_lon, max_lat, max_lon).
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
	"""Ejecuta el retrieve del CDS y traduce errores HTTP/red a mensajes accionables.

	Args:
		request: Dict de request para reanalysis-era5-land.
		target: Ruta donde guardar el archivo descargado.

	Raises:
		RuntimeError: Si faltan las licencias requeridas de ERA5-Land, si el CDS
			responde 500, o si falla la conexión al servicio.
		requests.HTTPError: Para otros errores HTTP no contemplados explícitamente.
	"""
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


def _normalize_lon(ds):
	"""Convierte longitudes 0-360 a -180/180, redondea lat/lon a GRID_DECIMALS y ordena.

	El redondeo + reasignación deja las grillas de todos los lotes bit-idénticas, de modo
	que xr.merge(join='outer') ya no crea columnas de longitud duplicadas por diferencias
	de ULP de float64. Único punto por el que pasan todos los datasets antes de ambos merge.

	Args:
		ds: Dataset de xarray con coordenada longitude (y opcionalmente latitude).

	Returns:
		El Dataset con longitudes normalizadas, lat/lon redondeadas y ordenado por longitude.
	"""
	lon = ds["longitude"].values
	if (lon > 180).any():
		lon = np.where(lon > 180, lon - 360, lon)
	ds = ds.assign_coords(longitude=np.round(lon, GRID_DECIMALS))
	if "latitude" in ds.coords:
		ds = ds.assign_coords(latitude=np.round(ds["latitude"].values, GRID_DECIMALS))
	return ds.sortby("longitude")


def _align_to_ref(ds, ref):
	"""Reindexa lat/lon de un dataset a la grilla de referencia (nearest, tolerancia sub-celda).

	Alinea lotes del CDS que difieren por ULP de float en sus coordenadas, para que
	xr.merge no cree columnas de longitud gemelas (que dejarían a una variable presente
	solo en un lote —p. ej. evavt— en una columna que luego se descartaría).

	Args:
		ds: Dataset a realinear.
		ref: Dataset de referencia cuya grilla lat/lon se adopta.

	Returns:
		El dataset reindexado a la grilla de `ref` (o sin cambios si no comparte coords).
	"""
	idx = {}
	if "latitude" in ds.coords and "latitude" in ref.coords:
		idx["latitude"] = ref["latitude"]
	if "longitude" in ds.coords and "longitude" in ref.coords:
		idx["longitude"] = ref["longitude"]
	if not idx:
		return ds
	return ds.reindex(method="nearest", tolerance=0.02, **idx)


def _unzip_if_needed(path: Path) -> None:
	"""Extrae los .nc de un ZIP del CDS y reemplaza el archivo, si vino comprimido.

	CDS puede entregar un ZIP con un único NetCDF (caso habitual) o con un NetCDF
	por variable (caso raro en invariantes). Si hay múltiples NCs, los normaliza
	(longitud y dimensión time degenérada) y los fusiona con xarray. Si el archivo
	no es un ZIP, no hace nada.

	Args:
		path: Ruta al archivo descargado que podría ser un ZIP.

	Raises:
		RuntimeError: Si el ZIP no contiene ningún NetCDF en su interior.
	"""
	if not zipfile.is_zipfile(path):
		return
	with zipfile.ZipFile(path) as zf:
		nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
		if not nc_names:
			raise RuntimeError(f"ZIP descargado sin NetCDF en su interior: {path}")
		tmp_paths: list[Path] = []
		for nc_name in nc_names:
			tmp = path.with_name(f"_unzip_{Path(nc_name).stem}.nc")
			with zf.open(nc_name) as src, tmp.open("wb") as dst:
				dst.write(src.read())
			tmp_paths.append(tmp)
	path.unlink()
	if len(tmp_paths) == 1:
		tmp_paths[0].rename(path)
	else:
		merged = path.with_name(path.stem + "_tmp_merged.nc")
		raw_datasets = [xr.open_dataset(p, engine="netcdf4") for p in tmp_paths]
		# Normaliza lon y colapsa dimensión time degenérada (size=1) en cada NC individual
		# antes de hacer el merge. Así se evita que NCs con timestamps distintos creen
		# una dimensión time ficticia al fusionarse con join='outer'.
		datasets = []
		for ds in raw_datasets:
			ds = _normalize_lon(ds)
			if "time" in ds.dims and ds.sizes["time"] == 1:
				ds = ds.squeeze(dim="time", drop=True)
			datasets.append(ds)
		try:
			xr.merge(datasets).to_netcdf(merged, engine="netcdf4")
		finally:
			for ds in datasets:
				ds.close()
			for p in tmp_paths:
				p.unlink(missing_ok=True)
		merged.rename(path)
	logger.info("ZIP extraído → %s", path.name)


def deduplicate_lon(
	path: Path,
	backup_dir: Path | None = None,
	ref_var: str = "t2m",
	decimals: int = GRID_DECIMALS,
) -> dict:
	"""Deduplica la grilla de longitud de un NetCDF ERA5: in-place, lossless e idempotente.

	El xr.merge de los lotes descargados del CDS (longitudes que difieren por ULP de
	float64) duplica cada columna de longitud: una con datos + su gemela donde esas mismas
	variables son NaN. Aquí, por cada longitud única (redondeada a `decimals`) se combinan
	las gemelas celda a celda tomando el primer valor no-NaN, de modo que una variable
	presente solo en una gemela (p. ej. evavt) no se pierde. Preserva el encoding del
	original y hace backup no destructivo antes de reescribir. Si el archivo ya está limpio,
	no hace nada.

	Args:
		path: Ruta al NetCDF a deduplicar (se reescribe in-place).
		backup_dir: Directorio donde copiar el original antes de reescribir; None para
			no hacer backup.
		ref_var: Obsoleto (se mantiene por compatibilidad de firma); ya no se usa, pues las
			gemelas se combinan en vez de descartarse según una variable de referencia.
		decimals: Decimales a los que se redondea la longitud para agrupar gemelas.

	Returns:
		Dict con métricas de la operación: path, skipped (True si ya estaba limpio),
		n_lon y, si se deduplicó, n_unique y removed.
	"""
	with xr.open_dataset(path) as ds:
		lon = ds["longitude"].values
		lon_r = np.round(lon, decimals)
		uniq = np.unique(lon_r)
		if uniq.size == lon.size:
			return {"path": str(path), "skipped": True, "n_lon": int(lon.size)}

		# Combina las columnas de longitud gemelas en vez de descartar una de ellas: para
		# cada longitud única toma, celda a celda, el primer valor no-NaN entre las gemelas.
		# Así no se pierde una variable que vive solo en una de ellas (p. ej. evavt, que
		# llega en un lote propio con la grilla desfasada).
		ds_r = ds.assign_coords(longitude=lon_r)
		parts = []
		for u in uniq:
			twins = np.where(lon_r == u)[0]
			sub = ds_r.isel(longitude=twins)
			combined = sub.isel(longitude=0)
			for j in range(1, int(twins.size)):
				combined = combined.combine_first(sub.isel(longitude=j))
			parts.append(combined)
		clean = xr.concat(parts, dim="longitude").load()

	clean = clean.assign_coords(longitude=uniq)
	# Compresión zlib uniforme: evita inflar el disco sin arriesgar el packing original
	# (copiar dtype/scale_factor del NetCDF leído puede truncar datos o pasar claves inválidas).
	enc = {v: {"zlib": True, "complevel": 4} for v in clean.data_vars}

	if backup_dir is not None:
		backup_dir.mkdir(parents=True, exist_ok=True)
		shutil.copy2(path, backup_dir / path.name)
	tmp = path.with_suffix(".dedup.tmp.nc")
	clean.to_netcdf(tmp, engine="netcdf4", encoding=enc)
	clean.close()
	tmp.replace(path)
	return {
		"path": str(path), "skipped": False, "n_lon": int(lon.size),
		"n_unique": int(uniq.size), "removed": int(lon.size - uniq.size),
	}


def _var_batches(variables: list[str], batch_size: int) -> list[list[str]]:
	"""Parte la lista de variables en lotes de tamaño máximo batch_size.

	Args:
		variables: Variables a dividir.
		batch_size: Tamaño máximo de cada lote.

	Returns:
		Lista de lotes (sublistas) de variables.
	"""
	return [variables[i : i + batch_size] for i in range(0, len(variables), batch_size)]


def download_era5_year(
	year: int,
	bbox: Optional[dict] = None,
	variables: Optional[Iterable[str]] = None,
	out_dir: Path = ERA5_RAW_DIR,
	overwrite: bool = False,
) -> Path:
	"""Descarga 1 año completo de ERA5-Land en NetCDF horario.

	Útil para años históricos completos. Para años parciales o corridas
	incrementales, preferir download_era5_month / download_era5_months.

	Args:
		year: Año a descargar.
		bbox: Bounding box (claves north/west/south/east); None usa CHILE_BBOX.
		variables: Variables a descargar; None usa ERA5_VARIABLES.
		out_dir: Directorio de salida.
		overwrite: Si False y el archivo ya existe, omite la descarga.

	Returns:
		Ruta al NetCDF anual descargado (o existente).
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
	max_vars_per_request: int = 6,
) -> Path:
	"""Descarga 1 mes de ERA5-Land en NetCDF horario.

	Si se especifica `days`, solo descarga esos días del mes.
	Si el número de variables supera `max_vars_per_request`, divide en lotes y
	fusiona los NetCDF resultantes con xarray. La nueva API de CDS rechaza
	requests con demasiadas variables × timesteps en un solo call.

	Args:
		year: Año a descargar.
		month: Mes a descargar.
		days: Días específicos del mes; None descarga el mes completo.
		bbox: Bounding box (claves north/west/south/east); None usa CHILE_BBOX.
		variables: Variables a descargar; None usa ERA5_VARIABLES.
		out_dir: Directorio de salida.
		overwrite: Si False y el archivo ya existe, omite la descarga.
		max_vars_per_request: Máximo de variables por request antes de dividir en lotes.

	Returns:
		Ruta al NetCDF mensual descargado (o existente).
	"""
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

	batches = _var_batches(variables, max_vars_per_request)
	if len(batches) == 1:
		request = _request(year, [f"{month:02d}"], request_days, bbox, variables)
		logger.info("Solicitando ERA5-Land %04d-%02d (%d día(s))", year, month, len(request_days))
		_retrieve(request, target)
		_unzip_if_needed(target)
	else:
		# Descarga un lote de variables a la vez y fusiona al final.
		# Necesario porque la API de CDS limita el "costo" por request.
		temp_paths: list[Path] = []
		for i, batch in enumerate(batches):
			tmp = out_dir / f"_tmp_{year}_{month:02d}_b{i}.nc"
			request = _request(year, [f"{month:02d}"], request_days, bbox, batch)
			logger.info(
				"Solicitando ERA5-Land %04d-%02d lote %d/%d (%d var(s), %d día(s))",
				year, month, i + 1, len(batches), len(batch), len(request_days),
			)
			_retrieve(request, tmp)
			_unzip_if_needed(tmp)
			temp_paths.append(tmp)
		logger.info("Fusionando %d lotes → %s", len(temp_paths), target)
		datasets = [_normalize_lon(xr.open_dataset(p, engine="netcdf4")) for p in temp_paths]
		try:
			# Reindexa cada lote a la grilla del primero antes del merge: los lotes del CDS
			# pueden venir con longitudes desfasadas por ULP de float y, sin alinear,
			# xr.merge(join='outer') crea columnas de longitud gemelas. Una variable que
			# llega sola en su lote (p. ej. evavt) quedaría en la gemela que luego se pierde.
			ref = datasets[0]
			aligned = [datasets[0]] + [_align_to_ref(d, ref) for d in datasets[1:]]
			xr.merge(aligned).to_netcdf(target, engine="netcdf4")
		finally:
			for ds in datasets:
				ds.close()
			for p in temp_paths:
				p.unlink(missing_ok=True)

	logger.info("Descargado: %s (%.1f MB)", target, target.stat().st_size / 1e6)
	return target


def download_era5_months(
	year_months: Iterable[tuple[int, int] | tuple[int, int, tuple[int, ...]]],
	bbox: Optional[dict] = None,
	variables: Optional[Iterable[str]] = None,
	out_dir: Path = ERA5_RAW_DIR,
	overwrite: bool = False,
) -> list[Path]:
	"""Descarga ERA5-Land para una lista de (año, mes) o (año, mes, días).

	Args:
		year_months: Iterable de tuplas (año, mes) o (año, mes, (días,)).
		bbox: Bounding box (claves north/west/south/east); None usa CHILE_BBOX.
		variables: Variables a descargar; None usa ERA5_VARIABLES.
		out_dir: Directorio de salida.
		overwrite: Si False y un archivo ya existe, omite esa descarga.

	Returns:
		Lista de rutas a los NetCDF mensuales descargados (o existentes).
	"""
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
	"""Descarga ERA5-Land para un rango de años completos (inclusivo).

	Args:
		start_year: Primer año del rango.
		end_year: Último año del rango (inclusivo).
		bbox: Bounding box (claves north/west/south/east); None usa CHILE_BBOX.
		variables: Variables a descargar; None usa ERA5_VARIABLES.
		out_dir: Directorio de salida.
		overwrite: Si False y un archivo ya existe, omite esa descarga.

	Returns:
		Lista de rutas a los NetCDF anuales descargados (o existentes).
	"""
	paths = []
	for year in range(start_year, end_year + 1):
		paths.append(download_era5_year(year, bbox, variables, out_dir, overwrite))
	return paths


def download_era5_invariants(
	bbox: Optional[dict] = None,
	invariants: Optional[Iterable[str]] = None,
	out_dir: Path = ERA5_RAW_DIR,
	overwrite: bool = False,
) -> Path:
	"""Descarga variables invariantes de ERA5-Land (un solo timestamp estático).

	Las invariantes (tipo de suelo, máscara tierra-mar, cobertura vegetal, etc.)
	no varían con el tiempo. El API de CDS igual requiere una fecha; se usa
	2002-01-01 como sentinel arbitrario. El NetCDF resultante tendrá una
	dimensión temporal degenérada (size=1) que se elimina al extraer con squeeze().

	Args:
		bbox: Bounding box (claves north/west/south/east); None usa CHILE_BBOX.
		invariants: Variables invariantes a descargar; None usa ERA5_INVARIANTS.
		out_dir: Directorio de salida.
		overwrite: Si False y el archivo ya existe, omite la descarga.

	Returns:
		Ruta al NetCDF de invariantes descargado (o existente).
	"""
	bbox = bbox or CHILE_BBOX
	invariants = list(invariants or ERA5_INVARIANTS)
	out_dir.mkdir(parents=True, exist_ok=True)
	target = era5_invariants_path(out_dir)

	if target.exists() and not overwrite:
		logger.info("ERA5 invariantes ya existen, skip: %s", target)
		return target

	request = {
		"variable": invariants,
		"year": "2002",
		"month": "01",
		"day": "01",
		"time": "00:00",
		"area": [bbox["north"], bbox["west"], bbox["south"], bbox["east"]],
		"data_format": "netcdf",
		"download_format": "unarchived",
	}

	logger.info("Solicitando invariantes ERA5-Land (%d variables)", len(invariants))
	_retrieve(request, target)
	_unzip_if_needed(target)
	logger.info("Descargado: %s (%.1f MB)", target, target.stat().st_size / 1e6)
	return target


# ============================================================
# Sección 2 — Extracción puntual nearest-neighbor
# ============================================================


def _utc_naive_timestamp(ts: pd.Timestamp) -> pd.Timestamp:
	"""Normaliza cualquier timestamp a UTC naive (sin tzinfo) para comparar con xarray.

	Args:
		ts: Timestamp de entrada, con o sin zona horaria.

	Returns:
		El timestamp en UTC sin tzinfo, o pd.NaT si la entrada es nula.
	"""
	value = pd.Timestamp(ts)
	if pd.isna(value):
		return pd.NaT
	if value.tzinfo is not None:
		value = value.tz_convert("UTC").tz_localize(None)
	return value


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
	"""Calcula la distancia haversine en km entre dos puntos (lat, lon) en grados.

	Args:
		lat1: Latitud del primer punto, en grados.
		lon1: Longitud del primer punto, en grados.
		lat2: Latitud del segundo punto, en grados.
		lon2: Longitud del segundo punto, en grados.

	Returns:
		Distancia haversine en kilómetros.
	"""
	r = 6371.0
	phi1, phi2 = np.radians(lat1), np.radians(lat2)
	dphi = np.radians(lat2 - lat1)
	dlmb = np.radians(lon2 - lon1)
	a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
	return float(2 * r * np.arcsin(np.sqrt(a)))


def _nan_result() -> dict:
	"""Construye el dict de resultado con todas las claves en None y quality "missing".

	Se usa cuando lat/lon/ts son inválidos y no se puede extraer ningún valor.

	Returns:
		Dict con las claves de EXPECTED_KEYS en None, las métricas era5_* en None y
		era5_match_quality = "missing".
	"""
	out = {k: None for k in EXPECTED_KEYS}
	out["era5_dist_km"] = None
	out["era5_dt_hours"] = None
	out["era5_land_snap_km"] = None
	out["era5_match_quality"] = "missing"
	return out


def build_land_index(ds: xr.Dataset, ref_var: str = "t2m"):
	"""Construye un buscador de la celda de tierra (valor no-NaN) más cercana.

	ERA5-Land es un producto solo-tierra: las celdas sobre mar son NaN. Para un
	punto cuyo grid cell más cercano cae en mar, este índice devuelve la celda de
	tierra más cercana y su distancia haversine, habilitando el "salto a tierra"
	(land snapping) en extract_point. Se construye una sola vez por NetCDF. El vecino
	se busca con un cKDTree en proyección equirectangular (longitud escalada por
	cos(lat medio)); el km exacto se recalcula con haversine real.

	Args:
		ds: Dataset de xarray con la variable de referencia y coordenadas lat/lon.
		ref_var: Variable usada para distinguir tierra (no-NaN) de mar (NaN).

	Returns:
		Función nearest_land(lat, lon) -> (lat, lon, dist_km) que devuelve la celda de
		tierra más cercana, o None si la variable de referencia no existe o no hay
		celdas de tierra en el NetCDF.
	"""
	ref_name = next((v for v in ds.data_vars if VAR_RENAMES.get(v, v) == ref_var), None)
	if ref_name is None:
		return None
	lat_name = "latitude" if "latitude" in ds.coords else "lat"
	lon_name = "longitude" if "longitude" in ds.coords else "lon"

	da = ds[ref_name]
	# Una celda es "tierra" si tiene dato en al menos un timestep.
	reduce_dims = [d for d in da.dims if d not in (lat_name, lon_name)]
	land2d = da.notnull().any(dim=reduce_dims) if reduce_dims else da.notnull()

	lats = ds[lat_name].values
	lons = ds[lon_name].values
	grid_lat, grid_lon = np.meshgrid(lats, lons, indexing="ij")
	mask = np.asarray(land2d.values, dtype=bool)
	if not mask.any():
		return None

	land_lat = grid_lat[mask]
	land_lon = grid_lon[mask]
	# KDTree en proyección equirectangular (lon escalada por cos(lat medio)) para el
	# vecino; el km exacto y el tope se calculan luego con haversine real.
	coslat = np.cos(np.radians(float(np.mean(lats)))) or 1.0
	tree = cKDTree(np.column_stack([land_lat, land_lon * coslat]))

	def nearest_land(lat: float, lon: float) -> tuple[float, float, float]:
		"""Busca la celda de tierra más cercana al punto consultado.

		Args:
			lat: Latitud del punto, en grados.
			lon: Longitud del punto, en grados.

		Returns:
			Tupla (lat, lon, dist_km) de la celda de tierra más cercana, con la
			distancia haversine real en kilómetros.
		"""
		_, i = tree.query([lat, lon * coslat])
		llat, llon = float(land_lat[i]), float(land_lon[i])
		return llat, llon, _haversine_km(lat, lon, llat, llon)

	return nearest_land


def extract_point(
	ds: xr.Dataset,
	lat: float,
	lon: float,
	ts: pd.Timestamp,
	land_index=None,
	max_snap_km: float = MAX_LAND_SNAP_KM,
) -> dict:
	"""Extrae los valores de ERA5 en el grid point y timestamp más cercano.

	Usa selección nearest-neighbor en las tres dimensiones (lat, lon, time).
	El dict de retorno siempre contiene todas las claves de EXPECTED_KEYS
	(con None si la variable no está en el NetCDF) más las métricas de calidad del
	match: era5_dist_km, era5_dt_hours, era5_land_snap_km, era5_match_quality.

	Salto a tierra (land snapping): si la celda más cercana cae en mar (t2m NaN) y se
	pasa `land_index` (de build_land_index), se reintenta en la celda de tierra más
	cercana siempre que esté a ≤ max_snap_km; la distancia del salto queda en
	era5_land_snap_km. La flag refleja si efectivamente hay valor: good (directo) /
	land_snapped (recuperado) / water (mar sin tierra dentro del tope) / poor (tiempo lejano).

	Args:
		ds: Dataset de xarray con las variables temporales de ERA5-Land.
		lat: Latitud del punto a extraer, en grados.
		lon: Longitud del punto a extraer, en grados.
		ts: Timestamp objetivo (se normaliza a UTC naive).
		land_index: Buscador de celda de tierra (de build_land_index) para el salto a
			tierra; None desactiva el land snapping.
		max_snap_km: Distancia máxima permitida para el salto a tierra, en kilómetros.

	Returns:
		Dict con las variables extraídas (claves de EXPECTED_KEYS, None si faltan) más
		las métricas era5_dist_km, era5_dt_hours, era5_land_snap_km y era5_match_quality.
	"""
	if pd.isna(lat) or pd.isna(lon):
		return _nan_result()

	query_ts = _utc_naive_timestamp(ts)
	if pd.isna(query_ts):
		return _nan_result()

	# Los nombres de coordenadas varían según la versión de xarray o el origen del NetCDF
	time_name = "time" if "time" in ds.coords else ("valid_time" if "valid_time" in ds.coords else "time")
	lat_name = "latitude" if "latitude" in ds.coords else "lat"
	lon_name = "longitude" if "longitude" in ds.coords else "lon"

	def _select(plat: float, plon: float):
		"""Selecciona el grid point nearest-neighbor en lat/lon y el timestamp objetivo.

		Args:
			plat: Latitud a seleccionar, en grados.
			plon: Longitud a seleccionar, en grados.

		Returns:
			El Dataset reducido al grid point y timestamp más cercanos.
		"""
		return ds.sel(
			{lat_name: plat, lon_name: plon, time_name: query_ts.to_datetime64()},
			method="nearest",
		)

	try:
		point = _select(lat, lon)
	except Exception:
		return _nan_result()

	# Variable de referencia (t2m) para detectar celda de mar (NaN) y decidir el salto.
	ref_name = next((v for v in ds.data_vars if VAR_RENAMES.get(v, v) == "t2m"), None)
	if ref_name is None and len(ds.data_vars):
		ref_name = next(iter(ds.data_vars))

	snap_km = 0.0
	if ref_name is not None and land_index is not None:
		ref_val = point[ref_name].values
		if np.ndim(ref_val) == 0 and np.isnan(ref_val):
			snapped = land_index(lat, lon)
			if snapped is not None and snapped[2] <= max_snap_km:
				try:
					point = _select(snapped[0], snapped[1])
					snap_km = snapped[2]
				except Exception:
					snap_km = 0.0

	out: dict = {}
	for var in ds.data_vars:
		key = VAR_RENAMES.get(var, var)
		val = point[var].values
		out[key] = float(val) if np.ndim(val) == 0 and not np.isnan(val) else (None if np.isnan(val) else float(val))

	# Métricas de calidad del match nearest-neighbor (sobre la celda finalmente usada)
	matched_lat = float(point[lat_name].values)
	matched_lon = float(point[lon_name].values)
	dist_km = _haversine_km(lat, lon, matched_lat, matched_lon)

	matched_time = pd.Timestamp(point[time_name].values)
	dt_hours = abs((matched_time - query_ts).total_seconds()) / 3600.0

	out["era5_dist_km"] = round(dist_km, 3)
	out["era5_dt_hours"] = round(dt_hours, 3)
	out["era5_land_snap_km"] = round(snap_km, 3)

	# Flag honesta: refleja si hay valor real, no solo la geometría del match.
	ref_key = VAR_RENAMES.get(ref_name, ref_name) if ref_name is not None else None
	has_value = ref_key is not None and out.get(ref_key) is not None
	if not has_value:
		quality = "water"
	elif dt_hours > MAX_TIME_HOURS:
		quality = "poor"
	elif snap_km > 0:
		quality = "land_snapped"
	elif dist_km <= MAX_DIST_KM:
		quality = "good"
	else:
		quality = "poor"
	out["era5_match_quality"] = quality

	# Garantiza que todas las claves esperadas existan aunque no estén en el NetCDF
	for k in EXPECTED_KEYS:
		out.setdefault(k, None)

	return out


def extract_invariant_point(ds: xr.Dataset, lat: float, lon: float) -> dict:
	"""Extrae variables invariantes en el grid point más cercano (sin dimensión temporal).

	CDS entrega las invariantes con una dimensión temporal degenérada (size=1)
	aunque los valores no varíen con el tiempo. El squeeze() la elimina para poder
	hacer la selección solo por lat/lon. Normaliza longitudes 0-360 a -180/180 si hace
	falta antes de seleccionar.

	Args:
		ds: Dataset de xarray con las variables invariantes de ERA5-Land.
		lat: Latitud del punto a extraer, en grados.
		lon: Longitud del punto a extraer, en grados.

	Returns:
		Dict con las claves de INVARIANT_KEYS (None si la variable falta o es NaN).
	"""
	if pd.isna(lat) or pd.isna(lon):
		return {k: None for k in INVARIANT_KEYS}

	lat_name = "latitude" if "latitude" in ds.coords else "lat"
	lon_name = "longitude" if "longitude" in ds.coords else "lon"

	# Normaliza longitudes 0-360 a -180/180 si es necesario
	if lon_name in ds.coords and float(ds[lon_name].max()) > 180:
		new_lons = np.where(ds[lon_name].values > 180, ds[lon_name].values - 360, ds[lon_name].values)
		ds = ds.assign_coords({lon_name: new_lons}).sortby(lon_name)

	# Elimina dimensión temporal degenérada; si time>1, toma el primer paso
	if "time" in ds.dims and ds.sizes["time"] > 1:
		ds_sq = ds.isel(time=0).squeeze(drop=True)
	else:
		ds_sq = ds.squeeze(drop=True)

	try:
		point = ds_sq.sel({lat_name: lat, lon_name: lon}, method="nearest")
	except Exception:
		return {k: None for k in INVARIANT_KEYS}

	out: dict = {}
	for var in ds_sq.data_vars:
		key = VAR_RENAMES.get(var, var)
		if key not in INVARIANT_KEYS:
			continue
		try:
			fval = float(point[var].values)
			out[key] = None if np.isnan(fval) else fval
		except (TypeError, ValueError):
			# Algunos tipos enteros de numpy no tienen NaN; float() puede fallar
			out[key] = None

	for k in INVARIANT_KEYS:
		out.setdefault(k, None)
	return out


__all__ = [
	# Descarga
	"download_era5_year",
	"download_era5_range",
	"download_era5_month",
	"download_era5_months",
	"download_era5_invariants",
	"era5_year_path",
	"era5_month_path",
	"era5_invariants_path",
	"deduplicate_lon",
	# Extracción
	"extract_point",
	"extract_invariant_point",
	"build_land_index",
	"EXPECTED_KEYS",
	"INVARIANT_KEYS",
	"MAX_DIST_KM",
	"MAX_TIME_HOURS",
	"MAX_LAND_SNAP_KM",
]
