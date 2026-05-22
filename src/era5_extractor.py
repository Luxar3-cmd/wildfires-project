"""Extracción puntual de ERA5: dado (lat, lon, timestamp) → diccionario de variables."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

# Umbral de distancia y delta temporal para clasificar un match como "good".
# MAX_DIST_KM=25 cubre la resolución nativa de ERA5-Land (~9 km) con margen amplio.
# MAX_TIME_HOURS=2 tolera desfases horarios típicos al redondear timestamps de incendio.
MAX_DIST_KM = 25.0
MAX_TIME_HOURS = 2.0

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
	"evavt", "pev", "e",
	"lai_hv", "lai_lv",
]

# INVARIANT_KEYS: variables estáticas extraídas del NetCDF de invariantes (sin dimensión temporal).
INVARIANT_KEYS = ["slt", "lsm", "cvh", "cvl", "tvh", "tvl"]


def _utc_naive_timestamp(ts: pd.Timestamp) -> pd.Timestamp:
	"""Normaliza cualquier timestamp a UTC naive (sin tzinfo) para comparar con xarray."""
	value = pd.Timestamp(ts)
	if pd.isna(value):
		return pd.NaT
	if value.tzinfo is not None:
		value = value.tz_convert("UTC").tz_localize(None)
	return value


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
	"""Distancia haversine en km entre dos puntos (lat, lon) en grados."""
	r = 6371.0
	phi1, phi2 = np.radians(lat1), np.radians(lat2)
	dphi = np.radians(lat2 - lat1)
	dlmb = np.radians(lon2 - lon1)
	a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
	return float(2 * r * np.arcsin(np.sqrt(a)))


def extract_point(
	ds: xr.Dataset,
	lat: float,
	lon: float,
	ts: pd.Timestamp,
) -> dict:
	"""Extrae los valores de ERA5 en el grid point y timestamp más cercano.

	Usa selección nearest-neighbor en las tres dimensiones (lat, lon, time).
	El dict de retorno siempre contiene todas las claves de EXPECTED_KEYS
	(con None si la variable no está en el NetCDF) más las métricas de calidad
	del match: era5_dist_km, era5_dt_hours, era5_match_quality.
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

	try:
		point = ds.sel(
			{lat_name: lat, lon_name: lon, time_name: query_ts.to_datetime64()},
			method="nearest",
		)
	except Exception:
		return _nan_result()

	out: dict = {}
	for var in ds.data_vars:
		key = VAR_RENAMES.get(var, var)
		val = point[var].values
		out[key] = float(val) if np.ndim(val) == 0 and not np.isnan(val) else (None if np.isnan(val) else float(val))

	# Métricas de calidad del match nearest-neighbor
	matched_lat = float(point[lat_name].values)
	matched_lon = float(point[lon_name].values)
	dist_km = _haversine_km(lat, lon, matched_lat, matched_lon)

	matched_time = pd.Timestamp(point[time_name].values)
	dt_hours = abs((matched_time - query_ts).total_seconds()) / 3600.0

	out["era5_dist_km"] = round(dist_km, 3)
	out["era5_dt_hours"] = round(dt_hours, 3)
	out["era5_match_quality"] = (
		"good" if (dist_km <= MAX_DIST_KM and dt_hours <= MAX_TIME_HOURS) else "poor"
	)

	# Garantiza que todas las claves esperadas existan aunque no estén en el NetCDF
	for k in EXPECTED_KEYS:
		out.setdefault(k, None)

	return out


def _nan_result() -> dict:
	"""Dict vacío con todas las claves en None — se usa cuando lat/lon/ts son inválidos."""
	out = {k: None for k in EXPECTED_KEYS}
	out["era5_dist_km"] = None
	out["era5_dt_hours"] = None
	out["era5_match_quality"] = "missing"
	return out


def extract_invariant_point(ds: xr.Dataset, lat: float, lon: float) -> dict:
	"""Extrae variables invariantes en el grid point más cercano (sin dimensión temporal).

	CDS entrega las invariantes con una dimensión temporal degenérada (size=1)
	aunque los valores no varíen con el tiempo. El squeeze() la elimina para poder
	hacer la selección solo por lat/lon.
	"""
	if pd.isna(lat) or pd.isna(lon):
		return {k: None for k in INVARIANT_KEYS}

	lat_name = "latitude" if "latitude" in ds.coords else "lat"
	lon_name = "longitude" if "longitude" in ds.coords else "lon"

	# Elimina cualquier dimensión degenérada (e.g. time con size=1) antes de seleccionar
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
	"extract_point",
	"extract_invariant_point",
	"EXPECTED_KEYS",
	"INVARIANT_KEYS",
	"MAX_DIST_KM",
	"MAX_TIME_HOURS",
]
