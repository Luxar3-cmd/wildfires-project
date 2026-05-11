"""Extracción puntual de ERA5: dado (lat, lon, timestamp) → diccionario de variables."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

# Umbrales para flag de calidad del nearest match
MAX_DIST_KM = 25.0
MAX_TIME_HOURS = 2.0

# Mapeo de nombres de variable CDS → claves cortas que esperamos en el output
# (xarray suele renombrar al short_name del NetCDF)
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
}

EXPECTED_KEYS = ["t2m", "d2m", "u10", "v10", "tp", "ssrd"]


def _utc_naive_timestamp(ts: pd.Timestamp) -> pd.Timestamp:
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

	Returns
	-------
	dict con claves de EXPECTED_KEYS + 'era5_match_quality' ('good'/'poor')
	+ 'era5_dist_km' + 'era5_dt_hours'.
	"""
	if pd.isna(lat) or pd.isna(lon):
		return _nan_result()

	query_ts = _utc_naive_timestamp(ts)
	if pd.isna(query_ts):
		return _nan_result()

	# Detectar nombre de coord temporal y espaciales (varía: "time"/"valid_time", "latitude"/"lat")
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

	# Distancia y delta temporal
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

	# Asegurar que todas las claves esperadas estén presentes
	for k in EXPECTED_KEYS:
		out.setdefault(k, None)

	return out


def _nan_result() -> dict:
	out = {k: None for k in EXPECTED_KEYS}
	out["era5_dist_km"] = None
	out["era5_dt_hours"] = None
	out["era5_match_quality"] = "missing"
	return out


__all__ = ["extract_point", "EXPECTED_KEYS", "MAX_DIST_KM", "MAX_TIME_HOURS"]
