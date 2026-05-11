"""Features derivadas a partir de las variables crudas de ERA5."""
from __future__ import annotations

import numpy as np
import pandas as pd

KELVIN_TO_C = 273.15


def kelvin_to_celsius(t_k: pd.Series) -> pd.Series:
	return pd.to_numeric(t_k, errors="coerce") - KELVIN_TO_C


def relative_humidity(t2m_k: pd.Series, d2m_k: pd.Series) -> pd.Series:
	"""Humedad relativa (%) desde temperatura y punto de rocío en Kelvin.

	Fórmula de Magnus-Tetens:
		es(T) = 6.112 * exp(17.625 * T / (T + 243.04))
		RH = 100 * es(Td) / es(T)
	"""
	t = pd.to_numeric(t2m_k, errors="coerce") - KELVIN_TO_C
	td = pd.to_numeric(d2m_k, errors="coerce") - KELVIN_TO_C
	es_t = 6.112 * np.exp(17.625 * t / (t + 243.04))
	es_td = 6.112 * np.exp(17.625 * td / (td + 243.04))
	rh = 100.0 * es_td / es_t
	return rh.clip(lower=0, upper=100)


def wind_speed(u: pd.Series, v: pd.Series) -> pd.Series:
	"""Magnitud del viento (m/s) desde componentes u, v."""
	u = pd.to_numeric(u, errors="coerce")
	v = pd.to_numeric(v, errors="coerce")
	return np.sqrt(u**2 + v**2)


def wind_direction(u: pd.Series, v: pd.Series) -> pd.Series:
	"""Dirección del viento meteorológica (grados desde donde sopla, 0=N, 90=E)."""
	u = pd.to_numeric(u, errors="coerce")
	v = pd.to_numeric(v, errors="coerce")
	deg = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
	return deg


def precipitation_mm(tp_m: pd.Series) -> pd.Series:
	"""Precipitación total (m → mm). ERA5 entrega tp en metros."""
	return pd.to_numeric(tp_m, errors="coerce") * 1000.0


def add_all(df: pd.DataFrame) -> pd.DataFrame:
	"""Aplica todas las features derivadas al DataFrame in-place y retorna.

	Espera columnas crudas: t2m, d2m, u10, v10, tp.
	Si alguna no existe, se omite la derivación correspondiente.
	"""
	df = df.copy()
	if "t2m" in df.columns:
		df["t2m_celsius"] = kelvin_to_celsius(df["t2m"])
	if "d2m" in df.columns:
		df["d2m_celsius"] = kelvin_to_celsius(df["d2m"])
	if {"t2m", "d2m"}.issubset(df.columns):
		df["relative_humidity"] = relative_humidity(df["t2m"], df["d2m"])
	if {"u10", "v10"}.issubset(df.columns):
		df["wind_speed"] = wind_speed(df["u10"], df["v10"])
		df["wind_direction"] = wind_direction(df["u10"], df["v10"])
	if "tp" in df.columns:
		df["tp_mm"] = precipitation_mm(df["tp"])
	return df


__all__ = [
	"kelvin_to_celsius",
	"relative_humidity",
	"wind_speed",
	"wind_direction",
	"precipitation_mm",
	"add_all",
]
