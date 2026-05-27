# =============================================================================
# XAI-project — Interpretable Prediction of Mega-Fires in Chile (XGBoost + Tree SHAP)
# Course:  INF-473 Explainable AI · UTFSM · Prof. Raquel Pezoa Rivera
# Authors: Eduardo Morales · Octavia Jara · Benjamín Reyes
# File:    src/derived_features.py — Derived meteorological features from raw ERA5 (RH, VPD, wind, precip)
# =============================================================================
"""Features derivadas a partir de las variables crudas de ERA5."""
from __future__ import annotations

import numpy as np
import pandas as pd

KELVIN_TO_C = 273.15


def kelvin_to_celsius(t_k: pd.Series) -> pd.Series:
	return pd.to_numeric(t_k, errors="coerce") - KELVIN_TO_C


def relative_humidity(t2m_k: pd.Series, d2m_k: pd.Series) -> pd.Series:
	"""Humedad relativa (%) desde temperatura y punto de rocío en Kelvin.

	Fórmula de Magnus-Tetens (coeficientes de August-Roche-Magnus):
	    es(T) = 6.1094 * exp(17.625 * T / (T + 243.04))   [T en °C, resultado en hPa]
	    RH = 100 * es(Td) / es(T)

	Coeficientes 6.1094 / 17.625 / 243.04: aproximación AERK de Alduchov & Eskridge
	(1996), forma August-Roche-Magnus (Lawrence, 2005). Válida en el rango
	-40 °C a +50 °C. El prefactor se cancela en RH; en VPD escala el resultado.
	"""
	t = pd.to_numeric(t2m_k, errors="coerce") - KELVIN_TO_C
	td = pd.to_numeric(d2m_k, errors="coerce") - KELVIN_TO_C
	es_t = 6.1094 * np.exp(17.625 * t / (t + 243.04))
	es_td = 6.1094 * np.exp(17.625 * td / (td + 243.04))
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
	"""Precipitación total (m → mm). ERA5 entrega tp en metros acumulados por hora."""
	return pd.to_numeric(tp_m, errors="coerce") * 1000.0


def vpd_hpa(t2m_k: pd.Series, d2m_k: pd.Series) -> pd.Series:
	"""Déficit de Presión de Vapor (hPa) desde temperatura y punto de rocío en Kelvin.

	VPD = e_sat(T) - e_sat(Td), donde e_sat es la presión de vapor de saturación.
	Físicamente: cuánta presión de vapor "falta" para que el aire esté saturado.
	Un VPD alto indica aire seco y caliente → mayor demanda evaporativa →
	mayor estrés hídrico en la vegetación → mayor riesgo de incendio.

	Mismos coeficientes Magnus-Tetens que relative_humidity() para consistencia.
	clip(lower=0): el VPD no puede ser negativo; valores cercanos a cero pueden dar
	float ligeramente negativo por errores de punto flotante.
	"""
	t = pd.to_numeric(t2m_k, errors="coerce") - KELVIN_TO_C
	td = pd.to_numeric(d2m_k, errors="coerce") - KELVIN_TO_C
	es_t = 6.1094 * np.exp(17.625 * t / (t + 243.04))
	es_td = 6.1094 * np.exp(17.625 * td / (td + 243.04))
	return (es_t - es_td).clip(lower=0)


def add_all(df: pd.DataFrame) -> pd.DataFrame:
	"""Aplica todas las features derivadas al DataFrame y retorna copia con columnas nuevas.

	Opera sobre las variables crudas de ERA5 (t2m, d2m, u10, v10, tp, stl1-4).
	Si alguna columna base no existe, se omite la derivación correspondiente —
	esto permite usar el mismo código con subsets del dataset.
	"""
	df = df.copy()
	if "t2m" in df.columns:
		df["t2m_celsius"] = kelvin_to_celsius(df["t2m"])
	if "d2m" in df.columns:
		df["d2m_celsius"] = kelvin_to_celsius(df["d2m"])
	if {"t2m", "d2m"}.issubset(df.columns):
		df["relative_humidity"] = relative_humidity(df["t2m"], df["d2m"])
		df["vpd_hpa"] = vpd_hpa(df["t2m"], df["d2m"])
	if {"u10", "v10"}.issubset(df.columns):
		df["wind_speed"] = wind_speed(df["u10"], df["v10"])
		df["wind_direction"] = wind_direction(df["u10"], df["v10"])
	if "tp" in df.columns:
		df["tp_mm"] = precipitation_mm(df["tp"])
	# Temperatura de suelo por capas: de Kelvin a Celsius
	for level in (1, 2, 3, 4):
		col = f"stl{level}"
		if col in df.columns:
			df[f"{col}_celsius"] = kelvin_to_celsius(df[col])
	return df


__all__ = [
	"kelvin_to_celsius",
	"relative_humidity",
	"wind_speed",
	"wind_direction",
	"precipitation_mm",
	"vpd_hpa",
	"add_all",
]
