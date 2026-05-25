"""MODIS FIRMS: descarga FRP histórico + matching CONAF + label L2 (Tedim 2018).

Fuente: NASA FIRMS Area API (https://firms.modaps.eosdis.nasa.gov/api/area/).
Producto subyacente: MODIS Thermal Anomalies/Fire Level 2 5-Min Swath 1 km
  (MOD14 Terra / MYD14 Aqua, Collection 6.1). FIRMS expone los fire pixels del
  L2 swath como CSV — preserva el FRP instantáneo de cada detección, insumo de
  la ecuación de Wooster 2003 (los composites diarios A1/A2 lo promedian y no sirven).
Auth: MAP_KEY gratuita (variable de entorno FIRMS_MAP_KEY).
Rate limit: 5000 transacciones / 10 min.

Flujo:
  1. download_firms_for_conaf(conaf) — descarga CSVs MODIS cubriendo días con eventos (±1d).
  2. load_firms_csvs(paths)          — concatena y normaliza acq_datetime_utc.
  3. match_modis_to_conaf(conaf, m)  — para cada evento, max FRP dentro de (5km, ±24h).
  4. label_l2(enriched, matches)     — Wooster FRP→FLI + umbral EWE 10.000 kW/m.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import CHILE_BBOX, FIRMS_BASE_URL, FIRMS_MAP_KEY, FIRMS_RAW_DIR

logger = logging.getLogger(__name__)

FIRMS_SOURCE = "MODIS_SP"          # Standard Processing (histórico)
DAY_RANGE_MAX = 5                  # máximo de días por request (límite API)

# --- Wooster 2003 ---
RADIANT_FRACTION = 0.17            # fracción radiante default (rango 0.13–0.20)
MODIS_PIXEL_LENGTH_M = 1000.0      # longitud del frente = tamaño pixel MODIS nadir (1 km)
                                   # Interpretación "peak local": el píxel más caliente del frente
                                   # ↔ su propia longitud. Evita diluir el FRP de 1 píxel sobre el
                                   # frente completo (el área CONAF es acumulada, no instantánea).

# --- Tedim 2018 ---
FLI_EWE_THRESHOLD_KW_M = 10000.0   # umbral categoría 5+ (Extreme Wildfire Event)
MIN_AREA_HA_FOR_L2 = 50.0          # guardia de coherencia área–FLI: un EWE no ocupa <50 ha.
                                   # = límite de detección MODIS; descarta falsos positivos donde
                                   # el radio de matching captó el FRP de un megaincendio vecino.

# --- Matching CONAF ↔ MODIS ---
MATCH_RADIUS_KM = 5.0
MATCH_TIME_HOURS = 24.0

REQUEST_TIMEOUT_S = 90


# ============================================================
# Descarga FIRMS
# ============================================================


def firms_csv_path(start_date: date, day_range: int, out_dir: Path = FIRMS_RAW_DIR) -> Path:
	"""Path local del CSV cacheado para el bloque (start_date, day_range)."""
	end_date = start_date + timedelta(days=day_range - 1)
	return out_dir / f"modis_sp_{start_date:%Y%m%d}_{end_date:%Y%m%d}.csv"


def _firms_url(bbox: dict, start_date: date, day_range: int, source: str = FIRMS_SOURCE) -> str:
	"""URL del endpoint Area. AREA_COORDINATES = west,south,east,north."""
	if not FIRMS_MAP_KEY:
		raise RuntimeError(
			"Falta FIRMS_MAP_KEY en .env. Obtén tu MAP_KEY en\n"
			"  https://firms.modaps.eosdis.nasa.gov/api/map_key/\n"
			"y agrégala al archivo .env como FIRMS_MAP_KEY."
		)
	area = f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}"
	return f"{FIRMS_BASE_URL}/{FIRMS_MAP_KEY}/{source}/{area}/{day_range}/{start_date:%Y-%m-%d}"


def download_firms_block(
	start_date: date,
	day_range: int = DAY_RANGE_MAX,
	bbox: dict | None = None,
	out_dir: Path = FIRMS_RAW_DIR,
	overwrite: bool = False,
) -> Path:
	"""Descarga un bloque histórico MODIS_SP (≤5 días) para el bbox dado."""
	bbox = bbox or CHILE_BBOX
	if not 1 <= day_range <= DAY_RANGE_MAX:
		raise ValueError(f"day_range debe estar en [1, {DAY_RANGE_MAX}]; recibido {day_range}")
	out_dir.mkdir(parents=True, exist_ok=True)
	target = firms_csv_path(start_date, day_range, out_dir)
	if target.exists() and not overwrite:
		logger.info("FIRMS %s (%dd) ya existe, skip: %s", start_date, day_range, target.name)
		return target

	url = _firms_url(bbox, start_date, day_range)
	logger.info("Descargando FIRMS %s (%d día(s))", start_date, day_range)
	response = requests.get(url, timeout=REQUEST_TIMEOUT_S)
	response.raise_for_status()

	# FIRMS devuelve 200 con cuerpo de texto plano ante errores (MAP_KEY inválido, etc.)
	body = response.content
	head = body[:200].decode("utf-8", errors="ignore").lower()
	if not head.startswith("latitude") and "country_id" not in head and "lat" not in head:
		raise RuntimeError(f"Respuesta FIRMS inesperada para {start_date}: {head[:160]!r}")

	target.write_bytes(body)
	return target


def _event_days_padded(conaf: pd.DataFrame) -> list[date]:
	"""Días únicos con evento CONAF, expandidos ±1 día.

	El padding ±1d garantiza que el matching temporal (±24h) tenga FRP disponible
	aunque la detección MODIS caiga en el día anterior o siguiente al inicio del fuego.
	"""
	ts_col = next((c for c in ("fecha_hora_inicio_utc", "fecha_hora_inicio") if c in conaf.columns), None)
	if ts_col is None:
		raise KeyError("CONAF sin columna de timestamp (fecha_hora_inicio_utc / fecha_hora_inicio).")
	event_days = pd.to_datetime(conaf[ts_col], errors="coerce").dt.date.dropna().unique()
	padded: set[date] = set()
	for d in event_days:
		padded.update({d - timedelta(days=1), d, d + timedelta(days=1)})
	return sorted(padded)


def _group_into_blocks(days: list[date]) -> list[tuple[date, int]]:
	"""Agrupa días ordenados en bloques contiguos de hasta DAY_RANGE_MAX días.

	Retorna [(start_date, day_range), ...]. Un día aislado produce (día, 1).
	"""
	if not days:
		return []
	blocks: list[tuple[date, int]] = []
	block_start = days[0]
	last = days[0]
	for d in days[1:]:
		if (d - block_start).days < DAY_RANGE_MAX:
			last = d
		else:
			blocks.append((block_start, (last - block_start).days + 1))
			block_start = d
			last = d
	blocks.append((block_start, (last - block_start).days + 1))
	return blocks


def download_firms_for_conaf(
	conaf: pd.DataFrame,
	bbox: dict | None = None,
	out_dir: Path = FIRMS_RAW_DIR,
) -> list[Path]:
	"""Descarga los CSVs FIRMS que cubren los días con eventos CONAF (±1d)."""
	bbox = bbox or CHILE_BBOX
	days = _event_days_padded(conaf)
	if not days:
		logger.warning("Sin días válidos en CONAF — no se descarga FIRMS")
		return []
	blocks = _group_into_blocks(days)
	logger.info("FIRMS: %d días (con padding ±1d) → %d bloque(s) de descarga", len(days), len(blocks))
	return [download_firms_block(start, day_range, bbox, out_dir) for start, day_range in blocks]


def load_firms_csvs(paths: list[Path]) -> pd.DataFrame:
	"""Carga y concatena CSVs FIRMS; normaliza acq_datetime_utc (naive UTC).

	acq_time llega como entero HHMM (e.g. 1335 = 13:35). Se combina con acq_date.
	"""
	if not paths:
		return pd.DataFrame()
	frames = []
	for path in paths:
		try:
			df = pd.read_csv(path)
		except (pd.errors.EmptyDataError, FileNotFoundError):
			logger.warning("CSV FIRMS vacío o ausente: %s", path.name)
			continue
		if "latitude" in df.columns and len(df):
			frames.append(df)
	if not frames:
		return pd.DataFrame()

	df = pd.concat(frames, ignore_index=True).drop_duplicates()
	# acq_time → HHMM con zero-padding; acq_date → YYYY-MM-DD
	acq_time = df["acq_time"].astype("Int64").astype(str).str.zfill(4)
	df["acq_datetime_utc"] = pd.to_datetime(
		df["acq_date"].astype(str) + acq_time,
		format="%Y-%m-%d%H%M",
		errors="coerce",
	)
	return df


# ============================================================
# Matching espacio-temporal
# ============================================================


def _haversine_km(lat1: float, lon1: float, lat2, lon2):
	"""Distancia haversine (km). Vectorizada sobre lat2/lon2 (arrays numpy)."""
	r = 6371.0
	phi1, phi2 = np.radians(lat1), np.radians(lat2)
	dphi = np.radians(lat2 - lat1)
	dlmb = np.radians(lon2 - lon1)
	a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
	return 2 * r * np.arcsin(np.sqrt(a))


def match_modis_to_conaf(
	conaf: pd.DataFrame,
	modis: pd.DataFrame,
	radius_km: float = MATCH_RADIUS_KM,
	time_window_h: float = MATCH_TIME_HOURS,
) -> pd.DataFrame:
	"""Para cada evento CONAF, agrega las detecciones MODIS dentro de (radius_km, ±time_window_h).

	Retorna DataFrame alineado al index de `conaf` con columnas:
	  modis_n_matches   — # detecciones MODIS dentro de la ventana
	  modis_frp_max_mw  — max FRP [MW] de los matches (NaN si ninguno)
	  modis_frp_sum_mw  — sum FRP [MW] de los matches (NaN si ninguno)
	"""
	empty = pd.DataFrame(
		{
			"modis_n_matches": np.zeros(len(conaf), dtype=int),
			"modis_frp_max_mw": np.full(len(conaf), np.nan),
			"modis_frp_sum_mw": np.full(len(conaf), np.nan),
		},
		index=conaf.index,
	)
	if modis.empty:
		return empty

	ts_col = next(c for c in ("fecha_hora_inicio_utc", "fecha_hora_inicio") if c in conaf.columns)
	lat_col = next(c for c in ("latitud", "latitude", "lat") if c in conaf.columns)
	lon_col = next(c for c in ("longitud", "longitude", "lon") if c in conaf.columns)

	conaf_ts = pd.to_datetime(conaf[ts_col], errors="coerce")
	m_lat = modis["latitude"].to_numpy(dtype=float)
	m_lon = modis["longitude"].to_numpy(dtype=float)
	m_ts = modis["acq_datetime_utc"].to_numpy()  # datetime64[ns]
	m_frp = pd.to_numeric(modis["frp"], errors="coerce").to_numpy(dtype=float)
	window_ns = np.timedelta64(int(time_window_h * 3600), "s")

	n_matches = np.zeros(len(conaf), dtype=int)
	frp_max = np.full(len(conaf), np.nan)
	frp_sum = np.full(len(conaf), np.nan)

	for pos, (idx, row) in enumerate(conaf.iterrows()):
		ts = conaf_ts.iloc[pos]
		lat, lon = row[lat_col], row[lon_col]
		if pd.isna(ts) or pd.isna(lat) or pd.isna(lon):
			continue
		# Filtro temporal (vectorizado)
		time_mask = np.abs(m_ts - np.datetime64(ts)) <= window_ns
		if not time_mask.any():
			continue
		# Filtro espacial sobre los candidatos temporales
		dist = _haversine_km(float(lat), float(lon), m_lat[time_mask], m_lon[time_mask])
		spatial = dist <= radius_km
		if not spatial.any():
			continue
		frps = m_frp[time_mask][spatial]
		frps = frps[~np.isnan(frps)]
		n_matches[pos] = int(spatial.sum())
		if frps.size:
			frp_max[pos] = float(frps.max())
			frp_sum[pos] = float(frps.sum())

	return pd.DataFrame(
		{"modis_n_matches": n_matches, "modis_frp_max_mw": frp_max, "modis_frp_sum_mw": frp_sum},
		index=conaf.index,
	)


# ============================================================
# Conversión FRP → FLI (Wooster 2003) y label L2
# ============================================================


def frp_to_fli(
	frp_mw: float,
	front_length_m: float,
	radiant_fraction: float = RADIANT_FRACTION,
) -> float:
	"""FLI [kW/m] desde FRP [MW] (Wooster 2003).

	FLI [kW/m] = (FRP [MW] · 1000 / η_r) / front_length [m]

	Derivación dimensional:
	  P_total [W]  = FRP [MW]·1e6 / η_r
	  FLI   [W/m]  = P_total / front_length
	  FLI  [kW/m]  = FLI[W/m] / 1e3  =  (FRP[MW]·1e3 / η_r) / front_length
	"""
	if pd.isna(frp_mw) or frp_mw <= 0 or front_length_m <= 0:
		return np.nan
	return (float(frp_mw) * 1000.0 / radiant_fraction) / float(front_length_m)


def label_l2(
	enriched: pd.DataFrame,
	matches: pd.DataFrame,
	radiant_fraction: float = RADIANT_FRACTION,
	fli_threshold: float = FLI_EWE_THRESHOLD_KW_M,
	min_area_ha: float = MIN_AREA_HA_FOR_L2,
) -> pd.DataFrame:
	"""Añade columnas L2 al DataFrame enriquecido y retorna copia.

	Interpretación "peak local": la FLI se estima sobre el píxel MODIS más caliente
	(frp_max) usando su propia longitud (MODIS_PIXEL_LENGTH_M = 1 km). El label marca
	si alguna parte del frente alcanzó intensidad EWE. Equivale a frp_max ≥ 1700 MW.

	Guardia de coherencia: además exige superficie_quemada_total_ha >= min_area_ha, para
	descartar falsos positivos donde el radio de matching captó el FRP de un fuego vecino
	(un EWE no ocupa <50 ha).

	Columnas nuevas:
	  modis_n_matches    # detecciones MODIS dentro de la ventana
	  modis_frp_max_mw   max FRP [MW] (NaN si sin match)
	  fli_estimado_kw_m  FLI por Wooster sobre el píxel pico (NaN si sin FRP)
	  label_l2           1 si FLI >= umbral EWE y superficie >= min_area_ha; 0 en otro caso
	"""
	out = enriched.copy()
	out["modis_n_matches"] = matches["modis_n_matches"].to_numpy()
	out["modis_frp_max_mw"] = matches["modis_frp_max_mw"].to_numpy()
	out["fli_estimado_kw_m"] = [
		frp_to_fli(frp, MODIS_PIXEL_LENGTH_M, radiant_fraction)
		for frp in out["modis_frp_max_mw"]
	]
	# NaN >= umbral → False → 0. Default 0: sin detección satelital ⇒ no catastrófico.
	fli_ok = out["fli_estimado_kw_m"] >= fli_threshold
	if "superficie_quemada_total_ha" in out.columns:
		area_ok = pd.to_numeric(out["superficie_quemada_total_ha"], errors="coerce") >= min_area_ha
		fli_ok = fli_ok & area_ok
	out["label_l2"] = fli_ok.astype(int)
	return out


def l2_summary(enriched: pd.DataFrame) -> dict:
	"""Resumen de la obtención de FRP y del label L2 (para log + summary del pipeline)."""
	n_total = int(len(enriched))
	n_match = int((enriched["modis_n_matches"] > 0).sum())
	n_frp_valid = int((pd.to_numeric(enriched["modis_frp_max_mw"], errors="coerce") > 0).sum())
	n_l2_pos = int((enriched["label_l2"] == 1).sum())

	def pct(n: int) -> float:
		return round(100.0 * n / n_total, 2) if n_total else 0.0

	return {
		"n_eventos_total": n_total,
		"n_con_deteccion_modis": n_match,
		"pct_con_deteccion": pct(n_match),
		"n_con_frp_valido": n_frp_valid,
		"pct_con_frp_valido": pct(n_frp_valid),
		"n_label_l2_positivo": n_l2_pos,
		"pct_label_l2_positivo": pct(n_l2_pos),
		"n_label_l2_negativo": n_total - n_l2_pos,
	}


def log_l2_summary(summary: dict) -> None:
	"""Imprime el resumen L2 en el log con formato legible."""
	logger.info(
		"L2/MODIS resumen:\n"
		"  eventos totales:      %8d\n"
		"  con detección MODIS:  %8d (%.1f%%)\n"
		"  con FRP>0 válido:     %8d (%.1f%%)\n"
		"  label_l2 = 1 (EWE):   %8d (%.2f%%)\n"
		"  label_l2 = 0:         %8d",
		summary["n_eventos_total"],
		summary["n_con_deteccion_modis"], summary["pct_con_deteccion"],
		summary["n_con_frp_valido"], summary["pct_con_frp_valido"],
		summary["n_label_l2_positivo"], summary["pct_label_l2_positivo"],
		summary["n_label_l2_negativo"],
	)


__all__ = [
	"download_firms_block",
	"download_firms_for_conaf",
	"load_firms_csvs",
	"match_modis_to_conaf",
	"frp_to_fli",
	"label_l2",
	"l2_summary",
	"log_l2_summary",
	"firms_csv_path",
	"FIRMS_RAW_DIR",
	"FLI_EWE_THRESHOLD_KW_M",
	"MIN_AREA_HA_FOR_L2",
	"RADIANT_FRACTION",
	"MODIS_PIXEL_LENGTH_M",
	"MATCH_RADIUS_KM",
	"MATCH_TIME_HOURS",
]
