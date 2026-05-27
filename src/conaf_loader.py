# =============================================================================
# XAI-project — Interpretable Prediction of Mega-Fires in Chile (XGBoost + Tree SHAP)
# Course:  INF-473 Explainable AI · UTFSM · Prof. Raquel Pezoa Rivera
# Authors: Eduardo Morales · Octavia Jara · Benjamín Reyes
# File:    src/conaf_loader.py — Download and clean the CONAF historical wildfire registry (Dataverse/itrend)
# =============================================================================
"""Descarga y limpieza del registro histórico de incendios forestales de CONAF.

Fuente: itrend / Plataforma de Datos para la Resiliencia ante Desastres
        (Dataverse en https://datospararesiliencia.cl)
DOI:    doi:10.71578/UXAUN5

Acceso programático: pyDataverse 0.3.3
Credenciales: https://datospararesiliencia.cl/dataverseuser.xhtml → API Token
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd

from src.config import (
	CONAF_DATASET_DOI,
	CONAF_RAW_DIR,
	DATA_INTERIM,
	ITREND_API_KEY,
	ITREND_BASE_URL,
)

logger = logging.getLogger(__name__)

CLEAN_PARQUET = DATA_INTERIM / "conaf_clean.parquet"
META_FILE = CONAF_RAW_DIR / "_dataverse_meta.json"
CONAF_LOCAL_TIMEZONE = "America/Santiago"


def _slugify_column(name: str) -> str:
	"""Normaliza nombres de columna a snake_case ASCII."""
	nfkd = unicodedata.normalize("NFKD", str(name))
	ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
	cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", ascii_only).strip("_").lower()
	return cleaned or "col"


def _local_to_utc_naive(ts: pd.Series) -> pd.Series:
	"""Convierte timestamps de hora local de Chile a UTC sin zona (naive)."""
	values = pd.to_datetime(ts, errors="coerce")
	if isinstance(values.dtype, pd.DatetimeTZDtype):
		return values.dt.tz_convert("UTC").dt.tz_localize(None)

	# ambiguous="NaT": durante el cambio de hora en Chile, un instante local
	# puede corresponder a dos momentos UTC distintos. Se marca NaT en vez de
	# elegir arbitrariamente cuál de los dos usar.
	localized = values.dt.tz_localize(CONAF_LOCAL_TIMEZONE, ambiguous="NaT", nonexistent="shift_forward")
	return localized.dt.tz_convert("UTC").dt.tz_localize(None)


def _download_via_dataverse(dest_dir: Path) -> Path:
	"""Descarga TODOS los archivos del dataset desde Dataverse y retorna el path principal."""
	try:
		from pyDataverse.api import NativeApi, DataAccessApi
	except ImportError as e:
		raise ImportError(
			"Falta pyDataverse 0.3.3. Instálalo con:\n"
			"    pip install pyDataverse==0.3.3"
		) from e

	if not ITREND_API_KEY:
		raise RuntimeError(
			"Falta credencial. Obtén tu API key en\n"
			"  https://datospararesiliencia.cl/dataverseuser.xhtml (sección API Token)\n"
			"y agrégala al archivo .env como ITREND_API_KEY."
		)

	dest_dir.mkdir(parents=True, exist_ok=True)

	api = NativeApi(ITREND_BASE_URL, ITREND_API_KEY)
	data_api = DataAccessApi(ITREND_BASE_URL, ITREND_API_KEY)

	logger.info("Consultando metadata del dataset %s en %s", CONAF_DATASET_DOI, ITREND_BASE_URL)
	dataset = api.get_dataset(CONAF_DATASET_DOI)
	if dataset.status_code != 200:
		raise RuntimeError(
			f"Dataverse respondió {dataset.status_code}: {dataset.text[:300]}\n"
			f"Verifica tu API key y que el DOI {CONAF_DATASET_DOI} existe."
		)

	files_list = dataset.json()["data"]["latestVersion"]["files"]
	logger.info("Dataset tiene %d archivo(s):", len(files_list))
	manifest = []
	for f in files_list:
		df = f["dataFile"]
		size_mb = df.get("filesize", 0) / 1e6 if df.get("filesize") else None
		logger.info("  - id=%s | %s%s", df["id"], df["filename"], f" ({size_mb:.1f} MB)" if size_mb else "")
		manifest.append({"id": df["id"], "filename": df["filename"], "contentType": df.get("contentType")})

	# Descarga uno por uno (más robusto que el ZIP combinado: si un archivo falla, los otros no se pierden)
	for f in files_list:
		df = f["dataFile"]
		filename = df["filename"]
		file_id = df["id"]
		out = dest_dir / filename
		if out.exists() and out.stat().st_size > 0:
			logger.info("  skip %s (ya existe)", filename)
			continue
		logger.info("  descargando %s (id=%s)...", filename, file_id)
		response = data_api.get_datafile(file_id, is_pid=False)
		if response.status_code != 200:
			logger.warning("    fallo HTTP %s para %s", response.status_code, filename)
			continue
		out.write_bytes(response.content)
		logger.info("    guardado %.1f MB en %s", len(response.content) / 1e6, out)

	META_FILE.write_text(
		json.dumps({"doi": CONAF_DATASET_DOI, "files": manifest}, ensure_ascii=False, indent=2)
	)

	primary = _find_primary_file(dest_dir)
	if primary is None:
		raise FileNotFoundError(f"Descarga completada pero no encontré CSV/GeoJSON/SHP en {dest_dir}")
	return primary


def _find_dataset_files(dest_dir: Path) -> list[Path]:
	"""Localiza archivos del dataset por prioridad de formato.

	Convención del dataset CONAF en Dataverse (itrend): los archivos de temporada
	tienen stems numéricos de exactamente 8 dígitos (e.g. "00234567.csv"). Cualquier
	otro CSV (índice, metadata) tiene stem no numérico o de longitud distinta.
	"""
	for ext in ("*.csv", "*.geojson", "*.shp"):
		candidates = sorted(
			path for path in dest_dir.rglob(ext)
			if re.fullmatch(r"\d{8}", path.stem)
		)
		if candidates:
			return candidates
	return []


def _find_primary_file(dest_dir: Path) -> Optional[Path]:
	files = _find_dataset_files(dest_dir)
	return files[0] if files else None


def _read_dataset(path: Path) -> gpd.GeoDataFrame:
	"""Lee CSV/GeoJSON/SHP y retorna siempre un GeoDataFrame.

	Para CSV usa autodetección de separador (sep=None, engine='python').
	"""
	suffix = path.suffix.lower()
	if suffix in {".shp", ".geojson"}:
		return gpd.read_file(path)
	if suffix == ".csv":
		first_line = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
		if "|" in first_line:
			df = pd.read_csv(path, sep="|")
		else:
			df = pd.read_csv(path, sep=None, engine="python")
		geom = None
		lat_col = next((c for c in df.columns if c.lower() in {"latitud", "latitude", "lat"}), None)
		lon_col = next((c for c in df.columns if c.lower() in {"longitud", "longitude", "lon", "lng"}), None)
		if lat_col and lon_col:
			lat = pd.to_numeric(df[lat_col], errors="coerce")
			lon = pd.to_numeric(df[lon_col], errors="coerce")
			geom = gpd.points_from_xy(lon, lat)
		return gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")
	raise ValueError(f"Extensión no soportada: {suffix}")


def _read_datasets(paths: list[Path]) -> gpd.GeoDataFrame:
	if not paths:
		raise FileNotFoundError(f"No encontré CSV/GeoJSON/SHP en {CONAF_RAW_DIR}")

	frames = []
	for path in paths:
		logger.info("Leyendo archivo crudo: %s", path)
		frames.append(_read_dataset(path))

	if len(frames) == 1:
		return frames[0]

	combined = pd.concat(frames, ignore_index=True)
	geometry_col = "geometry" if "geometry" in combined.columns else None
	return gpd.GeoDataFrame(combined, geometry=geometry_col, crs=frames[0].crs)


def _clean(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
	"""Limpieza estándar del GeoDataFrame de CONAF."""
	gdf = gdf.rename(columns={c: _slugify_column(c) for c in gdf.columns})

	date_cols = [c for c in gdf.columns if "fecha" in c]
	for col in date_cols:
		gdf[col] = pd.to_datetime(gdf[col], errors="coerce")

	hora_inicio = next((c for c in gdf.columns if c in {"hora_inicio", "hora"}), None)
	fecha_inicio = next((c for c in gdf.columns if c in {"fecha_inicio", "inicio", "fecha"}), None)
	if fecha_inicio and hora_inicio:
		hora = gdf[hora_inicio].astype("string").str.strip()
		hora = hora.mask(hora.str.lower().isin({"nan", "nat", "none"}), "")
		# La columna hora_inicio llega como string de formato variable ("8:30", "08:30", "8:30:00").
		# Se concatena en string en vez de usar aritmética de Timestamp porque pd.to_datetime
		# tolera múltiples formatos de hora en una misma pasada, mientras que timedelta requiere
		# parsear la hora a mano primero.
		combined = (
			gdf[fecha_inicio].dt.strftime("%Y-%m-%d").fillna("") + " " + hora.fillna("")
		).str.strip()
		gdf["fecha_hora_inicio"] = pd.to_datetime(combined, errors="coerce")
	elif fecha_inicio:
		gdf["fecha_hora_inicio"] = gdf[fecha_inicio]
	if "fecha_hora_inicio" in gdf.columns:
		gdf["fecha_hora_inicio_utc"] = _local_to_utc_naive(gdf["fecha_hora_inicio"])

	gdf = gdf.drop_duplicates()

	if "latitud" in gdf.columns:
		gdf["latitud"] = pd.to_numeric(gdf["latitud"], errors="coerce")
	if "longitud" in gdf.columns:
		gdf["longitud"] = pd.to_numeric(gdf["longitud"], errors="coerce")

	if gdf.geometry is not None and gdf.geometry.notna().any():
		if "latitud" not in gdf.columns:
			gdf["latitud"] = gdf.geometry.y
		if "longitud" not in gdf.columns:
			gdf["longitud"] = gdf.geometry.x

	return gdf


def _looks_like_lost_time_component(gdf: pd.DataFrame) -> bool:
	"""Detecta si el cache tiene timestamps truncados a medianoche.

	Versiones antiguas del dataset CONAF tenían la hora_inicio en una columna
	separada que se perdía durante la exportación a Parquet. El síntoma es que
	todos los fecha_hora_inicio terminan en 00:00 aunque hora_inicio tenga
	valores reales (no cero). Si esto ocurre, se descarta el cache y se
	reconstruye concatenando fecha + hora desde los crudos.
	"""
	hora_inicio = next((c for c in gdf.columns if c in {"hora_inicio", "hora"}), None)
	if not hora_inicio or "fecha_hora_inicio" not in gdf.columns:
		return False

	hora = gdf[hora_inicio].astype("string").str.strip().str.lower()
	has_non_midnight_time = hora.notna() & ~hora.isin({"", "nan", "nat", "none", "00:00", "0:00", "00:00:00"})
	if not has_non_midnight_time.any():
		return False

	ts = pd.to_datetime(gdf["fecha_hora_inicio"], errors="coerce")
	check_mask = has_non_midnight_time & ts.notna()
	if not check_mask.any():
		return False

	midnight = ts.dt.hour.eq(0) & ts.dt.minute.eq(0) & ts.dt.second.eq(0)
	return bool(midnight[check_mask].all())


def load_conaf(refresh: bool = False, save_clean: bool = True) -> gpd.GeoDataFrame:
	"""Carga el dataset CONAF, descargando si es necesario.

	Parameters
	----------
	refresh : si True, vuelve a descargar incluso si ya hay archivos en disco.
	save_clean : si True, guarda copia limpia en data/interim/conaf_clean.parquet.
	"""
	if not refresh and CLEAN_PARQUET.exists():
		logger.info("Cargando cache limpio: %s", CLEAN_PARQUET)
		cached = gpd.read_parquet(CLEAN_PARQUET)
		if (
			"fecha_hora_inicio" in cached.columns
			and "fecha_hora_inicio_utc" in cached.columns
			and not _looks_like_lost_time_component(cached)
		):
			return cached
		logger.warning("Cache limpio con timestamp incompleto; reconstruyendo desde crudos")

	files = _find_dataset_files(CONAF_RAW_DIR) if not refresh else []

	if not files:
		_download_via_dataverse(CONAF_RAW_DIR)
		files = _find_dataset_files(CONAF_RAW_DIR)
	else:
		logger.info("Usando %d archivo(s) crudo(s) existente(s)", len(files))

	gdf = _read_datasets(files)
	gdf = _clean(gdf)

	if save_clean:
		CLEAN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
		gdf.to_parquet(CLEAN_PARQUET)
		logger.info("Guardado cache limpio en %s (%d filas)", CLEAN_PARQUET, len(gdf))

	return gdf


__all__ = ["load_conaf", "CLEAN_PARQUET"]
