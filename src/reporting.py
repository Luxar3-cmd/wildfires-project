# =============================================================================
# XAI-project — Interpretable Prediction of Mega-Fires in Chile (XGBoost + Tree SHAP)
# Course:  INF-473 Explainable AI · UTFSM · Prof. Raquel Pezoa Rivera
# Authors: Eduardo Morales · Octavia Jara · Benjamín Reyes
# File:    src/reporting.py — Source attribution (CC-BY sidecars) and feature-profile reports
# =============================================================================
"""Reporting del pipeline: atribución de fuentes y perfil de features.

Dos responsabilidades cohesivas en un solo módulo:

1. Atribución (sidecar JSON `.attribution.json`):
   - metadata de fuentes primarias (CONAF, ERA5-Land)
   - notas de licencia CC-BY 4.0 obligatorias
   - registro de provenance del dataset derivado

2. Feature report (sidecar JSON `.features.json` + Markdown `features_report.md`):
   - perfil por columna (dtype, nulls, ejemplos, min/max)
   - clasificación de roles (conaf_context, era5_raw, era5_derived, etc.)
   - inventario de NetCDFs ERA5 locales
"""
from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import CONAF_RAW_DIR, DATA_INTERIM, DATA_PROCESSED, ERA5_RAW_DIR
from src.era5 import EXPECTED_KEYS, INVARIANT_KEYS

logger = logging.getLogger(__name__)

# xarray es opcional: si no está instalado, el inventario ERA5 se genera sin
# inspeccionar el contenido de los NetCDF (solo tamaños).
try:
	import xarray as xr
	_xr_available = True
except ImportError:
	_xr_available = False


# ============================================================
# Sección 1 — Atribución de fuentes
# ============================================================

# Metadata de las fuentes primarias usadas por el pipeline.
# "required_notice" refleja la obligación de atribución de CC BY 4.0 para ERA5-Land.
DATA_ATTRIBUTION = {
	"conaf": {
		"title": "Registro histórico de incendios forestales",
		"provider": "CONAF / itrend - Datos para Resiliencia",
		"source": "https://datospararesiliencia.cl",
		"doi": "10.71578/UXAUN5",
		"notes": "Ver términos del dataset en la plataforma de origen.",
	},
	"era5_land": {
		"title": "ERA5-Land hourly data from 1981 to present",
		"provider": "Copernicus Climate Change Service (C3S) / ECMWF",
		"creator": "Muñoz Sabater, J.",
		"year": 2019,
		"doi": "10.24381/cds.e2161bac",
		"source": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land",
		"license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
		"license_url": "https://creativecommons.org/licenses/by/4.0/",
		"required_notice": (
			"Al compartir outputs que incluyan o deriven de ERA5-Land, dar crédito a la fuente, "
			"incluir el enlace a CC BY 4.0 e indicar cambios realizados."
		),
	},
}

# Descripción del dataset derivado: qué transformaciones se aplicaron sobre las fuentes.
DERIVED_DATASET_NOTICE = {
	"description": "Dataset derivado que cruza eventos CONAF con variables climáticas ERA5-Land por ubicación y timestamp aproximado.",
	"changes": [
		"Limpieza y normalización de columnas CONAF.",
		"Filtrado por rango temporal solicitado.",
		"Extracción nearest-neighbor de ERA5-Land por latitud, longitud y timestamp.",
		"Cálculo de features derivadas: temperatura Celsius, humedad relativa, VPD, viento y precipitación en mm.",
	],
	"no_endorsement": "La atribución no implica respaldo de CONAF, itrend, Copernicus, C3S o ECMWF.",
	"no_warranty": "Las fuentes se entregan sin garantías según sus términos/licencias aplicables.",
}


def attribution_payload(extra: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Ensambla el payload completo de atribución.

	Args:
		extra: Contexto de ejecución opcional (parámetros del run, versión, etc.)
			que se incluye bajo la clave "run" del JSON resultante.

	Returns:
		Diccionario con las claves "sources" y "derived_dataset", más "run"
		si se entregó `extra`.
	"""
	payload = {
		"sources": DATA_ATTRIBUTION,
		"derived_dataset": DERIVED_DATASET_NOTICE,
	}
	if extra:
		payload["run"] = extra
	return payload


def write_attribution_sidecar(out_path: Path, extra: dict[str, Any] | None = None) -> Path:
	"""Escribe el sidecar JSON de attribution junto al artefacto indicado.

	El archivo se nombra igual que el artefacto pero con extensión .attribution.json
	(e.g. conaf_enriched_2002_2020.parquet → conaf_enriched_2002_2020.attribution.json).

	Args:
		out_path: Ruta del artefacto para el que se genera el sidecar.
		extra: Contexto de ejecución opcional que se propaga al payload.

	Returns:
		Ruta del sidecar escrito.
	"""
	sidecar = out_path.with_suffix(".attribution.json")
	sidecar.write_text(
		json.dumps(attribution_payload(extra), ensure_ascii=False, indent=2),
		encoding="utf-8",
	)
	return sidecar


# ============================================================
# Sección 2 — Helper JSON safe
# ============================================================


def _json_safe_default(obj: Any) -> Any:
	"""Convierte tipos no estándar para usarse como `default` de json.dumps.

	Convierte: Path → str, numpy scalars → Python nativo, datetime → isoformat,
	pd.NaN / float NaN / inf → None. Todo lo demás cae a str().

	Args:
		obj: Valor que json.dumps no pudo serializar de forma nativa.

	Returns:
		Representación JSON-serializable del valor.
	"""
	if isinstance(obj, Path):
		return str(obj)
	try:
		if pd.isna(obj):
			return None
	except (TypeError, ValueError):
		pass
	if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
		return None
	if hasattr(obj, "isoformat"):
		return obj.isoformat()
	if hasattr(obj, "item"):
		# numpy scalar (int64, float32, etc.) → Python nativo
		return obj.item()
	return str(obj)


# ============================================================
# Sección 3 — Feature report
# ============================================================

CONAF_CLEAN_PARQUET = DATA_INTERIM / "conaf_clean.parquet"

# Conjuntos de columnas ERA5 para la clasificación de roles.
# Derivados dinámicamente desde src.era5 para mantener coherencia al agregar variables.
ERA5_RAW_COLUMNS = set(EXPECTED_KEYS)
ERA5_INVARIANT_COLUMNS = set(INVARIANT_KEYS)
ERA5_DERIVED_COLUMNS = {
	"t2m_celsius", "d2m_celsius",
	"stl1_celsius", "stl2_celsius", "stl3_celsius", "stl4_celsius",
	"relative_humidity", "vpd_hpa",
	"wind_speed", "wind_direction",
	"tp_mm",
}
JOIN_QUALITY_COLUMNS = {"era5_dist_km", "era5_dt_hours", "era5_match_quality"}

# Lookup para _role_for_column: columnas con rol fijo por nombre canónico.
# Las columnas que no aparecen aquí se clasifican por prefijo o por pertenencia a los sets ERA5.
_ROLE_LOOKUP: dict[str, str] = {
	"region": "conaf_context", "provincia": "conaf_context", "comuna": "conaf_context",
	"temporada": "conaf_context", "nombre": "conaf_context",
	"fecha": "time", "hora_inicio": "time", "fecha_hora_inicio": "time",
	"fecha_inicio": "time", "inicio": "time",
	"latitud": "location", "longitud": "location", "datum": "location", "geometry": "location",
	"superficie_quemada_total_ha": "candidate_target",
	"causa": "conaf_feature", "alerta": "conaf_feature",
	"escenario": "conaf_feature", "duracion_minutos": "conaf_feature",
	# Label L2 (MODIS-FRP → FLI, Tedim 2018)
	"modis_n_matches": "modis_match", "modis_frp_max_mw": "modis_feature",
	"modis_frp_sum_mw": "modis_feature",
	"fli_estimado_kw_m": "modis_derived",
	"label_l2": "target",
}


def _safe_json_dump(payload: dict[str, Any], path: Path) -> None:
	"""Serializa un payload a JSON creando los directorios padre necesarios.

	Usa `_json_safe_default` como fallback para tipos no serializables.

	Args:
		payload: Diccionario a volcar como JSON.
		path: Ruta de destino del archivo.
	"""
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, ensure_ascii=False, indent=2, default=_json_safe_default),
		encoding="utf-8",
	)


def _read_csv(path: Path) -> pd.DataFrame:
	"""Lee un CSV detectando si el separador es "|" o ",".

	Inspecciona la primera línea para elegir el separador antes de parsear.

	Args:
		path: Ruta del archivo CSV.

	Returns:
		DataFrame con el contenido del CSV.
	"""
	lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
	sep = "|" if lines and "|" in lines[0] else ","
	return pd.read_csv(path, sep=sep, low_memory=False)


def _canonical_name(name: str) -> str:
	"""Normaliza un nombre de columna a snake_case ASCII minúsculo.

	Flujo: unicode NFKD → strip acentos → lowercase → reemplaza caracteres
	no alfanuméricos por _ → strip underscores extremos.

	Args:
		name: Nombre de columna original.

	Returns:
		Nombre canónico en snake_case ASCII.
	"""
	value = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
	value = value.lower().strip()
	value = value.replace("[ha]", "ha")
	value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
	return value


def _role_for_column(name: str) -> str:
	"""Clasifica una columna según su rol en el pipeline.

	Resuelve primero por lookup de nombre canónico, luego por prefijo y por
	pertenencia a los sets de columnas ERA5.

	Args:
		name: Nombre de columna a clasificar.

	Returns:
		Etiqueta de rol, una de:
		  conaf_context    : identificadores del evento (región, temporada, nombre)
		  time             : columnas de fecha/hora
		  location         : coordenadas geográficas
		  candidate_target : variable objetivo del modelo (superficie_quemada_total_ha)
		  conaf_feature    : otras variables del evento (causa, alerta, duración, vegetación)
		  era5_raw         : variables meteorológicas crudas de ERA5-Land
		  era5_invariant   : variables estáticas de ERA5 (tipo suelo, cobertura)
		  era5_derived     : features calculadas a partir de ERA5 (Celsius, VPD, etc.)
		  join_quality     : métricas de calidad del nearest-neighbor match
		  unknown          : no clasificada
	"""
	canonical = _canonical_name(name)
	if canonical in _ROLE_LOOKUP:
		return _ROLE_LOOKUP[canonical]
	if canonical.startswith("superficie_quemada_"):
		return "conaf_feature"
	if canonical in ERA5_RAW_COLUMNS:
		return "era5_raw"
	if canonical in ERA5_INVARIANT_COLUMNS:
		return "era5_invariant"
	if canonical in ERA5_DERIVED_COLUMNS:
		return "era5_derived"
	if canonical in JOIN_QUALITY_COLUMNS:
		return "join_quality"
	return "unknown"


def _source_for_column(name: str, dataset_kind: str) -> str:
	"""Determina la fuente de datos de origen de una columna.

	Deriva la fuente a partir del rol de la columna; si el rol no es concluyente,
	usa `dataset_kind` como fallback.

	Args:
		name: Nombre de columna.
		dataset_kind: Tipo de artefacto del que proviene la columna (e.g.
			"conaf_raw_season", "era5_netcdf").

	Returns:
		Identificador de fuente: "conaf", "era5_land", "pipeline" o el propio
		`dataset_kind`.
	"""
	role = _role_for_column(name)
	if role.startswith("conaf") or role in {"time", "candidate_target", "location"}:
		return "conaf"
	if role == "era5_raw":
		return "era5_land"
	if role in {"era5_invariant", "era5_derived"}:
		return "era5_land"
	if role == "join_quality":
		return "pipeline"
	if dataset_kind == "era5_netcdf":
		return "era5_land"
	return dataset_kind


def _clean_value(value: Any) -> Any:
	"""Convierte un valor a un tipo JSON-serializable.

	Más exhaustivo que _json_safe_default: también maneja bytes y float inf.
	Se usa al construir ejemplos y estadísticas del perfil de columnas.

	Args:
		value: Valor crudo extraído de una Series de pandas.

	Returns:
		Valor JSON-serializable; None para nulos, NaN o inf; hex truncado a 64
		caracteres para bytes.
	"""
	if value is None:
		return None
	try:
		if pd.isna(value):
			return None
	except TypeError:
		pass
	if hasattr(value, "item"):
		value = value.item()
	if isinstance(value, bytes):
		return value.hex()[:64]
	if isinstance(value, float):
		if math.isnan(value) or math.isinf(value):
			return None
	if hasattr(value, "isoformat"):
		return value.isoformat()
	return value


def _examples(series: pd.Series, limit: int = 3) -> list[Any]:
	"""Extrae valores de ejemplo distintos y no nulos de una Series.

	Args:
		series: Series de la que tomar ejemplos.
		limit: Cantidad máxima de ejemplos a devolver.

	Returns:
		Lista de hasta `limit` valores únicos ya saneados con `_clean_value`.
	"""
	values = series.dropna().unique()[:limit]
	return [_clean_value(value) for value in values]


def _series_profile(series: pd.Series, dataset_kind: str) -> dict[str, Any]:
	"""Construye el perfil de una columna individual.

	Incluye dtype, conteo y porcentaje de nulos, ejemplos, fuente y rol. Para
	columnas numéricas o de fecha agrega también min y max.

	Args:
		series: Columna a perfilar.
		dataset_kind: Tipo de artefacto de origen, usado para resolver la fuente.

	Returns:
		Diccionario con el perfil de la columna.
	"""
	null_count = int(series.isna().sum())
	profile: dict[str, Any] = {
		"name": str(series.name),
		"dtype": str(series.dtype),
		"null_count": null_count,
		"null_pct": round((null_count / len(series) * 100), 4) if len(series) else 0.0,
		"examples": _examples(series),
		"source": _source_for_column(str(series.name), dataset_kind),
		"role": _role_for_column(str(series.name)),
	}

	if pd.api.types.is_numeric_dtype(series):
		profile["min"] = _clean_value(series.min(skipna=True))
		profile["max"] = _clean_value(series.max(skipna=True))
	elif pd.api.types.is_datetime64_any_dtype(series):
		profile["min"] = _clean_value(series.min(skipna=True))
		profile["max"] = _clean_value(series.max(skipna=True))

	return profile


def _dataframe_profile(name: str, path: Path, df: pd.DataFrame, dataset_kind: str) -> dict[str, Any]:
	"""Construye el perfil completo de un DataFrame.

	Args:
		name: Nombre legible del artefacto.
		path: Ruta del archivo de origen.
		df: DataFrame a perfilar.
		dataset_kind: Tipo de artefacto.

	Returns:
		Diccionario con metadata del artefacto y el perfil de cada columna.
	"""
	return {
		"name": name,
		"path": str(path),
		"kind": dataset_kind,
		"rows": int(len(df)),
		"columns": int(df.shape[1]),
		"features": [_series_profile(df[column], dataset_kind) for column in df.columns],
	}


def _profile_csv(path: Path, dataset_kind: str) -> dict[str, Any]:
	"""Lee y perfila un artefacto CSV.

	Args:
		path: Ruta del archivo CSV.
		dataset_kind: Tipo de artefacto.

	Returns:
		Perfil del DataFrame leído.
	"""
	df = _read_csv(path)
	return _dataframe_profile(path.name, path, df, dataset_kind)


def _profile_parquet(path: Path, dataset_kind: str) -> dict[str, Any]:
	"""Lee y perfila un artefacto Parquet.

	Args:
		path: Ruta del archivo Parquet.
		dataset_kind: Tipo de artefacto.

	Returns:
		Perfil del DataFrame leído.
	"""
	df = pd.read_parquet(path)
	return _dataframe_profile(path.name, path, df, dataset_kind)


def _profile_era5_files() -> dict[str, Any]:
	"""Genera el inventario de los NetCDF ERA5 locales.

	Si no hay archivos, devuelve una nota. Si xarray no está disponible, lista
	solo rutas y tamaños. En caso contrario inspecciona dimensiones y variables
	de cada NetCDF; los errores de lectura por archivo se registran y se anexan
	como entrada con clave "error".

	Returns:
		Diccionario con metadata del directorio ERA5 y la lista de archivos.
	"""
	files = sorted(ERA5_RAW_DIR.glob("*.nc"))
	base = {
		"name": "ERA5 NetCDF",
		"path": str(ERA5_RAW_DIR),
		"kind": "era5_netcdf_inventory",
	}
	if not files:
		return {**base, "files": [], "note": "No hay archivos NetCDF ERA5 locales."}
	if not _xr_available:
		return {
			**base,
			"files": [{"path": str(path), "bytes": path.stat().st_size} for path in files],
			"note": "xarray no está instalado; no se inspeccionaron variables.",
		}

	inventory = []
	for path in files:
		try:
			with xr.open_dataset(path) as ds:
				inventory.append({
					"name": path.name,
					"path": str(path),
					"bytes": path.stat().st_size,
					"dims": {key: int(value) for key, value in ds.sizes.items()},
					"features": [
						{
							"name": name,
							"dtype": str(var.dtype),
							"dims": list(var.dims),
							"shape": [int(size) for size in var.shape],
							"source": "era5_land",
							"role": "era5_raw",
							"units": var.attrs.get("units"),
							"long_name": var.attrs.get("long_name"),
						}
						for name, var in ds.data_vars.items()
					],
				})
		except Exception as exc:
			logger.warning("No se pudo leer %s: %s", path.name, exc)
			inventory.append({
				"name": path.name,
				"path": str(path),
				"bytes": path.stat().st_size if path.exists() else None,
				"error": str(exc),
			})
	return {**base, "files": inventory}


def _artifact_row(artifact: dict[str, Any]) -> str:
	"""Formatea un artefacto como fila de la tabla Markdown de artefactos.

	Args:
		artifact: Diccionario con las claves name, kind, rows, columns y path.

	Returns:
		Fila Markdown con las celdas del artefacto.
	"""
	return (
		f"| `{artifact.get('name')}` | `{artifact.get('kind')}` | "
		f"{artifact.get('rows', '-')} | {artifact.get('columns', '-')} | `{artifact.get('path')}` |"
	)


def _features_table(features: list[dict[str, Any]]) -> list[str]:
	"""Construye la tabla Markdown con el perfil de un conjunto de features.

	Args:
		features: Lista de perfiles de columna (salida de `_series_profile`).

	Returns:
		Lista de líneas Markdown (encabezado, separador y una fila por feature).
	"""
	lines = [
		"| Feature | dtype | nulls | null % | source | role | examples | min | max |",
		"|---|---:|---:|---:|---|---|---|---:|---:|",
	]
	for feature in features:
		examples = ", ".join(str(value) for value in feature.get("examples", []))
		lines.append(
			f"| `{feature['name']}` | `{feature['dtype']}` | {feature['null_count']} | "
			f"{feature['null_pct']} | `{feature['source']}` | `{feature['role']}` | "
			f"{examples} | {feature.get('min', '')} | {feature.get('max', '')} |"
		)
	return lines


def _write_markdown(report: dict[str, Any], path: Path) -> None:
	"""Renderiza el feature report como Markdown y lo escribe en disco.

	Genera secciones para artefactos, inventario ERA5 local, CONAF limpio,
	dataset enriquecido, índice CONAF y CSV CONAF por temporada.

	Args:
		report: Diccionario del feature report (salida de `generate_feature_report`).
		path: Ruta de destino del archivo Markdown.
	"""
	lines = [
		"# Informe de features",
		"",
		f"Generado: `{report['generated_at']}`",
		"",
		"## Artefactos",
		"",
		"| Artefacto | Tipo | Filas | Columnas | Ruta |",
		"|---|---|---:|---:|---|",
	]
	lines.extend(_artifact_row(artifact) for artifact in report["artifacts"])
	lines.extend([
		"",
		"## ERA5 local",
		"",
		report["era5"].get("note") or f"Archivos NetCDF encontrados: {len(report['era5'].get('files', []))}",
		"",
	])

	for section_title, artifact in (
		("CONAF limpio", report.get("conaf_clean")),
		("Dataset enriquecido", report.get("enriched")),
		("Índice CONAF", report.get("conaf_index")),
	):
		if not artifact:
			continue
		lines.extend([f"## {section_title}", "", f"Ruta: `{artifact['path']}`", ""])
		lines.extend(_features_table(artifact.get("features", [])))
		lines.append("")

	lines.extend(["## CSV CONAF por temporada", ""])
	for artifact in report.get("conaf_seasons", []):
		lines.extend([f"### `{artifact['name']}`", "", f"Filas: {artifact['rows']}; columnas: {artifact['columns']}", ""])
		lines.extend(_features_table(artifact.get("features", [])))
		lines.append("")

	path.write_text("\n".join(lines), encoding="utf-8")


def generate_feature_report(enriched_path: Path, out_dir: Path = DATA_PROCESSED) -> dict[str, Any]:
	"""Genera el feature report JSON + Markdown con el perfil de todos los artefactos.

	Perfila los CSV de temporada e índice de CONAF, el CONAF limpio, el dataset
	enriquecido y el inventario ERA5 local. Escribe `features_report.json`,
	`features_report.md` y un sidecar `.features.json` junto al artefacto
	enriquecido.

	Convención de nombres de archivo en CONAF_RAW_DIR:
	  - stems de 8 dígitos (e.g. "00234567"): archivos de temporada (un CSV por temporada).
	  - otros stems: índice o metadata del dataset.
	Esta convención viene del formato de Dataverse / itrend.

	Args:
		enriched_path: Ruta del dataset enriquecido a perfilar y junto al cual
			se escribe el sidecar.
		out_dir: Directorio de salida para los reportes JSON y Markdown.

	Returns:
		Diccionario del report, incluyendo las rutas escritas bajo la clave "paths".
	"""
	from datetime import datetime, timezone

	# Archivos de temporada: stems numéricos de exactamente 8 dígitos
	season_paths = sorted(
		path for path in CONAF_RAW_DIR.glob("*.csv")
		if re.fullmatch(r"\d{8}", path.stem)
	)
	# Archivos de índice o metadata: cualquier otro CSV en el mismo directorio
	index_paths = sorted(
		path for path in CONAF_RAW_DIR.glob("*.csv")
		if not re.fullmatch(r"\d{8}", path.stem)
	)

	conaf_seasons = [_profile_csv(path, "conaf_raw_season") for path in season_paths]
	conaf_index = _profile_csv(index_paths[0], "conaf_raw_index") if index_paths else None
	conaf_clean = _profile_parquet(CONAF_CLEAN_PARQUET, "conaf_clean") if CONAF_CLEAN_PARQUET.exists() else None
	enriched = _profile_parquet(enriched_path, "enriched") if enriched_path.exists() else None
	era5 = _profile_era5_files()

	_ARTIFACT_KEYS = ("name", "kind", "path", "rows", "columns")
	artifacts = []
	artifacts.extend({k: artifact[k] for k in _ARTIFACT_KEYS} for artifact in conaf_seasons)
	if conaf_index:
		artifacts.append({k: conaf_index[k] for k in _ARTIFACT_KEYS})
	if conaf_clean:
		artifacts.append({k: conaf_clean[k] for k in _ARTIFACT_KEYS})
	if enriched:
		artifacts.append({k: enriched[k] for k in _ARTIFACT_KEYS})

	report = {
		"generated_at": datetime.now(timezone.utc).isoformat(),
		"artifacts": artifacts,
		"conaf_seasons": conaf_seasons,
		"conaf_index": conaf_index,
		"conaf_clean": conaf_clean,
		"era5": era5,
		"enriched": enriched,
	}

	json_path = out_dir / "features_report.json"
	md_path = out_dir / "features_report.md"
	sidecar_path = enriched_path.with_suffix(".features.json")
	report["paths"] = {
		"json": str(json_path),
		"markdown": str(md_path),
		"sidecar": str(sidecar_path),
	}
	_safe_json_dump(report, json_path)
	if enriched_path.exists():
		_safe_json_dump(report, sidecar_path)
	_write_markdown(report, md_path)
	return report


__all__ = [
	# Atribución
	"attribution_payload",
	"write_attribution_sidecar",
	"DATA_ATTRIBUTION",
	"DERIVED_DATASET_NOTICE",
	# Feature report
	"generate_feature_report",
	# Helper JSON
	"_json_safe_default",
]
