"""Reportes de features para artefactos CONAF, ERA5 y output enriquecido."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import CONAF_RAW_DIR, DATA_INTERIM, DATA_PROCESSED, ERA5_RAW_DIR
from src.era5_extractor import EXPECTED_KEYS, INVARIANT_KEYS
from src.utils import _json_safe_default

# xarray es opcional: si no está instalado, el inventario ERA5 se genera sin
# inspeccionar el contenido de los NetCDF (solo tamaños).
try:
	import xarray as xr
	_xr_available = True
except ImportError:
	_xr_available = False

CONAF_CLEAN_PARQUET = DATA_INTERIM / "conaf_clean.parquet"

# Conjuntos de columnas ERA5 para la clasificación de roles.
# Derivados dinámicamente desde era5_extractor para mantener coherencia al agregar variables.
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
}


def _safe_json_dump(payload: dict[str, Any], path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, ensure_ascii=False, indent=2, default=_json_safe_default),
		encoding="utf-8",
	)


def _read_csv(path: Path) -> pd.DataFrame:
	first_line = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
	sep = "|" if "|" in first_line else ","
	return pd.read_csv(path, sep=sep, low_memory=False)


def _canonical_name(name: str) -> str:
	"""Normaliza un nombre de columna a snake_case ASCII minúsculo.

	Flujo: unicode NFKD → strip acentos → lowercase → reemplaza caracteres
	no alfanuméricos por _ → strip underscores extremos.
	"""
	value = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
	value = value.lower().strip()
	value = value.replace("[ha]", "ha")
	value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
	return value


def _role_for_column(name: str) -> str:
	"""Clasifica una columna según su rol en el pipeline.

	Roles:
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
		import math
		if math.isnan(value) or math.isinf(value):
			return None
	if hasattr(value, "isoformat"):
		return value.isoformat()
	return value


def _examples(series: pd.Series, limit: int = 3) -> list[Any]:
	values = series.dropna().unique()[:limit]
	return [_clean_value(value) for value in values]


def _series_profile(series: pd.Series, dataset_kind: str) -> dict[str, Any]:
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
	return {
		"name": name,
		"path": str(path),
		"kind": dataset_kind,
		"rows": int(len(df)),
		"columns": int(df.shape[1]),
		"features": [_series_profile(df[column], dataset_kind) for column in df.columns],
	}


def _profile_csv(path: Path, dataset_kind: str) -> dict[str, Any]:
	df = _read_csv(path)
	return _dataframe_profile(path.name, path, df, dataset_kind)


def _profile_parquet(path: Path, dataset_kind: str) -> dict[str, Any]:
	df = pd.read_parquet(path)
	return _dataframe_profile(path.name, path, df, dataset_kind)


def _profile_era5_files() -> dict[str, Any]:
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
	return {**base, "files": inventory}


def _artifact_row(artifact: dict[str, Any]) -> str:
	return (
		f"| `{artifact.get('name')}` | `{artifact.get('kind')}` | "
		f"{artifact.get('rows', '-')} | {artifact.get('columns', '-')} | `{artifact.get('path')}` |"
	)


def _features_table(features: list[dict[str, Any]]) -> list[str]:
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
	"""Genera reporte JSON + Markdown con el perfil de features de todos los artefactos.

	Convención de nombres de archivo en CONAF_RAW_DIR:
	  - stems de 8 dígitos (e.g. "00234567"): archivos de temporada (un CSV por temporada).
	  - otros stems: índice o metadata del dataset.
	Esta convención viene del formato de Dataverse / itrend.
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
	_safe_json_dump(report, sidecar_path)
	_write_markdown(report, md_path)
	return report
