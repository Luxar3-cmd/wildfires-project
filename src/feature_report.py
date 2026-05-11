"""Reportes de features para artefactos CONAF, ERA5 y output enriquecido."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import CONAF_RAW_DIR, DATA_INTERIM, DATA_PROCESSED, ERA5_RAW_DIR

CONAF_CLEAN_PARQUET = DATA_INTERIM / "conaf_clean.parquet"
FEATURES_REPORT_JSON = DATA_PROCESSED / "features_report.json"
FEATURES_REPORT_MD = DATA_PROCESSED / "features_report.md"

ERA5_RAW_COLUMNS = {"t2m", "d2m", "u10", "v10", "tp", "ssrd"}
ERA5_DERIVED_COLUMNS = {"t2m_celsius", "d2m_celsius", "relative_humidity", "wind_speed", "wind_direction", "tp_mm"}
JOIN_QUALITY_COLUMNS = {"era5_dist_km", "era5_dt_hours", "era5_match_quality"}


def _json_default(value: Any) -> Any:
	if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
		return None
	if hasattr(value, "isoformat"):
		return value.isoformat()
	if hasattr(value, "item"):
		return value.item()
	return str(value)


def _safe_json_dump(payload: dict[str, Any], path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
		encoding="utf-8",
	)


def _read_csv(path: Path) -> pd.DataFrame:
	first_line = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
	sep = "|" if "|" in first_line else ","
	return pd.read_csv(path, sep=sep, low_memory=False)


def _role_for_column(name: str) -> str:
	canonical = _canonical_name(name)
	if canonical in {"region", "provincia", "comuna", "temporada", "nombre"}:
		return "conaf_context"
	if canonical in {"fecha", "hora_inicio", "fecha_hora_inicio", "fecha_inicio", "inicio"}:
		return "time"
	if canonical == "superficie_quemada_total_ha":
		return "candidate_target"
	if canonical.startswith("superficie_quemada_") or canonical in {"causa", "alerta", "escenario", "duracion_minutos"}:
		return "conaf_feature"
	if canonical in {"latitud", "longitud", "datum", "geometry"}:
		return "location"
	if canonical in ERA5_RAW_COLUMNS:
		return "era5_raw"
	if canonical in ERA5_DERIVED_COLUMNS:
		return "era5_derived"
	if canonical in JOIN_QUALITY_COLUMNS:
		return "join_quality"
	return "unknown"


def _canonical_name(name: str) -> str:
	value = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
	value = value.lower().strip()
	value = value.replace("[ha]", "ha")
	value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
	value = value.replace("superficie_quemada_", "superficie_quemada_")
	value = value.replace("duracion_minutos", "duracion_minutos")
	if value == "hora_inicio":
		return value
	if value == "superficie_quemada_total_ha":
		return value
	return value


def _source_for_column(name: str, dataset_kind: str) -> str:
	role = _role_for_column(name)
	if role.startswith("conaf") or role in {"time", "candidate_target", "location"}:
		return "conaf"
	if role == "era5_raw":
		return "era5_land"
	if role == "era5_derived":
		return "derived_from_era5"
	if role == "join_quality":
		return "pipeline"
	if dataset_kind == "era5_netcdf":
		return "era5_land"
	return dataset_kind


def _clean_value(value: Any) -> Any:
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
	if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
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
	if not files:
		return {
			"name": "ERA5 NetCDF",
			"path": str(ERA5_RAW_DIR),
			"kind": "era5_netcdf_inventory",
			"files": [],
			"note": "No hay archivos NetCDF ERA5 locales.",
		}

	inventory = []
	try:
		import xarray as xr
	except ImportError:
		return {
			"name": "ERA5 NetCDF",
			"path": str(ERA5_RAW_DIR),
			"kind": "era5_netcdf_inventory",
			"files": [{"path": str(path), "bytes": path.stat().st_size} for path in files],
			"note": "xarray no está instalado; no se inspeccionaron variables.",
		}

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
	return {
		"name": "ERA5 NetCDF",
		"path": str(ERA5_RAW_DIR),
		"kind": "era5_netcdf_inventory",
		"files": inventory,
	}


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
	from datetime import datetime, timezone

	season_paths = sorted(
		path for path in CONAF_RAW_DIR.glob("*.csv")
		if re.fullmatch(r"\d{8}", path.stem)
	)
	index_paths = sorted(
		path for path in CONAF_RAW_DIR.glob("*.csv")
		if not re.fullmatch(r"\d{8}", path.stem)
	)

	conaf_seasons = [_profile_csv(path, "conaf_raw_season") for path in season_paths]
	conaf_index = _profile_csv(index_paths[0], "conaf_raw_index") if index_paths else None
	conaf_clean = _profile_parquet(CONAF_CLEAN_PARQUET, "conaf_clean") if CONAF_CLEAN_PARQUET.exists() else None
	enriched = _profile_parquet(enriched_path, "enriched") if enriched_path.exists() else None
	era5 = _profile_era5_files()

	artifacts = []
	artifacts.extend({
		"name": artifact["name"],
		"kind": artifact["kind"],
		"path": artifact["path"],
		"rows": artifact["rows"],
		"columns": artifact["columns"],
	} for artifact in conaf_seasons)
	if conaf_index:
		artifacts.append({key: conaf_index[key] for key in ("name", "kind", "path", "rows", "columns")})
	if conaf_clean:
		artifacts.append({key: conaf_clean[key] for key in ("name", "kind", "path", "rows", "columns")})
	if enriched:
		artifacts.append({key: enriched[key] for key in ("name", "kind", "path", "rows", "columns")})

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
