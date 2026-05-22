"""Atribución de fuentes de datos usadas por el pipeline.

Genera un sidecar JSON (.attribution.json) junto a cada artefacto del pipeline.
El sidecar sirve como registro de provenance: qué fuentes se usaron, bajo qué
licencias, y qué transformaciones se aplicaron. Es útil para auditoría,
reproducibilidad y cumplimiento de los términos de uso de ERA5-Land (CC BY 4.0).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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

	`extra` permite añadir contexto de ejecución (parámetros del run, versión, etc.)
	bajo la clave "run" del JSON resultante.
	"""
	payload = {
		"sources": DATA_ATTRIBUTION,
		"derived_dataset": DERIVED_DATASET_NOTICE,
	}
	if extra:
		payload["run"] = extra
	return payload


def write_attribution_sidecar(out_path: Path, extra: dict[str, Any] | None = None) -> Path:
	"""Escribe el sidecar JSON junto al artefacto indicado.

	El archivo se nombra igual que el artefacto pero con extensión .attribution.json
	(e.g. conaf_enriched_2002_2020.parquet → conaf_enriched_2002_2020.attribution.json).
	"""
	sidecar = out_path.with_suffix(".attribution.json")
	sidecar.write_text(
		json.dumps(attribution_payload(extra), ensure_ascii=False, indent=2),
		encoding="utf-8",
	)
	return sidecar
