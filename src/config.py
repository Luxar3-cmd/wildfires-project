"""Paths del proyecto y carga de variables de entorno."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"

CONAF_RAW_DIR = DATA_RAW / "conaf"
ERA5_RAW_DIR = DATA_RAW / "era5"

CONAF_DATASET_DOI = "doi:10.71578/UXAUN5"

CHILE_BBOX = {"north": -34.0, "west": -74.0, "south": -42.0, "east": -70.0}

ERA5_VARIABLES = [
	"2m_temperature",
	"2m_dewpoint_temperature",
	"10m_u_component_of_wind",
	"10m_v_component_of_wind",
	"total_precipitation",
	"surface_solar_radiation_downwards",
	"soil_temperature_level_1",
	"soil_temperature_level_2",
	"soil_temperature_level_3",
	"soil_temperature_level_4",
	"volumetric_soil_water_layer_1",
	"volumetric_soil_water_layer_2",
	"volumetric_soil_water_layer_3",
	"volumetric_soil_water_layer_4",
	"evaporation_from_vegetation_transpiration",
	"potential_evaporation",
	"total_evaporation",
	"leaf_area_index_high_vegetation",
	"leaf_area_index_low_vegetation",
]

ERA5_INVARIANTS = [
	"soil_type",
	"land_sea_mask",
	"high_vegetation_cover",
	"low_vegetation_cover",
	"type_of_high_vegetation",
	"type_of_low_vegetation",
]

load_dotenv(BASE_DIR / ".env")


def _env(name: str, default: str | None = None) -> str | None:
	value = os.getenv(name)
	if value is None or not value.strip():
		os.environ.pop(name, None)
		return default
	return value.strip()


ITREND_API_KEY = os.getenv("ITREND_API_KEY")
ITREND_BASE_URL = os.getenv("ITREND_BASE_URL", "https://datospararesiliencia.cl")
CDSAPI_URL = _env("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")
CDSAPI_KEY = _env("CDSAPI_KEY")


def ensure_dirs() -> None:
	for d in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, CONAF_RAW_DIR, ERA5_RAW_DIR):
		d.mkdir(parents=True, exist_ok=True)


ensure_dirs()
