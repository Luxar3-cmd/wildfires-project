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
DATA_MODELS = DATA_DIR / "models"

CONAF_RAW_DIR = DATA_RAW / "conaf"
ERA5_RAW_DIR = DATA_RAW / "era5"

CONAF_DATASET_DOI = "doi:10.71578/UXAUN5"

# Bounding box de la zona de estudio: Maule, Biobío, Araucanía y Los Ríos.
# Se elige este recorte porque concentra >70 % de los eventos CONAF relevantes
# para el modelo XAI y permite descargas ERA5 mucho más pequeñas que Chile completo.
CHILE_BBOX = {"north": -34.0, "west": -74.0, "south": -42.0, "east": -70.0}

# Variables temporales de ERA5-Land descargadas en cada request.
# Se usan los nombres largos del API de CDS; era5_extractor.py los mapea a los
# short-names que expone xarray al leer el NetCDF.
ERA5_VARIABLES = [
	# Temperatura y punto de rocío
	"2m_temperature",
	"2m_dewpoint_temperature",
	# Temperatura del suelo por capas (0–7 cm, 7–28 cm, 28–100 cm, 100–289 cm)
	"soil_temperature_level_1",
	"soil_temperature_level_2",
	"soil_temperature_level_3",
	"soil_temperature_level_4",
	# Humedad del suelo volumétrica (mismas capas que temperatura)
	"volumetric_soil_water_layer_1",
	"volumetric_soil_water_layer_2",
	"volumetric_soil_water_layer_3",
	"volumetric_soil_water_layer_4",
	# Viento
	"10m_u_component_of_wind",
	"10m_v_component_of_wind",
	# Precipitación y radiación
	"total_precipitation",
	"surface_solar_radiation_downwards",
	# Evaporación
	"potential_evaporation",
	"total_evaporation",
	# Índice de área foliar (mide densidad del dosel vegetal)
	"leaf_area_index_high_vegetation",
	"leaf_area_index_low_vegetation",
	# Al final para quedar sola en batch 4 (CDS la silencia si va mezclada con otras)
	"evaporation_from_vegetation_transpiration",
]

# Variables invariantes de ERA5-Land: campos estáticos que no cambian con el tiempo
# (tipo de suelo, fracción de cobertura vegetal, etc.). Se descargan una sola vez
# con un único timestamp y se almacenan en un NetCDF separado.
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

# NASA FIRMS — MAP_KEY gratuita en https://firms.modaps.eosdis.nasa.gov/api/map_key/
FIRMS_MAP_KEY = _env("FIRMS_MAP_KEY")
FIRMS_BASE_URL = _env("FIRMS_BASE_URL", "https://firms.modaps.eosdis.nasa.gov/api/area/csv")

FIRMS_RAW_DIR = DATA_RAW / "firms"


def ensure_dirs() -> None:
	for d in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, DATA_MODELS, CONAF_RAW_DIR, ERA5_RAW_DIR, FIRMS_RAW_DIR):
		d.mkdir(parents=True, exist_ok=True)


ensure_dirs()
