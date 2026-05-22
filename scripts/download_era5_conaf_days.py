"""Descarga ERA5-Land solo para los días con eventos CONAF en 2016 y 2017.

Salida: data/raw/era5_conaf_days/
  era5_land_2016_01.nc, ..., era5_land_2017_12.nc
  era5_land_invariants.nc

Uso:
  PYTHONPATH=. python scripts/download_era5_conaf_days.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.conaf_loader import load_conaf
from src.config import ERA5_RAW_DIR
from src.era5_downloader import download_era5_invariants, download_era5_months
from src.pipeline import _download_bbox, _needed_year_months, filter_conaf_years

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
log = logging.getLogger(__name__)

START_YEAR, END_YEAR = 2016, 2017
OUT_DIR = ERA5_RAW_DIR.parent / "era5_conaf_days"

conaf = load_conaf()
conaf = filter_conaf_years(conaf, START_YEAR, END_YEAR)

year_months = _needed_year_months(conaf)
bbox = _download_bbox(conaf)

total_days = sum(len(days) for _, _, days in year_months)
log.info(
	"CONAF %d-%d: %d eventos → %d mes(es), %d día(s) únicos",
	START_YEAR, END_YEAR, len(conaf), len(year_months), total_days,
)
log.info("Bbox: %s", bbox)
log.info("Destino: %s", OUT_DIR)
for year, month, days in year_months:
	log.info("  %04d-%02d → días %s", year, month, sorted(days))

download_era5_months(year_months, bbox=bbox, out_dir=OUT_DIR)
download_era5_invariants(bbox=bbox, out_dir=OUT_DIR)
log.info("Descarga completa. Archivos en %s", OUT_DIR)
