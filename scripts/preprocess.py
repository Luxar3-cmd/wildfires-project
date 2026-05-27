# =============================================================================
# XAI-project — Interpretable Prediction of Mega-Fires in Chile (XGBoost + Tree SHAP)
# Course:  INF-473 Explainable AI · UTFSM · Prof. Raquel Pezoa Rivera
# Authors: Eduardo Morales · Octavia Jara · Benjamín Reyes
# File:    scripts/preprocess.py — CLI: run the CONAF+ERA5(+MODIS) preprocessing pipeline end-to-end
# =============================================================================
"""CLI de preprocesamiento: descarga CONAF + ERA5 y produce dataset enriquecido.

Uso:
    python scripts/preprocess.py --years 2002-2020
    python scripts/preprocess.py --years 2019-2019 --skip-download
    python scripts/preprocess.py --years 2016-2017 --download-only --era5-dir data/raw/era5_conaf_days
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

# Permite ejecutar como `python scripts/preprocess.py` sin instalar el paquete
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from src.conaf_loader import load_conaf  # noqa: E402
from src.config import ERA5_RAW_DIR  # noqa: E402
from src.era5 import download_era5_invariants, download_era5_months  # noqa: E402
from src.pipeline import (  # noqa: E402
	_download_bbox,
	_needed_year_months,
	filter_conaf_years,
	parse_year_range,
	run_pipeline,
)

app = typer.Typer(add_completion=False, help="Preprocesamiento CONAF + ERA5")

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
	datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)


def _download_only(start_year: int, end_year: int, era5_dir: Path) -> None:
	"""Descarga ERA5 acotado a los días con eventos CONAF, sin enriquecer."""
	conaf = load_conaf()
	conaf = filter_conaf_years(conaf, start_year, end_year)
	year_months = _needed_year_months(conaf)
	bbox = _download_bbox(conaf)

	total_days = sum(len(days) for _, _, days in year_months)
	typer.secho(
		f"CONAF {start_year}-{end_year}: {len(conaf)} eventos → "
		f"{len(year_months)} mes(es), {total_days} día(s) únicos",
		fg=typer.colors.CYAN,
	)
	typer.echo(f"Bbox: {bbox}")
	typer.echo(f"Destino: {era5_dir}")
	for year, month, days in year_months:
		typer.echo(f"  {year:04d}-{month:02d} → días {sorted(days)}")

	download_era5_months(year_months, bbox=bbox, out_dir=era5_dir)
	download_era5_invariants(bbox=bbox, out_dir=era5_dir)
	typer.secho(f"Descarga completa. Archivos en {era5_dir}", fg=typer.colors.GREEN)


@app.command()
def main(
	years: str = typer.Option("2002-2020", "--years", help="Rango YYYY-YYYY"),
	skip_download: bool = typer.Option(False, "--skip-download", help="Asume ERA5 ya descargado"),
	download_only: bool = typer.Option(False, "--download-only", help="Solo descarga ERA5; no enriquece ni escribe parquet"),
	skip_modis: bool = typer.Option(False, "--skip-modis", help="Saltar descarga MODIS y cálculo de label_l2"),
	refresh_conaf: bool = typer.Option(False, "--refresh-conaf", help="Re-descarga CONAF aunque haya cache"),
	out: Path | None = typer.Option(None, "--out", help="Ruta parquet opcional para el output versionado"),
	era5_dir: Path | None = typer.Option(None, "--era5-dir", help="Directorio ERA5 (default: data/raw/era5/)"),
):
	"""Pipeline completo: CONAF → ERA5 → dataset enriquecido."""
	try:
		start_year, end_year = parse_year_range(years)
	except ValueError as e:
		raise typer.BadParameter(str(e)) from e

	if download_only:
		if skip_download:
			raise typer.BadParameter("--download-only y --skip-download son incompatibles")
		_download_only(start_year, end_year, era5_dir or ERA5_RAW_DIR)
		return

	try:
		summary = run_pipeline(
			start_year,
			end_year,
			skip_download=skip_download,
			refresh_conaf=refresh_conaf,
			skip_modis=skip_modis,
			out_path=out,
			era5_dir=era5_dir,
		)
	except RuntimeError as e:
		typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
		raise typer.Exit(code=1) from e

	output = summary["output"]
	typer.secho(
		f"Listo: {output['rows']} filas, {output['columns']} columnas",
		fg=typer.colors.GREEN,
	)
	typer.echo(f"Output versionado: {output['versioned_output']}")
	typer.echo(f"Output latest: {output['latest_output']}")
	typer.echo(f"Informe de features: {output['features_report_path']}")


if __name__ == "__main__":
	app()
