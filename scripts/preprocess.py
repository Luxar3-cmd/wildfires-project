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
from src.era5 import MAX_LAND_SNAP_KM, deduplicate_lon, download_era5_invariants, download_era5_months  # noqa: E402
from src.pipeline import (  # noqa: E402
	_download_bbox,
	_needed_year_months,
	backfill_era5_water_cells,
	filter_conaf_years,
	parse_year_range,
	run_pipeline,
	versioned_output_path,
)

app = typer.Typer(add_completion=False, help="Preprocesamiento CONAF + ERA5")

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
	datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)


def _download_only(start_year: int, end_year: int, era5_dir: Path) -> None:
	"""Descarga ERA5 acotado a los días con eventos CONAF, sin enriquecer.

	Carga y filtra CONAF por rango de años, deriva los meses/días con eventos
	y el bbox, e imprime un resumen antes de descargar los .nc mensuales y los
	invariantes ERA5 al directorio destino.

	Args:
		start_year: Primer año del rango (inclusive).
		end_year: Último año del rango (inclusive).
		era5_dir: Directorio destino donde se escriben los archivos ERA5.
	"""
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
	backfill: bool = typer.Option(False, "--backfill", help="Rellena (no destructivo) celdas ERA5 NaN sobre mar saltando a tierra; preserva label_l2/modis"),
	max_snap_km: float = typer.Option(MAX_LAND_SNAP_KM, "--max-snap-km", help="Tope del salto a tierra en km (default 6)"),
	no_download: bool = typer.Option(False, "--no-download", help="En backfill, no descargar ERA5 aunque falte algún mes"),
	fill_all: bool = typer.Option(False, "--fill-all", help="Con --backfill: re-extrae ERA5 de TODAS las filas con cobertura (tras dedup), no solo las NaN"),
	dedup: bool = typer.Option(False, "--dedup", help="Deduplica la grilla de longitud de los .nc en era5-dir (columnas fantasma del merge); no enriquece"),
):
	"""Ejecuta el pipeline CONAF → ERA5 → dataset enriquecido de punta a punta.

	Según los flags, despacha a uno de los modos mutuamente excluyentes: dedup
	de la grilla de longitud, backfill no destructivo de celdas ERA5 NaN sobre
	mar, descarga acotada de ERA5, o el pipeline completo con enriquecimiento y
	escritura del parquet versionado.

	Args:
		years: Rango de años en formato YYYY-YYYY.
		skip_download: Asume ERA5 ya descargado y omite la descarga.
		download_only: Solo descarga ERA5; no enriquece ni escribe parquet.
		skip_modis: Salta la descarga MODIS y el cálculo de label_l2.
		refresh_conaf: Re-descarga CONAF aunque exista cache.
		out: Ruta parquet opcional para el output versionado.
		era5_dir: Directorio ERA5 (default: data/raw/era5/).
		backfill: Rellena (no destructivo) celdas ERA5 NaN sobre mar saltando a
			tierra; preserva label_l2/modis.
		max_snap_km: Tope del salto a tierra en km.
		no_download: En backfill, no descarga ERA5 aunque falte algún mes.
		fill_all: Con backfill, re-extrae ERA5 de TODAS las filas con cobertura
			(tras dedup), no solo las NaN.
		dedup: Deduplica la grilla de longitud de los .nc en era5_dir (columnas
			fantasma del merge); no enriquece.

	Raises:
		typer.BadParameter: Si el rango de años es inválido, no hay .nc para
			dedup, o se combinan flags incompatibles.
		typer.Exit: Con código 1 si el pipeline falla en tiempo de ejecución.
	"""
	try:
		start_year, end_year = parse_year_range(years)
	except ValueError as e:
		raise typer.BadParameter(str(e)) from e

	if dedup:
		target_dir = era5_dir or ERA5_RAW_DIR
		backup = target_dir / "_backup_pre_dedup"
		files = sorted(target_dir.glob("*.nc"))
		if not files:
			raise typer.BadParameter(f"No hay .nc en {target_dir}")
		cleaned = 0
		for p in files:
			res = deduplicate_lon(p, backup_dir=backup)
			if not res.get("skipped"):
				cleaned += 1
				typer.echo(f"  {p.name}: {res['n_lon']}→{res['n_unique']} lon ({res['removed']} fantasma)")
		typer.secho(
			f"Dedup: {cleaned}/{len(files)} archivos limpiados (resto ya limpios). Backup: {backup}",
			fg=typer.colors.GREEN,
		)
		return

	if backfill:
		if download_only:
			raise typer.BadParameter("--backfill y --download-only son incompatibles")
		parquet = out or versioned_output_path(start_year, end_year)
		summary = backfill_era5_water_cells(
			parquet,
			era5_dir=era5_dir or ERA5_RAW_DIR,
			max_land_snap_km=max_snap_km,
			allow_download=not no_download,
			fill_all=fill_all,
		)
		typer.secho(
			f"Backfill listo: {summary['recovered']} filas recuperadas "
			f"({summary['land_snapped']} por salto a tierra) de {summary['candidates']} candidatas",
			fg=typer.colors.GREEN,
		)
		typer.echo(f"Parquet: {summary['parquet']} | backup: {summary['backup']}")
		typer.echo(f"era5_match_quality: {summary['era5_match_quality']}")
		return

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
