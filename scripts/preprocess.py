"""CLI de preprocesamiento: descarga CONAF + ERA5 y produce dataset enriquecido.

Uso:
    python scripts/preprocess.py --years 2002-2020
    python scripts/preprocess.py --years 2019-2019 --skip-download
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

from src.pipeline import parse_year_range, run_pipeline  # noqa: E402

app = typer.Typer(add_completion=False, help="Preprocesamiento CONAF + ERA5")

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
	datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)


@app.command()
def main(
	years: str = typer.Option("2002-2020", "--years", help="Rango YYYY-YYYY"),
	skip_download: bool = typer.Option(False, "--skip-download", help="Asume ERA5 ya descargado"),
	refresh_conaf: bool = typer.Option(False, "--refresh-conaf", help="Re-descarga CONAF aunque haya cache"),
	out: Path | None = typer.Option(None, "--out", help="Ruta parquet opcional para el output versionado"),
):
	"""Pipeline completo: CONAF → ERA5 → dataset enriquecido."""
	try:
		start_year, end_year = parse_year_range(years)
	except ValueError as e:
		raise typer.BadParameter(str(e)) from e

	def report(stage: str, message: str, level: str, data: dict | None) -> None:
		color = typer.colors.YELLOW if level == "warning" else typer.colors.CYAN
		typer.secho(f"==> [{stage}] {message}", fg=color)

	try:
		summary = run_pipeline(
			start_year,
			end_year,
			skip_download=skip_download,
			refresh_conaf=refresh_conaf,
			out_path=out,
			reporter=report,
		)
	except RuntimeError as e:
		typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
		raise typer.Exit(code=1) from e

	output = summary["output"]
	typer.secho(
		f"==> Listo: {output['rows']} filas, {output['columns']} columnas",
		fg=typer.colors.GREEN,
	)
	typer.echo(f"Output versionado: {output['versioned_output']}")
	typer.echo(f"Output latest: {output['latest_output']}")
	typer.echo(f"Informe de features: {output['features_report_path']}")


if __name__ == "__main__":
	app()
