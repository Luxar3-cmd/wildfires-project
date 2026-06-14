from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import xarray as xr

from src.era5 import era5_month_path
from src.pipeline import _needed_year_months, backfill_era5_water_cells


class PipelineTest(unittest.TestCase):
	def test_needed_year_months_uses_utc_timestamp_for_era5(self) -> None:
		df = pd.DataFrame(
			{
				"fecha_hora_inicio": [pd.Timestamp("2002-12-31 23:30")],
				"fecha_hora_inicio_utc": [pd.Timestamp("2003-01-01 03:30")],
			}
		)

		self.assertEqual(_needed_year_months(df), [(2003, 1, (1,))])


class BackfillTest(unittest.TestCase):
	def _write_netcdf(self, era5_dir: Path) -> None:
		# Grid costero: (-36.0,-72.955) es mar (NaN); tierra más cercana en (-36.0,-73.0) a ~4 km.
		t2m = np.array([[[np.nan, 285.0], [286.0, 287.0]]])
		ds = xr.Dataset(
			{"t2m": (("time", "latitude", "longitude"), t2m)},
			coords={
				"time": pd.to_datetime(["2015-01-15 18:00"]),
				"latitude": [-36.0, -36.05],
				"longitude": [-72.955, -73.0],
			},
		)
		ds.to_netcdf(era5_month_path(2015, 1, era5_dir))

	def _make_parquet(self, path: Path) -> None:
		df = pd.DataFrame(
			{
				"latitud": [-36.0, -36.05, -50.0],
				"longitud": [-72.955, -73.0, -72.0],
				"fecha_hora_inicio_utc": [pd.Timestamp("2015-01-15 18:00")] * 3,
				"t2m": [np.nan, 287.0, np.nan],
				"era5_match_quality": ["good", "good", "out_of_coverage"],
				"label_l2": [1, 0, 0],
				"modis_frp_max_mw": [12000.0, np.nan, np.nan],
			}
		)
		df.to_parquet(path)

	def test_backfill_recovers_then_is_idempotent_and_nondestructive(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			tmp_path = Path(tmp)
			era5_dir = tmp_path / "era5"
			era5_dir.mkdir()
			self._write_netcdf(era5_dir)
			parquet = tmp_path / "enriched.parquet"
			self._make_parquet(parquet)

			# Evita tocar data/processed real (backup + copia latest)
			with mock.patch("src.pipeline.DATA_PROCESSED", tmp_path), \
				mock.patch("src.pipeline.LATEST_PARQUET", tmp_path / "latest.parquet"):
				s1 = backfill_era5_water_cells(parquet, era5_dir=era5_dir, allow_download=False)
				df1 = pd.read_parquet(parquet)
				s2 = backfill_era5_water_cells(parquet, era5_dir=era5_dir, allow_download=False)
				df2 = pd.read_parquet(parquet)

		# 1ª corrida: recupera la fila de mar saltando a tierra
		self.assertEqual(s1["candidates"], 1)
		self.assertEqual(s1["recovered"], 1)
		self.assertEqual(s1["land_snapped"], 1)
		self.assertEqual(df1.loc[0, "t2m"], 285.0)
		self.assertEqual(df1.loc[0, "era5_match_quality"], "land_snapped")
		# No destructivo: label_l2 / modis intactos; out_of_coverage no se toca
		self.assertEqual(list(df1["label_l2"]), [1, 0, 0])
		self.assertEqual(df1.loc[1, "t2m"], 287.0)
		self.assertTrue(pd.isna(df1.loc[2, "t2m"]))
		self.assertEqual(df1.loc[2, "era5_match_quality"], "out_of_coverage")
		# 2ª corrida: idempotente (nada que rellenar, parquet sin cambios)
		self.assertEqual(s2["candidates"], 0)
		pd.testing.assert_frame_equal(df1, df2)


if __name__ == "__main__":
	unittest.main()
