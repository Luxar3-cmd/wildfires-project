from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.era5 import _normalize_lon, build_land_index, deduplicate_lon, extract_point


def _coastal_dataset() -> xr.Dataset:
	"""Grid 2x2 con una celda de mar (NaN) y tres de tierra, para probar el salto a tierra.

	La celda (-36.0, -72.955) es mar (NaN); la tierra más cercana está en (-36.0, -73.0),
	a ~4 km — recuperable con el tope por defecto (6 km), no con un tope de 1 km.
	"""
	t2m = np.array([[[np.nan, 285.0], [286.0, 287.0]]])  # (time, lat, lon)
	return xr.Dataset(
		{"t2m": (("time", "latitude", "longitude"), t2m)},
		coords={
			"time": pd.to_datetime(["2015-01-15 18:00"]),
			"latitude": [-36.0, -36.05],
			"longitude": [-72.955, -73.0],
		},
	)


class Era5ExtractorTest(unittest.TestCase):
	def test_extract_point_normalizes_timezone_aware_timestamp_to_utc(self) -> None:
		ds = xr.Dataset(
			{"t2m": (("time", "latitude", "longitude"), np.array([[[280.0]]]))},
			coords={
				"time": pd.to_datetime(["2002-07-06 01:00"]),
				"latitude": [-29.935555],
				"longitude": [-71.108333],
			},
		)

		result = extract_point(
			ds,
			-29.935555,
			-71.108333,
			pd.Timestamp("2002-07-05 21:05", tz="America/Santiago"),
		)

		self.assertEqual(result["t2m"], 280.0)
		self.assertEqual(result["era5_match_quality"], "good")
		self.assertEqual(result["era5_dt_hours"], 0.083)

	def test_land_snapping_recovers_sea_cell_within_cap(self) -> None:
		ds = _coastal_dataset()
		land_index = build_land_index(ds)
		result = extract_point(ds, -36.0, -72.955, pd.Timestamp("2015-01-15 18:00"), land_index=land_index)

		self.assertEqual(result["t2m"], 285.0)  # valor de la celda de tierra más cercana
		self.assertEqual(result["era5_match_quality"], "land_snapped")
		self.assertGreater(result["era5_land_snap_km"], 0.0)
		self.assertLessEqual(result["era5_land_snap_km"], 6.0)

	def test_land_snapping_skipped_beyond_cap_stays_water(self) -> None:
		ds = _coastal_dataset()
		land_index = build_land_index(ds)
		result = extract_point(
			ds, -36.0, -72.955, pd.Timestamp("2015-01-15 18:00"),
			land_index=land_index, max_snap_km=1.0,
		)

		self.assertIsNone(result["t2m"])
		self.assertEqual(result["era5_match_quality"], "water")
		self.assertEqual(result["era5_land_snap_km"], 0.0)

	def test_land_cell_unchanged_no_snap(self) -> None:
		ds = _coastal_dataset()
		land_index = build_land_index(ds)
		result = extract_point(ds, -36.0, -73.0, pd.Timestamp("2015-01-15 18:00"), land_index=land_index)

		self.assertEqual(result["t2m"], 285.0)
		self.assertEqual(result["era5_match_quality"], "good")
		self.assertEqual(result["era5_land_snap_km"], 0.0)


class Era5GridDedupTest(unittest.TestCase):
	def test_normalize_lon_prevents_merge_duplication(self) -> None:
		# Dos "lotes" con la misma grilla de longitud salvo ULP de float64, variables distintas.
		base = np.array([-74.0, -73.9, -73.8])
		a = xr.Dataset(
			{"t2m": (("latitude", "longitude"), np.ones((1, 3)))},
			coords={"latitude": [-36.0], "longitude": base},
		)
		b = xr.Dataset(
			{"d2m": (("latitude", "longitude"), np.full((1, 3), 2.0))},
			coords={"latitude": [-36.0], "longitude": base + np.array([0.0, 1e-13, -1e-13])},
		)
		merged = xr.merge([_normalize_lon(a), _normalize_lon(b)])
		self.assertEqual(merged.sizes["longitude"], 3)  # sin el fix daría hasta 6 (columnas fantasma)
		self.assertEqual(int(merged["t2m"].isnull().sum()), 0)
		self.assertEqual(int(merged["d2m"].isnull().sum()), 0)

	def test_deduplicate_lon_is_lossless_and_idempotent(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			tmp = Path(tmp)
			p = tmp / "dup.nc"
			# Gemela fantasma: lon -73.9 aparece 2x (real en idx1 + all-NaN en idx2).
			lon = np.array([-74.0, -73.9, -73.9, -73.8])
			t2m = np.array([[[280.0, 281.0, np.nan, 282.0]]])  # (time, lat, lon)
			xr.Dataset(
				{"t2m": (("valid_time", "latitude", "longitude"), t2m)},
				coords={
					"valid_time": pd.to_datetime(["2015-01-15 18:00"]),
					"latitude": [-36.0],
					"longitude": lon,
				},
			).to_netcdf(p)

			res = deduplicate_lon(p, backup_dir=tmp / "bk")
			self.assertFalse(res["skipped"])
			self.assertEqual(res["removed"], 1)

			with xr.open_dataset(p) as ds:
				self.assertEqual(ds.sizes["longitude"], 3)
				self.assertEqual(
					[round(float(x), 5) for x in ds["longitude"].values], [-74.0, -73.9, -73.8]
				)
				self.assertEqual(
					float(ds["t2m"].isel(valid_time=0, latitude=0, longitude=1).values), 281.0
				)  # quedó la columna real, no la fantasma
				self.assertEqual(int(ds["t2m"].isnull().sum()), 0)

			self.assertTrue((tmp / "bk" / "dup.nc").exists())  # backup no destructivo

			res2 = deduplicate_lon(p, backup_dir=tmp / "bk2")  # 2ª pasada: ya limpio
			self.assertTrue(res2["skipped"])


if __name__ == "__main__":
	unittest.main()
