from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import xarray as xr

from src.era5 import extract_point


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


if __name__ == "__main__":
	unittest.main()
