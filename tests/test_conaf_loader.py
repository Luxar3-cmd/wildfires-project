from __future__ import annotations

import unittest

import geopandas as gpd
import pandas as pd

from src.conaf_loader import _clean, _looks_like_lost_time_component


class ConafLoaderTest(unittest.TestCase):
	def test_clean_combines_fecha_and_string_dtype_hora(self) -> None:
		gdf = gpd.GeoDataFrame(
			{
				"Fecha": ["2002-07-05", "2002-10-25"],
				"Hora inicio": pd.Series(["21:05", "11:50"], dtype="string"),
				"Latitud": [-29.935555, -29.933611],
				"Longitud": [-71.108333, -71.211944],
			},
			geometry=gpd.points_from_xy([-71.108333, -71.211944], [-29.935555, -29.933611]),
			crs="EPSG:4326",
		)

		clean = _clean(gdf)

		self.assertEqual(clean.loc[0, "fecha_hora_inicio"], pd.Timestamp("2002-07-05 21:05"))
		self.assertEqual(clean.loc[1, "fecha_hora_inicio"], pd.Timestamp("2002-10-25 11:50"))
		self.assertEqual(clean.loc[0, "fecha_hora_inicio_utc"], pd.Timestamp("2002-07-06 01:05"))

	def test_detects_cached_timestamps_with_lost_time_component(self) -> None:
		gdf = pd.DataFrame(
			{
				"hora_inicio": ["21:05", "11:50"],
				"fecha_hora_inicio": pd.to_datetime(["2002-07-05", "2002-10-25"]),
			}
		)

		self.assertTrue(_looks_like_lost_time_component(gdf))


if __name__ == "__main__":
	unittest.main()
