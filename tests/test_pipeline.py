from __future__ import annotations

import unittest

import pandas as pd

from src.pipeline import _needed_year_months


class PipelineTest(unittest.TestCase):
	def test_needed_year_months_uses_utc_timestamp_for_era5(self) -> None:
		df = pd.DataFrame(
			{
				"fecha_hora_inicio": [pd.Timestamp("2002-12-31 23:30")],
				"fecha_hora_inicio_utc": [pd.Timestamp("2003-01-01 03:30")],
			}
		)

		self.assertEqual(_needed_year_months(df), [(2003, 1, (1,))])


if __name__ == "__main__":
	unittest.main()
