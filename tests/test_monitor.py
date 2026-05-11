from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.monitor import latest_output_status


class MonitorLatestOutputTest(unittest.TestCase):
	def test_latest_output_status_reads_parquet_metadata_and_quality(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			path = Path(tmp) / "conaf_enriched_latest.parquet"
			pd.DataFrame(
				{
					"era5_match_quality": ["good", "missing", "good", None],
					"value": [1, 2, 3, 4],
				}
			).to_parquet(path)
			path.with_suffix(".attribution.json").write_text(
				json.dumps({"run": {"params": {"out_path": "/tmp/versioned.parquet", "start_year": 2002}}}),
				encoding="utf-8",
			)
			(path.parent / "features_report.md").write_text("# reporte", encoding="utf-8")
			(path.parent / "features_report.json").write_text("{}", encoding="utf-8")

			status = latest_output_status(path)

			self.assertTrue(status["exists"])
			self.assertEqual(status["status"], "available")
			self.assertEqual(status["rows"], 4)
			self.assertEqual(status["columns"], 2)
			self.assertEqual(
				status["era5_match_quality"],
				{"good": 2, "missing": 1, "null": 1},
			)
			self.assertEqual(status["latest_output"], str(path))
			self.assertEqual(status["versioned_output"], "/tmp/versioned.parquet")
			self.assertEqual(status["params"]["start_year"], 2002)
			self.assertEqual(status["features_report_path"], str(path.parent / "features_report.md"))

	def test_latest_output_status_handles_missing_file(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			path = Path(tmp) / "conaf_enriched_latest.parquet"

			status = latest_output_status(path)

			self.assertFalse(status["exists"])
			self.assertEqual(status["status"], "missing")
			self.assertEqual(status["latest_output"], str(path))
			self.assertIsNone(status["rows"])
			self.assertEqual(status["era5_match_quality"], {})


if __name__ == "__main__":
	unittest.main()
