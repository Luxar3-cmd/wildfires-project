from __future__ import annotations

import math
import unittest
from datetime import date

import numpy as np
import pandas as pd

from src.modis import (
	FLI_EWE_THRESHOLD_KW_M,
	MODIS_PIXEL_LENGTH_M,
	_group_into_blocks,
	frp_to_fli,
	label_l2,
	match_modis_to_conaf,
)


class FrpToFliTest(unittest.TestCase):
	def test_wooster_threshold_value(self) -> None:
		# FRP=1700 MW, front=1000 m, η_r=0.17 → FLI = 1700/0.17 = 10.000 kW/m (umbral EWE exacto)
		self.assertAlmostEqual(frp_to_fli(1700, 1000, 0.17), 10000.0, places=6)

	def test_wooster_typical_value(self) -> None:
		# FRP=170 MW, front=1000 m, η_r=0.17 → 1000 kW/m
		self.assertAlmostEqual(frp_to_fli(170, 1000, 0.17), 1000.0, places=6)

	def test_returns_nan_on_invalid_frp(self) -> None:
		self.assertTrue(math.isnan(frp_to_fli(float("nan"), 1000, 0.17)))
		self.assertTrue(math.isnan(frp_to_fli(0, 1000, 0.17)))
		self.assertTrue(math.isnan(frp_to_fli(-5, 1000, 0.17)))


class GroupIntoBlocksTest(unittest.TestCase):
	def test_groups_consecutive_into_blocks_of_five(self) -> None:
		days = [date(2017, 1, 1), date(2017, 1, 2), date(2017, 1, 3),
		        date(2017, 1, 8), date(2017, 1, 9)]
		blocks = _group_into_blocks(days)
		# [1,2,3] → (1, 3); [8,9] → (8, 2)
		self.assertEqual(blocks, [(date(2017, 1, 1), 3), (date(2017, 1, 8), 2)])

	def test_long_run_splits_at_five_days(self) -> None:
		days = [date(2017, 1, d) for d in range(1, 8)]  # 7 días consecutivos
		blocks = _group_into_blocks(days)
		# Primer bloque cubre días 1-5 (range=5), segundo días 6-7 (range=2)
		self.assertEqual(blocks, [(date(2017, 1, 1), 5), (date(2017, 1, 6), 2)])


class MatchModisToConafTest(unittest.TestCase):
	def _conaf(self) -> pd.DataFrame:
		return pd.DataFrame({
			"latitud": [-35.0],
			"longitud": [-72.0],
			"fecha_hora_inicio_utc": [pd.Timestamp("2017-01-26 12:00")],
		})

	def test_matches_only_within_radius_and_time(self) -> None:
		# 3 detecciones MODIS:
		#   A: ~2km, +6h  → dentro (lat shift 0.018° ≈ 2km)
		#   B: ~50km, +1h → fuera por distancia
		#   C: ~2km, +48h → fuera por tiempo
		modis = pd.DataFrame({
			"latitude":  [-35.018, -35.45, -35.018],
			"longitude": [-72.0,   -72.0,  -72.0],
			"frp":       [320.0,   999.0,  500.0],
			"acq_datetime_utc": [
				pd.Timestamp("2017-01-26 18:00"),
				pd.Timestamp("2017-01-26 13:00"),
				pd.Timestamp("2017-01-28 12:00"),
			],
		})
		out = match_modis_to_conaf(self._conaf(), modis, radius_km=5.0, time_window_h=24.0)
		self.assertEqual(int(out["modis_n_matches"].iloc[0]), 1)
		self.assertAlmostEqual(out["modis_frp_max_mw"].iloc[0], 320.0, places=3)

	def test_empty_modis_returns_zero_matches(self) -> None:
		out = match_modis_to_conaf(self._conaf(), pd.DataFrame())
		self.assertEqual(int(out["modis_n_matches"].iloc[0]), 0)
		self.assertTrue(math.isnan(out["modis_frp_max_mw"].iloc[0]))


class LabelL2Test(unittest.TestCase):
	def test_label_threshold_is_1700_mw_peak_local(self) -> None:
		# Peak-local: FLI = frp·1000/0.17/1000. Umbral EWE ⟺ frp ≳ 1700 MW.
		# Superficie >= 50 ha para pasar la guardia de coherencia.
		enriched = pd.DataFrame({"superficie_quemada_total_ha": [100.0, 100.0, 100.0]})
		matches = pd.DataFrame({
			"modis_n_matches": [4, 2, 1],
			"modis_frp_max_mw": [2000.0, 1800.0, 170.0],
			"modis_frp_sum_mw": [2000.0, 1800.0, 170.0],
		})
		out = label_l2(enriched, matches)
		self.assertEqual(out["label_l2"].tolist(), [1, 1, 0])
		self.assertGreaterEqual(out["fli_estimado_kw_m"].iloc[1], FLI_EWE_THRESHOLD_KW_M)
		# El área CONAF ya no entra en la FLI (peak-local usa L_f fijo = pixel MODIS).
		self.assertNotIn("front_length_m", out.columns)

	def test_area_guard_blocks_small_fire_with_high_frp(self) -> None:
		# FRP alto (FLI > umbral) pero superficie < 50 ha → falso positivo por atribución
		# espacial, debe quedar en label_l2 = 0.
		enriched = pd.DataFrame({"superficie_quemada_total_ha": [0.01, 0.24, 200.0]})
		matches = pd.DataFrame({
			"modis_n_matches": [80, 40, 28],
			"modis_frp_max_mw": [2940.0, 2230.0, 2906.0],
			"modis_frp_sum_mw": [2940.0, 2230.0, 2906.0],
		})
		out = label_l2(enriched, matches)
		# Los dos primeros superan el umbral de FLI pero no la guardia de área; solo el tercero pasa.
		self.assertTrue((out["fli_estimado_kw_m"] >= FLI_EWE_THRESHOLD_KW_M).all())
		self.assertEqual(out["label_l2"].tolist(), [0, 0, 1])

	def test_label_zero_without_match(self) -> None:
		enriched = pd.DataFrame({"superficie_quemada_total_ha": [5000.0]})
		matches = pd.DataFrame({
			"modis_n_matches": [0],
			"modis_frp_max_mw": [np.nan],
			"modis_frp_sum_mw": [np.nan],
		})
		out = label_l2(enriched, matches)
		self.assertEqual(out["label_l2"].iloc[0], 0)
		self.assertTrue(math.isnan(out["fli_estimado_kw_m"].iloc[0]))


if __name__ == "__main__":
	unittest.main()
