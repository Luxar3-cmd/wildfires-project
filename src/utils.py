"""Utilidades compartidas del proyecto."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd


def _json_safe_default(obj: Any) -> Any:
	"""Handler para json.dumps(default=...) — cubre tipos no estándar comunes.

	Convierte: Path → str, numpy scalars → Python nativo, datetime → isoformat,
	pd.NaN / float NaN / inf → None. Todo lo demás cae a str().
	"""
	if isinstance(obj, Path):
		return str(obj)
	try:
		if pd.isna(obj):
			return None
	except (TypeError, ValueError):
		pass
	if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
		return None
	if hasattr(obj, "isoformat"):
		return obj.isoformat()
	if hasattr(obj, "item"):
		# numpy scalar (int64, float32, etc.) → Python nativo
		return obj.item()
	return str(obj)


__all__ = ["_json_safe_default"]
