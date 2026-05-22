"""Persistencia simple de runs del pipeline.

Cada run se almacena en data/runs/<run_id>/ con dos archivos:
  - status.json   : estado actual del run (queued/running/done/error) + parámetros + summary.
  - events.jsonl  : log de eventos en formato JSON Lines, uno por línea.

JSONL permite appends atómicos sin parsear el archivo completo —
es más robusto que reescribir un JSON grande en cada evento.
"""
from __future__ import annotations

import json
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import DATA_DIR
from src.utils import _json_safe_default

RUNS_DIR = DATA_DIR / "runs"


def utc_now() -> str:
	return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_safe_default))


def read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
	if not path.exists():
		return default or {}
	return json.loads(path.read_text())


class RunStore:
	def __init__(self, base_dir: Path = RUNS_DIR):
		self.base_dir = base_dir
		self.base_dir.mkdir(parents=True, exist_ok=True)

	def create(self, params: dict[str, Any]) -> dict[str, Any]:
		# run_id combina timestamp UTC (legibilidad / ordenamiento cronológico)
		# con un sufijo UUID corto (unicidad en caso de runs simultáneos)
		run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
		run_dir = self.base_dir / run_id
		status = {
			"run_id": run_id,
			"status": "queued",
			"stage": "queued",
			"params": params,
			"created_at": utc_now(),
			"started_at": None,
			"finished_at": None,
			"error": None,
			"summary": {},
		}
		write_json(run_dir / "status.json", status)
		(run_dir / "events.jsonl").touch()
		return status

	def update(self, run_id: str, **changes: Any) -> dict[str, Any]:
		path = self.base_dir / run_id / "status.json"
		status = read_json(path)
		status.update(changes)
		write_json(path, status)
		return status

	def event(
		self,
		run_id: str,
		stage: str,
		message: str,
		level: str = "info",
		data: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		event = {
			"ts": utc_now(),
			"stage": stage,
			"level": level,
			"message": message,
			"data": data or {},
		}
		path = self.base_dir / run_id / "events.jsonl"
		path.parent.mkdir(parents=True, exist_ok=True)
		with path.open("a", encoding="utf-8") as f:
			f.write(json.dumps(event, ensure_ascii=False, default=_json_safe_default) + "\n")
		self.update(run_id, stage=stage)
		return event

	def status(self, run_id: str) -> dict[str, Any]:
		return read_json(self.base_dir / run_id / "status.json")

	def events(self, run_id: str, limit: int = 300) -> list[dict[str, Any]]:
		path = self.base_dir / run_id / "events.jsonl"
		if not path.exists():
			return []
		# deque(maxlen=limit) acumula solo las últimas N líneas sin leer el archivo completo
		last_lines = deque(maxlen=limit)
		with path.open(encoding="utf-8") as f:
			for line in f:
				if line.strip():
					last_lines.append(line)
		return [json.loads(line) for line in last_lines]

	def list(self) -> list[dict[str, Any]]:
		runs = []
		for path in sorted(self.base_dir.glob("*/status.json"), reverse=True):
			runs.append(read_json(path))
		return runs
