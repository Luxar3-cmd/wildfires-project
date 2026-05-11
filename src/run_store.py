"""Persistencia simple de runs del pipeline."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import DATA_DIR

RUNS_DIR = DATA_DIR / "runs"


def utc_now() -> str:
	return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
	if isinstance(value, Path):
		return str(value)
	if hasattr(value, "item"):
		return value.item()
	return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


def read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
	if not path.exists():
		return default or {}
	return json.loads(path.read_text())


class RunStore:
	def __init__(self, base_dir: Path = RUNS_DIR):
		self.base_dir = base_dir
		self.base_dir.mkdir(parents=True, exist_ok=True)

	def create(self, params: dict[str, Any]) -> dict[str, Any]:
		run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
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
			f.write(json.dumps(event, ensure_ascii=False, default=_json_default) + "\n")
		self.update(run_id, stage=stage)
		return event

	def status(self, run_id: str) -> dict[str, Any]:
		return read_json(self.base_dir / run_id / "status.json")

	def events(self, run_id: str, limit: int = 300) -> list[dict[str, Any]]:
		path = self.base_dir / run_id / "events.jsonl"
		if not path.exists():
			return []
		lines = path.read_text().splitlines()
		return [json.loads(line) for line in lines[-limit:] if line.strip()]

	def list(self) -> list[dict[str, Any]]:
		runs = []
		for path in sorted(self.base_dir.glob("*/status.json"), reverse=True):
			runs.append(read_json(path))
		return runs

