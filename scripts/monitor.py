"""Dashboard local para lanzar y monitorear el pipeline CONAF + ERA5."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from src.pipeline import LATEST_PARQUET, parse_year_range, run_pipeline  # noqa: E402
from src.run_store import RunStore, utc_now  # noqa: E402

app = FastAPI(title="XAI Pipeline Monitor")
store = RunStore()
executor = ThreadPoolExecutor(max_workers=1)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
_latest_output_cache: dict[str, Any] = {"key": None, "payload": None}


class RunRequest(BaseModel):
	years: str = "2002-2020"
	skip_download: bool = True
	refresh_conaf: bool = False


def _active_run() -> dict[str, Any] | None:
	for run in store.list():
		if run.get("status") in {"queued", "running"}:
			return run
	return None


def _without_traceback(payload: dict[str, Any]) -> dict[str, Any]:
	public = payload
	error = payload.get("error")
	if isinstance(error, dict) and "traceback" in error:
		public = dict(public)
		public["error"] = {key: value for key, value in error.items() if key != "traceback"}

	data = payload.get("data")
	if isinstance(data, dict) and "traceback" in data:
		public = dict(public)
		public["data"] = {key: value for key, value in data.items() if key != "traceback"}
	return public


def _path_if_exists(path: Path) -> str | None:
	return str(path) if path.exists() else None


def _read_latest_params(path: Path) -> dict[str, Any]:
	if not path.exists():
		return {}
	try:
		payload = json.loads(path.read_text(encoding="utf-8"))
	except json.JSONDecodeError:
		logger.warning("Attribution inválido: %s", path)
		return {}
	params = payload.get("run", {}).get("params", {})
	return params if isinstance(params, dict) else {}


def _quality_counts(path: Path, column: str) -> dict[str, int]:
	values = pq.read_table(path, columns=[column]).column(column).to_pandas()
	counts = values.value_counts(dropna=False)
	return {
		("null" if pd.isna(key) else str(key)): int(value)
		for key, value in counts.items()
	}


def _latest_cache_key(path: Path) -> tuple[str, int | None, int | None]:
	if not path.exists():
		return (str(path), None, None)
	stat = path.stat()
	return (str(path), stat.st_mtime_ns, stat.st_size)


def latest_output_status(latest_path: Path = LATEST_PARQUET) -> dict[str, Any]:
	key = _latest_cache_key(latest_path)
	if _latest_output_cache["key"] == key:
		return dict(_latest_output_cache["payload"])

	attribution_path = latest_path.with_suffix(".attribution.json")
	features_report_path = latest_path.parent / "features_report.md"
	features_report_json_path = latest_path.parent / "features_report.json"
	params = _read_latest_params(attribution_path)
	base_payload: dict[str, Any] = {
		"exists": latest_path.exists(),
		"latest_output": str(latest_path),
		"versioned_output": params.get("out_path"),
		"attribution_path": _path_if_exists(attribution_path),
		"features_report_path": _path_if_exists(features_report_path),
		"features_report_json_path": _path_if_exists(features_report_json_path),
		"params": params,
	}
	if not latest_path.exists():
		payload = {
			**base_payload,
			"status": "missing",
			"stage": "latest_output",
			"modified_at": None,
			"rows": None,
			"columns": None,
			"era5_match_quality": {},
		}
		_latest_output_cache.update({"key": key, "payload": payload})
		return dict(payload)

	try:
		parquet = pq.ParquetFile(latest_path)
		columns = parquet.schema_arrow.names
		stat = latest_path.stat()
		payload = {
			**base_payload,
			"status": "available",
			"stage": "latest_output",
			"modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
			"rows": int(parquet.metadata.num_rows),
			"columns": int(len(columns)),
			"era5_match_quality": (
				_quality_counts(latest_path, "era5_match_quality")
				if "era5_match_quality" in columns
				else {}
			),
		}
	except Exception as e:
		logger.exception("No se pudo leer latest output: %s", latest_path)
		payload = {
			**base_payload,
			"status": "error",
			"stage": "latest_output",
			"modified_at": datetime.fromtimestamp(latest_path.stat().st_mtime, timezone.utc).isoformat(),
			"rows": None,
			"columns": None,
			"era5_match_quality": {},
			"error": {"type": type(e).__name__, "message": str(e)},
		}

	_latest_output_cache.update({"key": key, "payload": payload})
	return dict(payload)


def _execute_run(run_id: str, request: RunRequest) -> None:
	def report(stage: str, message: str, level: str, data: dict[str, Any] | None) -> None:
		store.event(run_id, stage=stage, message=message, level=level, data=data)

	try:
		start_year, end_year = parse_year_range(request.years)
		store.update(run_id, status="running", started_at=utc_now(), stage="starting")
		report("starting", "Run iniciado", "info", request.model_dump())
		summary = run_pipeline(
			start_year,
			end_year,
			skip_download=request.skip_download,
			refresh_conaf=request.refresh_conaf,
			reporter=report,
		)
		store.update(
			run_id,
			status="completed",
			stage="completed",
			finished_at=utc_now(),
			summary=summary,
		)
		report("completed", "Run completado", "info", summary.get("output", {}))
	except Exception as e:
		logger.exception("Run %s falló", run_id)
		error = {"type": type(e).__name__, "message": str(e)}
		store.update(run_id, status="failed", stage="failed", finished_at=utc_now(), error=error)
		report("failed", str(e), "error", error)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
	return HTML


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
	return [_without_traceback(run) for run in store.list()]


@app.get("/api/latest-output")
def get_latest_output() -> dict[str, Any]:
	return _without_traceback(latest_output_status())


@app.post("/api/runs")
def create_run(request: RunRequest) -> dict[str, Any]:
	try:
		parse_year_range(request.years)
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e)) from e

	active = _active_run()
	if active:
		raise HTTPException(status_code=409, detail=f"Ya hay un run activo: {active['run_id']}")

	run = store.create(request.model_dump())
	executor.submit(_execute_run, run["run_id"], request)
	return run


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
	status = store.status(run_id)
	if not status:
		raise HTTPException(status_code=404, detail="Run no encontrado")
	return _without_traceback(status)


@app.get("/api/runs/{run_id}/events")
def get_events(run_id: str, limit: int = 300) -> list[dict[str, Any]]:
	if not store.status(run_id):
		raise HTTPException(status_code=404, detail="Run no encontrado")
	return [_without_traceback(event) for event in store.events(run_id, limit=limit)]


HTML = r"""<!doctype html>
<html lang="es">
<head>
	<meta charset="utf-8">
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<title>XAI Pipeline Monitor</title>
	<style>
		:root {
			--bg: #f2f0ea;
			--ink: #20231f;
			--muted: #6c7169;
			--line: #d6d1c4;
			--panel: #fbfaf6;
			--accent: #0f6b5f;
			--accent-2: #b6462f;
			--warn: #9b6a00;
			--error: #a83232;
			--ok: #1f7650;
			--shadow: 0 18px 60px rgba(32, 35, 31, 0.08);
		}
		* { box-sizing: border-box; }
		body {
			margin: 0;
			background: var(--bg);
			color: var(--ink);
			font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
			letter-spacing: 0;
		}
		button, input, label { font: inherit; }
		.shell {
			width: min(1480px, calc(100vw - 32px));
			margin: 0 auto;
			padding: 24px 0 32px;
		}
		header {
			display: grid;
			grid-template-columns: 1fr auto;
			gap: 24px;
			align-items: end;
			border-bottom: 1px solid var(--line);
			padding-bottom: 18px;
		}
		h1 {
			margin: 0;
			font-size: clamp(28px, 4vw, 54px);
			line-height: 0.95;
			font-weight: 760;
		}
		.subhead {
			margin-top: 10px;
			color: var(--muted);
			max-width: 720px;
		}
		.grid {
			display: grid;
			grid-template-columns: minmax(310px, 380px) 1fr;
			gap: 18px;
			margin-top: 18px;
		}
		.panel {
			background: var(--panel);
			border: 1px solid var(--line);
			border-radius: 8px;
			box-shadow: var(--shadow);
		}
		.panel.pad { padding: 18px; }
		.stack { display: grid; gap: 14px; }
		.label {
			display: block;
			font-size: 12px;
			font-weight: 720;
			text-transform: uppercase;
			color: var(--muted);
			margin-bottom: 7px;
		}
		input[type="text"] {
			width: 100%;
			border: 1px solid var(--line);
			background: #fffdf8;
			color: var(--ink);
			border-radius: 6px;
			padding: 11px 12px;
			outline: none;
		}
		input[type="text"]:focus {
			border-color: var(--accent);
			box-shadow: 0 0 0 3px rgba(15, 107, 95, 0.16);
		}
		.check {
			display: flex;
			align-items: center;
			gap: 9px;
			color: var(--ink);
		}
		.actions {
			display: flex;
			gap: 10px;
			align-items: center;
			flex-wrap: wrap;
		}
		button {
			border: 0;
			border-radius: 6px;
			padding: 11px 14px;
			background: var(--accent);
			color: white;
			font-weight: 720;
			cursor: pointer;
		}
		button.secondary {
			background: transparent;
			color: var(--accent);
			border: 1px solid color-mix(in srgb, var(--accent), var(--line) 55%);
		}
		button.secondary.active { background: #eef6f1; }
		button:disabled { opacity: 0.55; cursor: not-allowed; }
		.status-strip {
			display: grid;
			grid-template-columns: repeat(4, 1fr);
			border-bottom: 1px solid var(--line);
		}
		.metric {
			padding: 16px 18px;
			border-right: 1px solid var(--line);
			min-width: 0;
		}
		.metric:last-child { border-right: 0; }
		.metric span {
			display: block;
			color: var(--muted);
			font-size: 12px;
			font-weight: 720;
			text-transform: uppercase;
		}
		.metric strong {
			display: block;
			margin-top: 4px;
			font-size: 22px;
			white-space: nowrap;
			overflow: hidden;
			text-overflow: ellipsis;
		}
		.content-grid {
			display: grid;
			grid-template-columns: 1.1fr 0.9fr;
			gap: 18px;
			padding: 18px;
		}
		h2 {
			margin: 0 0 12px;
			font-size: 15px;
			text-transform: uppercase;
			color: var(--muted);
			letter-spacing: 0;
		}
		.timeline {
			display: grid;
			gap: 10px;
		}
		.event {
			display: grid;
			grid-template-columns: 96px 1fr;
			gap: 12px;
			padding: 10px 0;
			border-bottom: 1px solid var(--line);
		}
		.event:last-child { border-bottom: 0; }
		.event time { color: var(--muted); font-size: 12px; }
		.badge {
			display: inline-flex;
			align-items: center;
			border-radius: 999px;
			padding: 3px 8px;
			font-size: 12px;
			font-weight: 720;
			background: #ece7db;
			color: var(--muted);
			margin-right: 7px;
		}
		.badge.warning { background: #f2dfad; color: var(--warn); }
		.badge.error { background: #f4c8c5; color: var(--error); }
		.badge.completed { background: #cfe6da; color: var(--ok); }
		.inventory {
			display: grid;
			grid-template-columns: repeat(auto-fill, minmax(76px, 1fr));
			gap: 8px;
		}
		.year {
			border: 1px solid var(--line);
			border-radius: 6px;
			padding: 9px;
			background: #fffdf8;
		}
		.year strong { display: block; }
		.year small { color: var(--muted); }
		.year.missing { border-color: #e6b8a9; background: #fff3ee; }
		.logbox {
			margin-top: 18px;
			background: #24261f;
			color: #eee8dc;
			border-radius: 8px;
			padding: 14px;
			height: 260px;
			overflow: auto;
			font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
			font-size: 12px;
			line-height: 1.55;
		}
		.run-list {
			display: grid;
			gap: 8px;
			max-height: 360px;
			overflow: auto;
		}
		.run-item {
			width: 100%;
			text-align: left;
			background: transparent;
			color: var(--ink);
			border: 1px solid var(--line);
			border-radius: 6px;
			padding: 10px;
		}
		.run-item.active { border-color: var(--accent); background: #eef6f1; }
		.empty {
			color: var(--muted);
			border: 1px dashed var(--line);
			border-radius: 6px;
			padding: 18px;
		}
		.pathbox {
			color: var(--muted);
			word-break: break-all;
			font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
			font-size: 12px;
			line-height: 1.5;
			background: #fffdf8;
			border: 1px solid var(--line);
			border-radius: 6px;
			padding: 10px;
		}
		@media (max-width: 980px) {
			header, .grid, .content-grid, .status-strip { grid-template-columns: 1fr; }
			.metric { border-right: 0; border-bottom: 1px solid var(--line); }
			.metric:last-child { border-bottom: 0; }
		}
	</style>
</head>
<body>
	<div class="shell">
		<header>
			<div>
				<h1>XAI Pipeline Monitor</h1>
				<div class="subhead">Control local para correr CONAF + ERA5 y ver estado, gaps, eventos y outputs sin leer logs crudos.</div>
			</div>
			<div id="clock" class="subhead"></div>
		</header>

		<div class="grid">
			<aside class="panel pad stack">
				<section>
					<h2>Nuevo run</h2>
					<label class="label" for="years">Rango de años</label>
					<input id="years" type="text" value="2002-2002" autocomplete="off">
				</section>
				<label class="check"><input id="skipDownload" type="checkbox" checked> Saltar descarga ERA5</label>
				<label class="check"><input id="refreshConaf" type="checkbox"> Re-descargar CONAF</label>
				<div class="actions">
					<button id="startBtn">Iniciar</button>
					<button class="secondary" id="refreshBtn">Actualizar</button>
					<button class="secondary active" id="latestBtn">Último output</button>
				</div>
				<section>
					<h2>Runs</h2>
					<div id="runs" class="run-list"></div>
				</section>
			</aside>

			<main class="panel">
				<div class="status-strip">
					<div class="metric"><span>Status</span><strong id="status">Sin run</strong></div>
					<div class="metric"><span>Etapa</span><strong id="stage">-</strong></div>
					<div class="metric"><span>Filas output</span><strong id="rows">-</strong></div>
					<div class="metric"><span>ERA5 missing</span><strong id="missing">-</strong></div>
				</div>
				<div class="content-grid">
					<section>
						<h2>Timeline</h2>
						<div id="timeline" class="timeline empty">No hay eventos todavía.</div>
						<div id="logs" class="logbox"></div>
					</section>
					<section>
						<h2>Inventario solicitado</h2>
						<div id="inventory" class="inventory"></div>
						<div style="height:18px"></div>
						<h2>Calidad ERA5</h2>
						<div id="quality" class="empty">Sin output.</div>
						<div style="height:18px"></div>
						<h2>Output</h2>
						<div id="outputPath" class="pathbox">-</div>
					</section>
				</div>
			</main>
		</div>
	</div>
	<script>
		let activeRunId = null;
		let pollTimer = null;

		const $ = (id) => document.getElementById(id);
		const fmt = (ts) => ts ? new Date(ts).toLocaleTimeString('es-CL', {hour: '2-digit', minute: '2-digit', second: '2-digit'}) : '-';
		const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (ch) => ({
			'&': '&amp;',
			'<': '&lt;',
			'>': '&gt;',
			'"': '&quot;',
			"'": '&#39;'
		})[ch]);

		function statusBadge(status) {
			const cls = status === 'completed' ? 'completed' : status === 'failed' ? 'error' : status === 'running' ? 'warning' : '';
			return `<span class="badge ${cls}">${esc(status || '-')}</span>`;
		}

		async function api(path, options) {
			const res = await fetch(path, options);
			if (!res.ok) {
				const body = await res.json().catch(() => ({}));
				throw new Error(body.detail || res.statusText);
			}
			return res.json();
		}

		async function startRun() {
			$('startBtn').disabled = true;
			try {
				const run = await api('/api/runs', {
					method: 'POST',
					headers: {'Content-Type': 'application/json'},
					body: JSON.stringify({
						years: $('years').value,
						skip_download: $('skipDownload').checked,
						refresh_conaf: $('refreshConaf').checked
					})
				});
				activeRunId = run.run_id;
				await refresh();
				startPolling();
			} catch (e) {
				alert(e.message);
			} finally {
				$('startBtn').disabled = false;
			}
		}

		async function refreshRuns() {
			const runs = await api('/api/runs');
			if (!activeRunId) {
				const activeRun = runs.find(run => ['queued', 'running'].includes(run.status));
				if (activeRun) activeRunId = activeRun.run_id;
			}
			$('runs').innerHTML = runs.length ? runs.map(run => `
				<button class="run-item ${run.run_id === activeRunId ? 'active' : ''}" data-run="${esc(run.run_id)}">
					${statusBadge(run.status)}
					<strong>${esc(run.params.years)}</strong><br>
					<small>${esc(run.run_id)}</small>
				</button>
			`).join('') : '<div class="empty">No hay runs guardados.</div>';
			document.querySelectorAll('[data-run]').forEach(btn => {
				btn.onclick = () => { activeRunId = btn.dataset.run; refresh(); startPolling(); };
			});
			$('latestBtn').classList.toggle('active', !activeRunId);
			return runs;
		}

		function renderSummary(run) {
			const summary = run.summary || {};
			const output = summary.output || {};
			const era5 = summary.era5 || {};
			$('status').innerHTML = statusBadge(run.status);
			$('stage').textContent = run.stage || '-';
			$('rows').textContent = output.rows ?? '-';
			$('missing').textContent = ((era5.missing_months || era5.missing_years || [])).length || '-';

			const after = era5.after || {};
			$('inventory').innerHTML = Object.keys(after).length ? Object.entries(after).map(([year, ok]) => `
				<div class="year ${ok ? '' : 'missing'}">
					<strong>${esc(year)}</strong>
					<small>${ok ? 'ERA5 ok' : 'sin NetCDF'}</small>
				</div>
			`).join('') : '<div class="empty">Sin inventario todavía.</div>';

			const quality = output.era5_match_quality || {};
			$('quality').innerHTML = Object.keys(quality).length ? Object.entries(quality).map(([k, v]) => `
				<div class="event"><time>${esc(k)}</time><div><strong>${esc(v)}</strong> filas</div></div>
			`).join('') : '<div class="empty">Sin output.</div>';
			$('outputPath').innerHTML = output.versioned_output ? `
				<strong>versioned</strong><br>${esc(output.versioned_output)}<br><br>
				<strong>latest</strong><br>${esc(output.latest_output || '-')}<br><br>
				<strong>features</strong><br>${esc(output.features_report_path || '-')}
			` : '-';
		}

		function renderLatestOutput(latest) {
			const quality = latest.era5_match_quality || {};
			$('status').innerHTML = latest.exists ? statusBadge(latest.status === 'error' ? 'failed' : 'latest') : statusBadge('sin output');
			$('stage').textContent = latest.stage || '-';
			$('rows').textContent = latest.rows ?? '-';
			$('missing').textContent = quality.missing ?? '-';
			$('inventory').innerHTML = '<div class="empty">Inventario disponible al seleccionar un run monitoreado.</div>';
			$('quality').innerHTML = Object.keys(quality).length ? Object.entries(quality).map(([k, v]) => `
				<div class="event"><time>${esc(k)}</time><div><strong>${esc(v)}</strong> filas</div></div>
			`).join('') : '<div class="empty">Sin output.</div>';

			if (!latest.exists) {
				$('outputPath').textContent = '-';
				$('timeline').classList.add('empty');
				$('timeline').textContent = 'No hay output latest todavía.';
				$('logs').textContent = '';
				return;
			}

			const updatedAt = latest.modified_at ? new Date(latest.modified_at).toLocaleString('es-CL') : '-';
			$('outputPath').innerHTML = `
				<strong>latest</strong><br>${esc(latest.latest_output)}<br><br>
				<strong>versioned</strong><br>${esc(latest.versioned_output || '-')}<br><br>
				<strong>updated</strong><br>${esc(updatedAt)}<br><br>
				<strong>features</strong><br>${esc(latest.features_report_path || '-')}<br><br>
				<strong>attribution</strong><br>${esc(latest.attribution_path || '-')}
			`;
			$('timeline').classList.remove('empty');
			$('timeline').innerHTML = `
				<div class="event">
					<time>${fmt(latest.modified_at)}</time>
					<div>${statusBadge(latest.status === 'error' ? 'failed' : 'completed')}<strong>latest output</strong><br>${esc(latest.latest_output)}</div>
				</div>
			`;
			$('logs').innerHTML = esc(`[${fmt(latest.modified_at)}] LATEST ${latest.latest_output}`);
		}

		function renderEvents(events) {
			$('timeline').classList.toggle('empty', !events.length);
			$('timeline').innerHTML = events.length ? events.slice(-10).reverse().map(ev => `
				<div class="event">
					<time>${fmt(ev.ts)}</time>
					<div>${statusBadge(ev.level)}<strong>${esc(ev.stage)}</strong><br>${esc(ev.message)}</div>
				</div>
			`).join('') : 'No hay eventos todavía.';
			$('logs').innerHTML = events.map(ev => esc(`[${fmt(ev.ts)}] ${String(ev.level).toUpperCase()} ${ev.stage}: ${ev.message}`)).join('<br>');
			$('logs').scrollTop = $('logs').scrollHeight;
		}

		async function refresh() {
			$('clock').textContent = new Date().toLocaleString('es-CL');
			await refreshRuns();
			if (!activeRunId) {
				renderLatestOutput(await api('/api/latest-output'));
				return;
			}
			const [run, events] = await Promise.all([
				api(`/api/runs/${activeRunId}`),
				api(`/api/runs/${activeRunId}/events`)
			]);
			renderSummary(run);
			renderEvents(events);
			if (!['queued', 'running'].includes(run.status) && pollTimer) {
				clearInterval(pollTimer);
				pollTimer = null;
			}
		}

		function startPolling() {
			if (pollTimer) clearInterval(pollTimer);
			pollTimer = setInterval(refresh, 1200);
		}

		$('startBtn').onclick = startRun;
		$('refreshBtn').onclick = refresh;
		$('latestBtn').onclick = () => { activeRunId = null; refresh(); startPolling(); };
		refresh().then(startPolling);
	</script>
</body>
</html>"""


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Dashboard local XAI Pipeline Monitor")
	parser.add_argument("--host", default="127.0.0.1")
	parser.add_argument("--port", type=int, default=8000)
	args = parser.parse_args()
	uvicorn.run(app, host=args.host, port=args.port)
