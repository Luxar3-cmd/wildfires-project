# =============================================================================
# XAI-project — Interpretable Prediction of Mega-Fires in Chile (XGBoost + Tree SHAP)
# Course:  INF-473 Explainable AI · UTFSM · Prof. Raquel Pezoa Rivera
# Authors: Eduardo Morales · Octavia Jara · Benjamín Reyes
# File:    scripts/megafire_thresholds.py — Per-region statistical mega-fire thresholds (P95/99, log-normal, Pareto)
# =============================================================================
"""
Propuestas de umbral estadístico para megaincendios CONAF.
Métodos: percentil (P95/P98/P99), log-normal (μ+2σ, μ+2.5σ, μ+3σ),
         Pareto-80% del área, benchmarks de literatura (200/500/1000 ha).
Entrada: data/processed/conaf_enriched_latest.parquet (subset 2016-2017 con ERA5).
Salida: tabla impresa + data/processed/megafire_thresholds.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent
PARQUET = ROOT / "data" / "processed" / "conaf_enriched_latest.parquet"
OUT_MD = ROOT / "data" / "processed" / "megafire_thresholds.md"

BENCHMARKS = [200, 500, 1000]   # ha — anclas de literatura internacional
MIN_EVENTS = 50                  # mínimo para clasificar como muestra suficiente


def compute_thresholds(series: pd.Series) -> dict:
    """Aplica los 4 métodos sobre una serie de hectáreas positivas."""
    pos = series[series > 0].dropna()
    n = len(pos)
    total_area = pos.sum()

    row: dict = {"n_positivos": n, "area_total_ha": round(total_area, 1)}

    # --- Percentiles ---
    for p in [95, 98, 99]:
        row[f"p{p}_ha"] = round(np.percentile(pos, p), 1) if n > 0 else np.nan

    # --- Log-normal (ajuste por MLE sobre log(ha)) ---
    if n >= 10:
        log_vals = np.log(pos)
        mu, sigma = log_vals.mean(), log_vals.std(ddof=1)
        for k, label in [(2, "2s"), (2.5, "2.5s"), (3, "3s")]:
            row[f"lognorm_{label}_ha"] = round(np.exp(mu + k * sigma), 1)
    else:
        for label in ["2s", "2.5s", "3s"]:
            row[f"lognorm_{label}_ha"] = np.nan

    # --- Pareto-80%: umbral mínimo donde el 20% superior acumula ≥80% del área ---
    if n >= 10:
        sorted_pos = np.sort(pos.values)[::-1]   # mayor a menor
        cumsum = np.cumsum(sorted_pos)
        target = 0.80 * total_area
        idx = np.searchsorted(cumsum, target)
        idx = min(idx, n - 1)
        row["pareto80_ha"] = round(float(sorted_pos[idx]), 1)
    else:
        row["pareto80_ha"] = np.nan

    # --- Benchmarks de literatura: percentil de cada umbral fijo ---
    for ha in BENCHMARKS:
        pct = (pos <= ha).mean() * 100 if n > 0 else np.nan
        row[f"pct_rank_{ha}ha"] = round(pct, 1) if not np.isnan(pct) else np.nan

    # --- Umbral recomendado ---
    p99 = row.get("p99_ha", np.nan)
    pareto = row.get("pareto80_ha", np.nan)
    if n >= MIN_EVENTS:
        vals = [v for v in [p99, pareto] if not np.isnan(v)]
        row["recomendado_ha"] = round(max(vals), 1) if vals else np.nan
    else:
        row["recomendado_ha"] = round(p99, 1) if not np.isnan(p99) else np.nan

    row["nota"] = "baja_muestra" if n < MIN_EVENTS else ""
    return row


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    col = "superficie_quemada_total_ha"
    rows = []

    # Fila global
    global_row = compute_thresholds(df[col])
    global_row["region"] = "GLOBAL"
    global_row["n_total"] = len(df)
    rows.append(global_row)

    # Por región
    for region, grp in sorted(df.groupby("region")):
        r = compute_thresholds(grp[col])
        r["region"] = region
        r["n_total"] = len(grp)
        rows.append(r)

    cols_order = [
        "region", "n_total", "n_positivos",
        "p95_ha", "p98_ha", "p99_ha",
        "lognorm_2s_ha", "lognorm_2.5s_ha", "lognorm_3s_ha",
        "pareto80_ha",
        "pct_rank_200ha", "pct_rank_500ha", "pct_rank_1000ha",
        "recomendado_ha", "nota",
    ]
    return pd.DataFrame(rows)[cols_order]


def save_markdown(table: pd.DataFrame) -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    n_total = int(table.loc[table["region"] == "GLOBAL", "n_total"].iloc[0])
    lines = [
        "# Propuestas de umbral: megaincendio CONAF (subset 2016-2017)",
        "",
        f"**Variable**: `superficie_quemada_total_ha` | **Período**: 2016-2017 | **N eventos**: {n_total:,}",
        "**Fuente**: `data/processed/conaf_enriched_latest.parquet` (CONAF + ERA5 matched).",
        "",
        "## Métodos",
        "| Método | Descripción |",
        "|--------|-------------|",
        "| `p95/98/99_ha` | Percentil sobre eventos con superficie >0 |",
        "| `lognorm_Xs_ha` | exp(μ + X·σ) sobre log(ha), fit MLE |",
        "| `pareto80_ha` | Umbral mínimo donde eventos encima acumulan ≥80% del área total |",
        "| `pct_rank_Yha` | Percentil empírico donde cae el benchmark Y ha |",
        "| `recomendado_ha` | max(P99, Pareto-80) si n≥50; P99 si n<50 |",
        "",
        "## Resultados",
        "",
    ]

    # Construir tabla markdown manualmente
    headers = list(table.columns)
    sep = ["-" * max(len(h), 5) for h in headers]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(sep) + " |")
    for _, row in table.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float) and np.isnan(v):
                cells.append("—")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Interpretación rápida",
        "",
        "- **Pareto-80%** es el umbral más conservador con impacto operativo claro: los eventos encima explican el 80% del área quemada total.",
        "- **P99** captura el 1% más extremo de la distribución estadística.",
        "- `baja_muestra` indica regiones con <50 eventos positivos — los umbrales son orientativos.",
        "- Los `pct_rank_*` muestran dónde caen los benchmarks internacionales en la distribución chilena.",
    ]

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReporte guardado en: {OUT_MD}")


def main():
    if not PARQUET.exists():
        print(f"ERROR: no se encontró {PARQUET}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(PARQUET)
    table = build_table(df)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.1f}".format)
    print("\n=== UMBRALES DE MEGAINCENDIO — CONAF 2016-2017 (subset enriched con ERA5) ===\n")
    print(table.to_string(index=False))

    save_markdown(table)


if __name__ == "__main__":
    main()
