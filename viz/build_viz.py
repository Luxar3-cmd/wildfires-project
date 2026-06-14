# =============================================================================
# XAI-project — Visualizador del pipeline ERA5
# File: viz/build_viz.py — Ensambla el .html autocontenido (React inline + datos embebidos)
# =============================================================================
"""Inyecta React UMD (cacheado) + viz_data.json en app.template.html → viz/pipeline_viz.html.

Uso:
    .venv/bin/python viz/export_viz_data.py   # genera viz_data.json (requiere red para Chile)
    .venv/bin/python viz/build_viz.py         # genera pipeline_viz.html (requiere red la 1ª vez)
"""
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "_vendor"
TEMPLATE = HERE / "app.template.html"
DATA = HERE / "viz_data.json"
OUT = HERE / "pipeline_viz.html"

REACT_UMD = {
    "react.production.min.js": "https://unpkg.com/react@18/umd/react.production.min.js",
    "react-dom.production.min.js": "https://unpkg.com/react-dom@18/umd/react-dom.production.min.js",
}


def vendor(name, url):
    p = VENDOR / name
    if not p.exists():
        print(f"  bajando {name}...")
        VENDOR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=30) as r:
            p.write_bytes(r.read())
    return p.read_text(encoding="utf-8")


def main():
    if not DATA.exists():
        raise SystemExit("Falta viz_data.json — corre primero: python viz/export_viz_data.py")

    react_tags = "\n".join(
        f"<script>{vendor(name, url)}</script>" for name, url in REACT_UMD.items()
    )
    html = TEMPLATE.read_text(encoding="utf-8")
    data_text = DATA.read_text(encoding="utf-8")

    html = html.replace("<!--__REACT__-->", react_tags)
    html = html.replace("__DATA__", data_text)

    OUT.write_text(html, encoding="utf-8")
    mb = OUT.stat().st_size / 1e6
    print(f"\n✅ {OUT}  ({mb:.2f} MB) — ábrelo en el navegador.")


if __name__ == "__main__":
    main()
