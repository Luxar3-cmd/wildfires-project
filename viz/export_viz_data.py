# =============================================================================
# XAI-project — Visualizador del pipeline ERA5 (grilla, KDTree, salto a tierra)
# File: viz/export_viz_data.py — Exporta datos compactos para el artefacto React
# =============================================================================
"""Genera viz/viz_data.json (compacto) desde el parquet enriquecido + la grilla ERA5.

Por incendio: lat/lon, región, año, calidad del match, snap_km, área, FRP, L1/L2 y,
para los que están dentro del bbox, la celda de tierra más cercana (destino del salto,
recomputada porque el parquet solo guarda la distancia). Incluye la grilla tierra/mar,
las celdas de tierra del KDTree y el contorno de Chile (Natural Earth, simplificado).
"""
import base64
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import DATA_PROCESSED  # noqa: E402
from src.era5 import _haversine_km  # noqa: E402

PARQUET = DATA_PROCESSED / "conaf_enriched_2012_2018.parquet"
REF_NC = ROOT / "data/raw/era5/era5_land_2012_01.nc"
OUT = Path(__file__).resolve().parent / "viz_data.json"
CHILE_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson"

QUALITY_ORDER = ["good", "land_snapped", "water", "out_of_coverage", "poor", "missing"]
MEGAFIRE_HA = 1000


def f32_b64(arr):
    """Empaqueta un array como Float32 little-endian codificado en base64.

    Formato compacto y rápido de parsear en JavaScript.

    Args:
        arr: Array (o secuencia) de números a serializar.

    Returns:
        Cadena base64 con los bytes Float32 little-endian del array.
    """
    return base64.b64encode(np.asarray(arr, dtype="<f4").tobytes()).decode("ascii")


def build_land_arrays(ds):
    """Extrae la máscara tierra/mar y las coordenadas de la grilla del NetCDF de referencia.

    Args:
        ds: Dataset xarray del NetCDF ERA5-Land de referencia.

    Returns:
        Tupla ``(lats, lons, land2d)``: vectores de latitudes y longitudes y la máscara
        booleana 2D de celdas de tierra (donde la variable de referencia no es nula).
    """
    ref = next((v for v in ds.data_vars if v == "t2m"), None) or next(iter(ds.data_vars))
    latn = "latitude" if "latitude" in ds.coords else "lat"
    lonn = "longitude" if "longitude" in ds.coords else "lon"
    da = ds[ref]
    reduce_dims = [d for d in da.dims if d not in (latn, lonn)]
    land2d = (da.notnull().any(dim=reduce_dims) if reduce_dims else da.notnull()).values.astype(bool)
    lats = ds[latn].values.astype(float)
    lons = ds[lonn].values.astype(float)
    return lats, lons, land2d


def nearest_land_index(land_lat, land_lon, coslat):
    """Crea una función de consulta del vecino de tierra más cercano vía KDTree.

    Construye un KDTree equirectangular sobre las celdas de tierra (corrigiendo la
    longitud por el coseno de la latitud media).

    Args:
        land_lat: Latitudes de las celdas de tierra.
        land_lon: Longitudes de las celdas de tierra.
        coslat: Coseno de la latitud media, factor de corrección de la longitud.

    Returns:
        Función ``q(lat, lon)`` que devuelve ``(lat_tierra, lon_tierra, dist_km)``.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([land_lat, land_lon * coslat]))

    def q(lat, lon):
        """Devuelve la celda de tierra más cercana a (lat, lon) y su distancia en km."""
        _, i = tree.query([lat, lon * coslat])
        llat, llon = float(land_lat[i]), float(land_lon[i])
        return llat, llon, _haversine_km(lat, lon, llat, llon)
    return q


def fetch_chile_outline():
    """Descarga el contorno de Chile (Natural Earth), lo simplifica y lo devuelve como anillos.

    Returns:
        Lista de anillos ``[[[lon, lat], ...], ...]`` (polígonos exteriores simplificados),
        o ``None`` si la descarga falla o no se encuentra Chile en el GeoJSON.
    """
    try:
        with urllib.request.urlopen(CHILE_URL, timeout=30) as r:
            gj = json.load(r)
    except Exception as e:
        print(f"  WARN: no se pudo bajar el contorno de Chile ({e}); se omite.")
        return None
    feat = next((f for f in gj["features"]
                 if (f["properties"].get("ADMIN") or f["properties"].get("NAME")) == "Chile"), None)
    if feat is None:
        print("  WARN: no se halló 'Chile' en el GeoJSON; se omite.")
        return None
    from shapely.geometry import shape
    geom = shape(feat["geometry"]).simplify(0.03, preserve_topology=True)  # ~3 km, liviano
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    rings = []
    for p in polys:
        coords = [[round(x, 3), round(y, 3)] for x, y in p.exterior.coords]
        if len(coords) >= 4:
            rings.append(coords)
    print(f"  Chile: {len(rings)} anillos, {sum(len(r) for r in rings)} vértices")
    return rings


def main():
    """Exporta ``viz_data.json``: incendios, grilla tierra/mar, saltos a tierra y contorno de Chile."""
    print("Cargando parquet...")
    df = pd.read_parquet(PARQUET)
    n = len(df)
    ts = pd.to_datetime(df["fecha_hora_inicio"], errors="coerce")

    lat = pd.to_numeric(df["latitud"], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(df["longitud"], errors="coerce").to_numpy(float)
    year = ts.dt.year.fillna(0).astype(int).to_numpy()
    area = pd.to_numeric(df["superficie_quemada_total_ha"], errors="coerce").to_numpy(float)
    frp = pd.to_numeric(df.get("modis_frp_max_mw"), errors="coerce").to_numpy(float)
    snap = pd.to_numeric(df.get("era5_land_snap_km"), errors="coerce").to_numpy(float)
    l1 = (area >= MEGAFIRE_HA).astype(np.uint8)
    l2 = pd.to_numeric(df.get("label_l2"), errors="coerce").fillna(0).astype(np.uint8).to_numpy()

    regions = df["region"].astype(str).fillna("?")
    reg_levels = sorted(regions.unique().tolist())
    reg_code = {r: i for i, r in enumerate(reg_levels)}
    region_idx = regions.map(reg_code).to_numpy(np.uint8)

    qual = df["era5_match_quality"].astype(str).fillna("missing")
    qlevels = [q for q in QUALITY_ORDER if q in set(qual.unique())]
    qcode = {q: i for i, q in enumerate(qlevels)}
    qual_idx = qual.map(lambda q: qcode.get(q, len(qlevels))).to_numpy(np.uint8)

    print("Construyendo grilla / KDTree de referencia...")
    ds = xr.open_dataset(REF_NC)
    lats, lons, land2d = build_land_arrays(ds)
    ds.close()
    glat, glon = np.meshgrid(lats, lons, indexing="ij")
    land_lat = glat[land2d]
    land_lon = glon[land2d]
    coslat = float(np.cos(np.radians(float(np.mean(lats))))) or 1.0
    query = nearest_land_index(land_lat, land_lon, coslat)

    south, north = float(lats.min()), float(lats.max())
    west, east = float(lons.min()), float(lons.max())

    print("Recomputando destino del salto por incendio (dentro del bbox)...")
    nl_lat = np.full(n, np.nan)
    nl_lon = np.full(n, np.nan)
    nl_km = np.full(n, np.nan)
    in_bbox = (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east) & np.isfinite(lat) & np.isfinite(lon)
    for i in np.where(in_bbox)[0]:
        llat, llon, km = query(lat[i], lon[i])
        nl_lat[i], nl_lon[i], nl_km[i] = llat, llon, km
    print(f"  dentro del bbox: {int(in_bbox.sum())} / {n}")

    chile = fetch_chile_outline()

    # Empaqueta máscara como bits (compacto)
    mask_bits = np.packbits(land2d.ravel()).tolist()

    data = {
        "meta": {
            "n": int(n),
            "bbox": {"south": south, "north": north, "west": west, "east": east},
            "region_levels": reg_levels,
            "quality_levels": qlevels,
            "megafire_ha": MEGAFIRE_HA,
        },
        "grid": {
            "lats": [round(float(x), 4) for x in lats],
            "lons": [round(float(x), 4) for x in lons],
            "nlat": len(lats), "nlon": len(lons),
            "land_mask_bits": mask_bits,  # packbits row-major (nlat*nlon)
        },
        "land_cells": {"lat_b64": f32_b64(land_lat), "lon_b64": f32_b64(land_lon), "n": int(land_lat.size)},
        "fires": {
            "lat_b64": f32_b64(lat),
            "lon_b64": f32_b64(lon),
            "nl_lat_b64": f32_b64(np.nan_to_num(nl_lat, nan=0.0)),
            "nl_lon_b64": f32_b64(np.nan_to_num(nl_lon, nan=0.0)),
            "nl_km_b64": f32_b64(np.nan_to_num(nl_km, nan=-1.0)),
            "snap_b64": f32_b64(np.nan_to_num(snap, nan=0.0)),
            "area_b64": f32_b64(np.nan_to_num(area, nan=0.0)),
            "frp_b64": f32_b64(np.nan_to_num(frp, nan=-1.0)),
            "year": year.astype(int).tolist(),
            "region": region_idx.tolist(),
            "quality": qual_idx.tolist(),
            "l1": l1.tolist(),
            "l2": l2.tolist(),
            "in_bbox": in_bbox.astype(np.uint8).tolist(),
        },
        "chile": chile,
    }

    OUT.write_text(json.dumps(data, separators=(",", ":")))
    mb = OUT.stat().st_size / 1e6
    print(f"\n✅ {OUT}  ({mb:.2f} MB)")
    print(f"   incendios={n} | land_snapped(snap>0)={int((snap>0).sum())} | "
          f"out_of_coverage={int((qual=='out_of_coverage').sum())} | "
          f"grilla={len(lats)}x{len(lons)} | celdas_tierra={land_lat.size}")


if __name__ == "__main__":
    main()
