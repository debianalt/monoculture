"""
01 — Geocodificacion de sociedades a radios censales via Nominatim
===================================================================
Nominatim OSM, 1 req/1.1s, checkpoint cada 200 queries.
Resumable: si se interrumpe, retoma desde el cache.

    python 01_geocode_sociedades.py           # geocode + assign
    python 01_geocode_sociedades.py geocode   # solo geocode (resumable)
    python 01_geocode_sociedades.py assign    # solo spatial join
"""

import json
import re
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from sqlalchemy import create_engine

import requests

PROJECT = Path(__file__).parent
DATA_DIR = PROJECT / "data"
ARCA_DATA = Path(r"C:\Users\ant\OneDrive\gee\arca\data")
CACHE_FILE = DATA_DIR / "geocode_cache.json"
OUTPUT_FILE = DATA_DIR / "geocoded_sociedades.parquet"
DB_URL = "postgresql://postgres:postgres@localhost:5432/ndvi_misiones"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "MisionesInstitutionalEcology/1.0 (academic research; regomez@conicet.gov.ar)"}
RATE_LIMIT = 1.1
CHECKPOINT_EVERY = 200
MISIONES_BBOX = {"lat_min": -28.2, "lat_max": -25.5, "lon_min": -56.1, "lon_max": -53.5}


def normalize_address(calle, numero, localidad):
    calle = str(calle).strip() if pd.notna(calle) else ""
    numero = str(numero).strip() if pd.notna(numero) else ""
    localidad = str(localidad).strip() if pd.notna(localidad) else ""

    if calle in ("-", ".", "0", "00", "000", "S/D", "S/N", "SN", ""):
        return f"{localidad}, Misiones, Argentina"
    if numero in ("-", ".", "0", "00", "000", "S/N", "SN", "nan", ""):
        numero = ""

    calle = re.sub(r"^AV\.?\s+", "AVENIDA ", calle)
    calle = re.sub(r"^CNEL\.?\s+", "CORONEL ", calle)
    calle = re.sub(r"^GRAL\.?\s+", "GENERAL ", calle)
    calle = re.sub(r"^DR\.?\s+", "DOCTOR ", calle)
    calle = re.sub(r"^STA\.?\s+", "SANTA ", calle)
    calle = re.sub(r"\s+(PISO|P\.?\s*\d|DPTO|DTO|OF\.?|OFICINA|LOCAL)\s*.*$", "", calle, flags=re.IGNORECASE)

    parts = [calle, numero] if numero else [calle]
    return f"{' '.join(parts)}, {localidad}, Misiones, Argentina"


def geocode_one(address):
    params = {"q": address, "format": "json", "limit": 1, "countrycodes": "ar"}
    try:
        r = requests.get(NOMINATIM_URL, params=params, headers=HEADERS, timeout=15)
        if r.status_code == 200 and r.json():
            res = r.json()[0]
            lat, lon = float(res["lat"]), float(res["lon"])
            if (MISIONES_BBOX["lat_min"] <= lat <= MISIONES_BBOX["lat_max"] and
                MISIONES_BBOX["lon_min"] <= lon <= MISIONES_BBOX["lon_max"]):
                return {"lat": lat, "lon": lon, "type": res.get("type", ""), "class": res.get("class", "")}
        if r.status_code == 429:
            print("  429! Esperando 120s...", flush=True)
            time.sleep(120)
            return geocode_one(address)
    except requests.exceptions.RequestException as e:
        print(f"  Request error: {e}", flush=True)
    return None


def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def run_geocode():
    print("=" * 60, flush=True)
    print("GEOCODE — Nominatim street-level", flush=True)
    print("=" * 60, flush=True)

    soc = pd.read_parquet(ARCA_DATA / "sociedades_misiones.parquet")
    soc["address"] = soc.apply(
        lambda r: normalize_address(r["dom_fiscal_calle"], r["dom_fiscal_numero"], r["dom_fiscal_localidad"]),
        axis=1,
    )

    unique_addresses = soc["address"].unique()
    cache = load_cache()
    to_query = [a for a in unique_addresses if a not in cache]

    print(f"  Total: {len(soc)} | Unique: {len(unique_addresses)} | Cached: {len(unique_addresses)-len(to_query)} | To query: {len(to_query)}", flush=True)

    if not to_query:
        print("  All cached!", flush=True)
        return

    eta_min = len(to_query) * RATE_LIMIT / 60
    print(f"  ETA: {eta_min:.0f} min ({eta_min/60:.1f} hrs)", flush=True)

    hits, misses = 0, 0
    for i, addr in enumerate(to_query):
        result = geocode_one(addr)
        if result:
            cache[addr] = result
            hits += 1
        else:
            cache[addr] = None
            misses += 1

        if (i + 1) % 50 == 0:
            pct = (i + 1) / len(to_query) * 100
            print(f"  [{i+1}/{len(to_query)}] {pct:.0f}% | hits={hits} misses={misses} ({hits/(i+1)*100:.0f}%)", flush=True)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_cache(cache)

        time.sleep(RATE_LIMIT)

    save_cache(cache)
    total_h = sum(1 for v in cache.values() if v is not None)
    print(f"\n  Done: {total_h} hits / {len(cache)} total ({total_h/len(cache)*100:.1f}%)", flush=True)


def run_assign():
    print("\n" + "=" * 60, flush=True)
    print("ASSIGN — Spatial join to census radios", flush=True)
    print("=" * 60, flush=True)

    soc = pd.read_parquet(ARCA_DATA / "sociedades_misiones.parquet")
    soc["address"] = soc.apply(
        lambda r: normalize_address(r["dom_fiscal_calle"], r["dom_fiscal_numero"], r["dom_fiscal_localidad"]),
        axis=1,
    )

    cache = load_cache()
    soc["lat"] = soc["address"].map(lambda a: cache.get(a, {}).get("lat") if cache.get(a) else None)
    soc["lon"] = soc["address"].map(lambda a: cache.get(a, {}).get("lon") if cache.get(a) else None)
    soc["osm_type"] = soc["address"].map(lambda a: cache.get(a, {}).get("type") if cache.get(a) else None)

    geocoded = soc[soc["lat"].notna()].copy()
    not_geocoded = soc[soc["lat"].isna()].copy()
    print(f"  Geocoded: {len(geocoded)}/{len(soc)} ({len(geocoded)/len(soc)*100:.1f}%)", flush=True)
    print(f"  Not geocoded: {len(not_geocoded)}", flush=True)

    if len(not_geocoded) > 0:
        print(f"  Top unresolved localidades:", flush=True)
        for loc, n in not_geocoded["dom_fiscal_localidad"].value_counts().head(10).items():
            print(f"    {loc}: {n}", flush=True)

    if len(geocoded) == 0:
        print("  Nothing to assign!", flush=True)
        return

    geometry = [Point(lon, lat) for lon, lat in zip(geocoded["lon"], geocoded["lat"])]
    gdf = gpd.GeoDataFrame(geocoded, geometry=geometry, crs="EPSG:4326")

    engine = create_engine(DB_URL)
    print("  Loading radios...", flush=True)
    radios = gpd.read_postgis("SELECT redcode, dpto, geom FROM radios_misiones", engine, geom_col="geom")

    print("  Spatial join...", flush=True)
    joined = gpd.sjoin(gdf, radios, how="left", predicate="within")

    # Nearest for points outside all polygons
    unmatched = joined["redcode"].isna()
    if unmatched.sum() > 0:
        print(f"  Nearest for {unmatched.sum()} outside polygons...", flush=True)
        radio_cents = radios.geometry.centroid
        for idx in joined[unmatched].index:
            pt = joined.at[idx, "geometry"]
            ni = radio_cents.distance(pt).idxmin()
            joined.at[idx, "redcode"] = radios.at[ni, "redcode"]

    out = joined.drop_duplicates(subset="cuit", keep="first").copy()

    # Add departamento
    lookup_path = DATA_DIR / "lookup_localidad_departamento.csv"
    if lookup_path.exists():
        lk = pd.read_csv(lookup_path)
        out["departamento"] = out["dom_fiscal_localidad"].map(dict(zip(lk["localidad"], lk["departamento"])))

    cols = [
        "cuit", "razon_social", "tipo_societario",
        "fecha_hora_contrato_social", "actividad_estado",
        "actividad_codigo", "actividad_descripcion",
        "dom_fiscal_localidad", "dom_fiscal_calle", "dom_fiscal_numero", "dom_fiscal_cp",
        "departamento", "lat", "lon", "osm_type", "redcode",
    ]
    final = out[[c for c in cols if c in out.columns]].copy()
    final.to_parquet(OUTPUT_FILE, index=False)

    print(f"\n  === SUMMARY ===", flush=True)
    print(f"  Total: {len(final)}", flush=True)
    print(f"  With coords: {final['lat'].notna().sum()}", flush=True)
    print(f"  With redcode: {final['redcode'].notna().sum()}", flush=True)
    print(f"  Unique redcodes: {final['redcode'].nunique()}", flush=True)
    print(f"  Saved: {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "todo"
    if modo in ("todo", "geocode"):
        run_geocode()
    if modo in ("todo", "assign"):
        run_assign()
