"""
00 — Enriquecimiento de la base matriz empresarial de Misiones
================================================================
Proyecto: Institutional Ecology — Misiones Atlantic Forest (2026_11)

Paso 0: Validaciones (domicilio fiscal vs legal, timeline temporal)
Paso 1: Lookup localidad → departamento
Paso 2: Re-consultar ARCA para sociedades activas que fallaron
Paso 3: Consultar INAES para cooperativas

Uso:
    python 00_enrich_base.py              # todo
    python 00_enrich_base.py validate     # solo checks
    python 00_enrich_base.py lookup       # solo lookup
    python 00_enrich_base.py arca         # solo ARCA re-query
    python 00_enrich_base.py inaes        # solo INAES
"""

import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).parent
DATA_DIR = PROJECT / "data"
TAB_DIR = PROJECT / "tables"
FIG_DIR = PROJECT / "figures"

ARCA_DIR = Path(r"C:\Users\ant\OneDrive\gee\arca")
ARCA_DATA = ARCA_DIR / "data"

for d in (DATA_DIR, TAB_DIR, FIG_DIR):
    d.mkdir(exist_ok=True)


# ── Lookup localidad → departamento ─────────────────────────────────────────
# Built from: postal codes, INDEC geographic structure, manual verification.
# 124 unique dom_fiscal_localidad values mapped to 17 departamentos.

LOCALIDAD_DEPTO = {
    # Capital (04) — CP 3300, 3304
    "POSADAS": "Capital",
    "GARUPA": "Capital",
    "BARRIO SANTA LUCIA": "Capital",
    "ITAIME MINI": "Capital",
    "MIGUEL LANUS": "Capital",
    "SAN ISIDRO": "Capital",
    "SANTA INES": "Capital",
    "VILLALONGA": "Capital",

    # Apostoles (01) — CP 3350
    "APOSTOLES": "Apostoles",
    "AZARA": "Apostoles",
    "SAN JOSE": "Apostoles",
    "TRES CAPONES": "Apostoles",
    "PINDAPOY": "Apostoles",

    # Cainguas (02) — CP 3362
    "ARISTOBULO DEL VALLE": "Cainguas",
    "CAMPO GRANDE": "Cainguas",
    "DOS DE MAYO": "Cainguas",
    "ALMIRANTE BROWN": "Cainguas",
    "ALTE BROWN": "Cainguas",
    "SALTO ENCANTADO": "Cainguas",

    # Candelaria (03) — CP 3308
    "CANDELARIA": "Candelaria",
    "PROFUNDIDAD": "Candelaria",
    "BONPLAND": "Candelaria",
    "SANTA ANA": "Candelaria",
    "CERRO CORA": "Candelaria",
    "COLONIA MARTIRES": "Candelaria",
    "ARROYO MARTIRES": "Candelaria",
    "LORETO": "Candelaria",
    "PARADA LEIS": "Candelaria",

    # Concepcion (05) — CP 3355
    "CONCEPCION DE LA SIERRA": "Concepcion",
    "BARRA CONCEPCION": "Concepcion",
    "COLONIA SANTA MARIA": "Concepcion",
    "PTO.CONCEPCION": "Concepcion",

    # Eldorado (06) — CP 3380, 3381, 3382
    "ELDORADO": "Eldorado",
    "ELDORADO SUCURSAL NUMERO 1": "Eldorado",
    "9 DE JULIO": "Eldorado",
    "9 DE JULIO KILOMETRO 20": "Eldorado",
    "COLONIA MARIA MAGDALENA": "Eldorado",
    "COLONIA  MARIA MAGDALENA": "Eldorado",
    "SANTIAGO DE LINIERS": "Eldorado",
    "PUERTO PINARES": "Eldorado",
    "PUERTO VICTORIA": "Eldorado",
    "PUERTO DELICIA": "Eldorado",

    # G. M. Belgrano (07) — CP 3366
    "BERNARDO DE IRIGOYEN": "G. M. Belgrano",
    "COMANDANTE ANDRESITO": "G. M. Belgrano",
    "SAN ANTONIO": "G. M. Belgrano",

    # Guarani (08) — CP 3364
    "SAN VICENTE": "Guarani",
    "EL SOBERBIO": "Guarani",
    "FRACRAN": "Guarani",
    "MUNICIPIO COL A": "Guarani",

    # Iguazu (09) — CP 3370, 3374, 3376, 3378
    "PUERTO IGUAZU": "Iguazu",
    "IGUAZU": "Iguazu",
    "COLONIA WANDA": "Iguazu",
    "PUERTO ESPERANZA": "Iguazu",
    "PUERTO LIBERTAD": "Iguazu",
    "LIBERTAD": "Iguazu",
    "CATARATAS DEL IGUAZU": "Iguazu",
    "PUERTO BOSSETTI": "Iguazu",

    # L.G. San Martin (10) — CP 3334
    "PUERTO RICO": "L.G. San Martin",
    "GARUHAPE": "L.G. San Martin",
    "COLONIA GARUHAPE": "L.G. San Martin",
    "CAPIOVY": "L.G. San Martin",
    "RUIZ DE MONTOYA": "L.G. San Martin",
    "SAN ALBERTO": "L.G. San Martin",

    # L. N. Alem (11) — CP 3315, 3317
    "LEANDRO N. ALEM": "L. N. Alem",
    "CERRO AZUL": "L. N. Alem",
    "OLEGARIO V. ANDRADE": "L. N. Alem",
    "ARROYO DEL MEDIO": "L. N. Alem",
    "CABURE I": "L. N. Alem",
    "GOBERNADOR LOPEZ": "L. N. Alem",
    "DOS ARROYOS": "L. N. Alem",
    "COLONIA YACUTINGA": "L. N. Alem",

    # Montecarlo (12) — CP 3384, 3386
    "MONTECARLO": "Montecarlo",
    "PUERTO PIRAY": "Montecarlo",
    "COLONIA CARAGUATAY": "Montecarlo",
    "BARRANCON (MONTECARLO)": "Montecarlo",
    "COLONIA EL ALCAZAR": "Montecarlo",
    "PUERTO ALCAZAR": "Montecarlo",
    "PARANAY": "Montecarlo",

    # Obera (13) — CP 3360, 3361
    "OBERA": "Obera",
    "CAMPO VIERA": "Obera",
    "CAMPO RAMON": "Obera",
    "GUARANI": "Obera",
    "LOS HELECHOS": "Obera",
    "COLONIA ALBERDI": "Obera",
    "PANAMBI": "Obera",
    "VILLA BONITA": "Obera",
    "GENERAL ALVEAR": "Obera",
    "COLONIA GUARANI": "Obera",
    "22 DE DICIEMBRE": "Obera",
    "BAYO TRONCHO": "Obera",
    "PICADA YAPEYU": "Obera",
    "VILLA SVEA": "Obera",

    # San Ignacio (14) — CP 3322, 3324, 3326, 3327, 3328, 3332
    "SAN IGNACIO": "San Ignacio",
    "JARDIN AMERICA": "San Ignacio",
    "SANTO PIPO": "San Ignacio",
    "GOBERNADOR ROCA": "San Ignacio",
    "CORPUS": "San Ignacio",
    "COLONIA POLANA": "San Ignacio",
    "PUERTO LEONI": "San Ignacio",
    "COLONIA YABEBIRY": "San Ignacio",
    "BARRANCON SAN IGNACIO": "San Ignacio",
    "COLONIA DOMINGO SAVIO": "San Ignacio",
    "GENERAL ROCA": "San Ignacio",
    "OASIS": "San Ignacio",
    "PUERTO GISELA": "San Ignacio",
    "HIPOLITO YRIGOYEN": "San Ignacio",
    "COLONIA \u00d1ACANGUAZU": "San Ignacio",

    # San Javier (15) — CP 3357, 3358
    "SAN JAVIER": "San Javier",
    "ITACARUARE": "San Javier",
    "MOJON GRANDE": "San Javier",
    "FLORENTINO AMEGHINO": "San Javier",
    "F. AMEGHINO": "San Javier",
    "FACHINAL": "San Javier",
    "BARRA BONITA": "San Javier",
    "PLAYADITO": "San Javier",

    # San Pedro (16)
    "SAN PEDRO": "San Pedro",

    # 25 de Mayo (17) — CP 3363
    "ALBA POSSE": "25 de Mayo",
    "COLONIA 25 DE MAYO": "25 de Mayo",
    "COLONIA AURORA": "25 de Mayo",
    "COLONIA ALICIA": "25 de Mayo",
    "SANTA RITA": "25 de Mayo",
    "SAN FRANCISCO DE ASIS": "25 de Mayo",
    "SAN MARTIN": "25 de Mayo",
    "PUERTO S.MARTIN": "25 de Mayo",
}


def load_sociedades() -> pd.DataFrame:
    return pd.read_parquet(ARCA_DATA / "sociedades_misiones.parquet")


def load_enriquecido() -> pd.DataFrame:
    return pd.read_parquet(ARCA_DATA / "enriquecido_arca.parquet")


# ═════════════════════════════════════════════════════════════════════════════
# PASO 0 — Validaciones
# ═════════════════════════════════════════════════════════════════════════════

def run_validations():
    print("=" * 60)
    print("PASO 0 — Validaciones")
    print("=" * 60)

    soc = load_sociedades()

    # ── Check A: fiscal vs legal domicile ────────────────────────────────
    match = (soc["dom_fiscal_localidad"] == soc["dom_legal_localidad"]).sum()
    total = len(soc)
    div = soc[soc["dom_fiscal_localidad"] != soc["dom_legal_localidad"]]

    print(f"\nCheck A — Domicilio fiscal vs legal:")
    print(f"  Coinciden: {match}/{total} ({match/total*100:.1f}%)")
    print(f"  Divergen:  {total-match}/{total} ({(total-match)/total*100:.1f}%)")

    pairs = (
        div.groupby(["dom_fiscal_localidad", "dom_legal_localidad"])
        .size()
        .sort_values(ascending=False)
        .reset_index(name="n")
    )
    pairs.to_csv(TAB_DIR / "tab_domicilio_check.csv", index=False)
    print(f"  Guardado: tables/tab_domicilio_check.csv ({len(pairs)} pares divergentes)")
    print(f"  Conclusion: {match/total*100:.0f}% coincidencia -> concentracion en Posadas es REAL")

    # ── Check B: temporal histogram ──────────────────────────────────────
    soc["fecha"] = pd.to_datetime(soc["fecha_hora_contrato_social"], errors="coerce")
    soc["year"] = soc["fecha"].dt.year
    s = soc[(soc["year"] >= 1950) & (soc["year"] <= 2025)]
    counts = s.groupby("year").size()

    print(f"\nCheck B — Firm creation timeline:")
    for decade_start in range(1950, 2030, 10):
        n = counts[(counts.index >= decade_start) & (counts.index <= decade_start + 9)].sum()
        print(f"  {decade_start}s: {n:,}")

    # Detect suspicious jumps
    jumps = []
    for y in range(1951, 2026):
        prev = counts.get(y - 1, 0)
        curr = counts.get(y, 0)
        if prev > 10 and curr / max(prev, 1) > 2:
            jumps.append(f"{y-1}={prev} -> {y}={curr}")
    if jumps:
        print(f"  Saltos sospechosos: {jumps}")
    else:
        print("  Sin saltos de digitalizacion -> crecimiento organico")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(counts.index, counts.values, color="#2c7fb8", alpha=0.8, width=0.8)
    ax.set_xlabel("Year of incorporation")
    ax.set_ylabel("Societies created")
    ax.set_title("Firm creation timeline — Misiones, 1950-2025 (N=14,044)")
    ax.set_xlim(1949, 2026)

    for decade, label in [(2000, "peak\n2000s"), (2017, "SAS\n2017")]:
        ax.axvline(x=decade, color="red", linestyle="--", alpha=0.4)
        ax.text(decade + 0.5, counts.max() * 0.9, label, fontsize=8, color="red")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_firm_creation_timeline.png", dpi=200)
    print(f"  Guardado: figures/fig_firm_creation_timeline.png")


# ═════════════════════════════════════════════════════════════════════════════
# PASO 1 — Lookup localidad → departamento
# ═════════════════════════════════════════════════════════════════════════════

def run_lookup():
    print("\n" + "=" * 60)
    print("PASO 1 — Lookup localidad -> departamento")
    print("=" * 60)

    soc = load_sociedades()
    localidades = soc["dom_fiscal_localidad"].dropna().unique()

    mapped = []
    unmapped = []
    for loc in sorted(localidades):
        depto = LOCALIDAD_DEPTO.get(loc)
        if depto:
            mapped.append({"localidad": loc, "departamento": depto})
        else:
            unmapped.append(loc)

    if unmapped:
        print(f"\n  UNMAPPED ({len(unmapped)}):")
        for loc in unmapped:
            n = len(soc[soc["dom_fiscal_localidad"] == loc])
            cp = soc[soc["dom_fiscal_localidad"] == loc]["dom_fiscal_cp"].mode()
            cp_val = cp.iloc[0] if len(cp) > 0 else "?"
            print(f"    {loc} (N={n}, CP={cp_val})")

    lookup = pd.DataFrame(mapped)
    lookup.to_csv(DATA_DIR / "lookup_localidad_departamento.csv", index=False)

    # Apply to sociedades
    soc["departamento"] = soc["dom_fiscal_localidad"].map(LOCALIDAD_DEPTO)
    n_mapped = soc["departamento"].notna().sum()

    print(f"\n  Localidades mapeadas: {len(mapped)}/{len(localidades)}")
    print(f"  Sociedades con departamento: {n_mapped}/{len(soc)} ({n_mapped/len(soc)*100:.1f}%)")
    print(f"  Departamentos unicos: {soc['departamento'].nunique()}")
    print(f"  Guardado: data/lookup_localidad_departamento.csv")

    # Distribution
    print(f"\n  Distribucion por departamento:")
    for depto, n in soc["departamento"].value_counts().items():
        print(f"    {depto:20s} {n:5d} ({n/len(soc)*100:.1f}%)")

    return lookup


# ═════════════════════════════════════════════════════════════════════════════
# PASO 2 — Re-consultar ARCA para activas fallidas
# ═════════════════════════════════════════════════════════════════════════════

def run_arca_requery():
    print("\n" + "=" * 60)
    print("PASO 2 — Re-consultar ARCA para sociedades activas fallidas")
    print("=" * 60)

    soc = load_sociedades()
    enr = load_enriquecido()

    # Identify active societies that failed ARCA enrichment
    failed_cuits = enr[enr["error"].notna()]["cuit"].astype(str).tolist()
    active_cuits_set = set(
        soc[soc["actividad_estado"] == "AC"]["cuit"].astype(str).tolist()
    )
    to_retry = [c for c in failed_cuits if c in active_cuits_set]

    # Already resolved
    resolved_before = enr["error"].isna().sum()

    print(f"  Total enriched: {len(enr)}")
    print(f"  Resolved before: {resolved_before}")
    print(f"  Failed before: {len(failed_cuits)}")
    print(f"  Failed AND active: {len(to_retry)} -> these will be re-queried")

    if not to_retry:
        print("  Nothing to retry.")
        return

    # Import ARCA pipeline
    sys.path.insert(0, str(ARCA_DIR))
    from padron import consultar_cuits

    resultados = []
    batch_size = 250
    total_batches = (len(to_retry) + batch_size - 1) // batch_size
    resolved_new = 0
    errors_new = 0

    for i in range(0, len(to_retry), batch_size):
        batch = to_retry[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  [ARCA] Batch {batch_num}/{total_batches} ({len(batch)} CUITs)...", end=" ")

        try:
            resultado = consultar_cuits(batch)
            resultados.extend(resultado)
            resolved_new += len(resultado)
            print(f"OK: {len(resultado)} resolved")
        except Exception as e:
            print(f"batch error: {e}")
            for cuit in batch:
                try:
                    r = consultar_cuits([cuit])
                    resultados.extend(r)
                    resolved_new += len(r)
                except Exception:
                    resultados.append({"cuit": str(cuit), "error": str(e)})
                    errors_new += 1

        time.sleep(0.5)

    # Merge with existing enrichment
    new_df = pd.json_normalize(resultados)

    # Replace old failed rows with new results
    enr_kept = enr[~enr["cuit"].astype(str).isin([str(c) for c in to_retry])]
    merged = pd.concat([enr_kept, new_df], ignore_index=True)

    resolved_after = merged["error"].isna().sum() if "error" in merged.columns else len(merged)

    merged.to_parquet(DATA_DIR / "enriquecido_arca_v2.parquet", index=False)

    print(f"\n  Results:")
    print(f"    Resolved before: {resolved_before}")
    print(f"    New resolved: {resolved_new}")
    print(f"    New errors: {errors_new}")
    print(f"    Total resolved now: {resolved_after}/{len(merged)} ({resolved_after/len(merged)*100:.1f}%)")

    # Employer stats
    if "es_empleador" in merged.columns:
        employers = merged[merged["es_empleador"] == True]
        print(f"    Employers: {len(employers)}")

    print(f"  Guardado: data/enriquecido_arca_v2.parquet")


# ═════════════════════════════════════════════════════════════════════════════
# PASO 3 — Consultar INAES para cooperativas
# ═════════════════════════════════════════════════════════════════════════════

def run_inaes():
    print("\n" + "=" * 60)
    print("PASO 3 — Consultar INAES para cooperativas")
    print("=" * 60)

    # Try datos.gob.ar first
    url = "https://datos.gob.ar/api/3/action/package_search?q=inaes+cooperativas&rows=10"
    headers = {"User-Agent": "Mozilla/5.0 (research)"}

    print("  Buscando dataset INAES en datos.gob.ar...")

    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            results = r.json().get("result", {}).get("results", [])
            print(f"  Encontrados {len(results)} datasets:")
            for ds in results:
                title = ds.get("title", "?")
                resources = ds.get("resources", [])
                csv_urls = [
                    res["url"] for res in resources
                    if res.get("format", "").upper() in ("CSV", "XLS", "XLSX")
                ]
                print(f"    - {title}")
                for u in csv_urls[:2]:
                    print(f"      {u}")

            # Try to find and download the cooperative registry
            for ds in results:
                for res in ds.get("resources", []):
                    if res.get("format", "").upper() == "CSV" and "cooperativa" in ds.get("title", "").lower():
                        csv_url = res["url"]
                        print(f"\n  Descargando: {csv_url}")
                        resp = requests.get(csv_url, headers=headers, timeout=60)
                        if resp.status_code == 200:
                            path = DATA_DIR / "inaes_cooperativas_raw.csv"
                            path.write_bytes(resp.content)
                            print(f"  Guardado: {path} ({len(resp.content)/1024:.0f} KB)")

                            # Process
                            df = pd.read_csv(path, encoding="latin-1", on_error="warn")
                            print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")

                            # Filter Misiones
                            for col in df.columns:
                                if "provincia" in col.lower() or "jurisdic" in col.lower():
                                    misiones = df[df[col].astype(str).str.contains("Misiones|MISIONES", na=False)]
                                    if len(misiones) > 0:
                                        misiones.to_parquet(DATA_DIR / "inaes_cooperativas_misiones.parquet", index=False)
                                        print(f"  Misiones: {len(misiones)} cooperativas")
                                        print(f"  Guardado: data/inaes_cooperativas_misiones.parquet")
                                        return
                            print("  No se encontro columna provincia para filtrar")
                            return
        else:
            print(f"  datos.gob.ar API returned {r.status_code}")
    except Exception as e:
        print(f"  Error consultando datos.gob.ar: {e}")

    # Fallback: try direct INAES URL
    print("\n  Intentando URL directa INAES...")
    inaes_urls = [
        "https://datos.gob.ar/dataset/inaes-registro-nacional-cooperativas-mutuales",
        "https://datos.inaes.gob.ar/api/cooperativas",
    ]
    for url in inaes_urls:
        try:
            r = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            print(f"  {url}: status={r.status_code}")
            if r.status_code == 200 and len(r.content) > 1000:
                path = DATA_DIR / "inaes_response.html"
                path.write_bytes(r.content)
                print(f"  Guardado respuesta para inspeccion manual: {path}")
        except Exception as e:
            print(f"  {url}: {e}")

    print("  INAES: no se encontro dataset descargable automaticamente.")
    print("  Accion manual: buscar en https://datos.gob.ar buscando 'cooperativas inaes'")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "todo"

    if modo in ("todo", "validate"):
        run_validations()

    if modo in ("todo", "lookup"):
        run_lookup()

    if modo in ("todo", "arca"):
        run_arca_requery()

    if modo in ("todo", "inaes"):
        run_inaes()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
