"""
Phase 0 diagnostic — can the canonical §4.4/§4.5 spec reproduce
org_cluster_assignments_v2.csv (the file all headline analyses consume)?

NON-DESTRUCTIVE. Writes nothing the pipeline depends on. Console only.

Tests several MCA specs against v2 cluster sizes (767/3444/1676/6317/651/1050
at k=6) and per-record label agreement (best-permutation).
"""
import warnings
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import prince
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
GEO = ROOT / "data" / "geocoded_sociedades.parquet"
ENR = ROOT / "data" / "enriquecido_arca_v2.parquet"
V2 = ROOT / "acm" / "data" / "org_cluster_assignments_v2.csv"

import re

SUBTIPO_KEYWORDS = {
    "agro": ["AGROPECUAR", "AGRO ", "AGRICOL", "GANADER", "YERBA", "TABAC",
             "FORESTAL", "MADERA", "ASERRADERO", "VIVERO", "APICOL", "CITRICOL"],
    "religious": ["IGLESIA", "EVANGELICA", "PASTORAL", "PARROQUIA", "TEMPLO",
                  "CRISTIANA", "ADVENTISTA", "BAUTISTA", "PENTECOSTAL",
                  "ASAMBLEA DE DIOS", "METODISTA", "LUTERANA", "MENONITA",
                  "CONGREGACION", "MINISTERIO CRISTIAN", "CULTO"],
    "sports": ["CLUB ", "DEPORTIV", "FUTBOL", "ATLETICO"],
    "indigenous": ["COMUNIDAD ABORIGEN", "COMUNIDAD INDIGENA", "MBYA GUARANI"],
    "education": ["ESCUELA", "COLEGIO", "INSTITUTO", "EDUCACI", "BIBLIOTECA"],
    "health": ["HOSPITAL", "CLINICA", "SANATORIO", "SALUD", "MEDIC", "FARMAC"],
    "transport": ["TRANSPORT", "REMIS", "TAXI", "COLECTIV", "CAMION"],
    "construction": ["CONSTRUC", "INMOBILIAR", "INMUEBLE", "VIVIENDA"],
    "commerce": ["COMERCI", "MERCADO", "SUPERMERCADO", "DISTRIBUID"],
    "tourism": ["TURIS", "HOTEL", "HOSTEL", "CABANA", "ALOJAMIENTO"],
}
RELIG_EXCLUDE = re.compile(
    r"MISIONERA\s+S[.\s]|MISIONERO\s+S[.\s]|MISIONERAS\s+SA|ARENERA|AGUAS\s+MISION", re.I)


def classify_tipo(t):
    t = str(t).upper()
    if "COOPERATIVA" in t: return "Coop"
    if "ASOCIACION CIVIL" in t: return "Asoc"
    if "FUNDACION" in t: return "Fund"
    if "MUTUAL" in t: return "Mutual"
    if "RESPONSABILIDAD LIMITADA" in t: return "SRL"
    if t == "SOCIEDAD ANONIMA": return "SA"
    if "ACCION SIMPLIFICADA" in t: return "SAS"
    return "Otra"


def classify_subtipo(rs):
    rs = str(rs).upper()
    for sub, kws in SUBTIPO_KEYWORDS.items():
        if any(kw in rs for kw in kws):
            if sub == "religious" and RELIG_EXCLUDE.search(rs):
                continue
            return sub
    return "other"


def political_era(y):
    if pd.isna(y): return "unknown"
    y = int(y)
    if y <= 1989: return "pre1990"
    if y <= 1999: return "menem"
    if y <= 2002: return "crisis"
    if y <= 2007: return "n_kirchner"
    if y <= 2015: return "c_kirchner"
    if y <= 2019: return "macri"
    if y <= 2023: return "fernandez"
    return "milei"


def best_agreement(a, b):
    """Max label-permutation agreement between two integer labelings."""
    a = np.asarray(a); b = np.asarray(b)
    ua, ub = np.unique(a), np.unique(b)
    M = np.zeros((len(ua), len(ub)))
    for i, x in enumerate(ua):
        for j, y in enumerate(ub):
            M[i, j] = np.sum((a == x) & (b == y))
    r, c = linear_sum_assignment(-M)
    return M[r, c].sum() / len(a)


def load_base():
    geo = pd.read_parquet(GEO)
    enr = pd.read_parquet(ENR)
    geo["cuit"] = geo["cuit"].astype(str)
    enr["cuit"] = enr["cuit"].astype(str)
    geo["tipo"] = geo["tipo_societario"].map(classify_tipo)
    geo["subtipo"] = geo["razon_social"].map(classify_subtipo)
    geo["year"] = pd.to_datetime(geo["fecha_hora_contrato_social"], errors="coerce").dt.year
    geo["era"] = geo["year"].map(political_era)
    enr["viva_arca"] = enr["error"].isna()
    geo = geo.merge(enr[["cuit", "viva_arca", "es_empleador", "categoria_iva"]],
                    on="cuit", how="left")
    return geo


def run_spec(geo, active_cols, k, drop_unknown_fiscal=True):
    g = geo.copy()
    g = g[g["era"] != "unknown"]
    if drop_unknown_fiscal:
        g = g[g["actividad_estado"].notna()]
    g["estado_bin"] = np.where(g["actividad_estado"] == "BD", "cancelled", "active")
    g["estado_raw"] = g["actividad_estado"].fillna("unknown")
    g["departamento"] = g["departamento"].fillna("unknown")
    act = g[active_cols].copy()
    for col in act.columns:
        vc = act[col].value_counts()
        rare = vc[vc < 30].index
        if len(rare):
            act.loc[act[col].isin(rare), col] = "other_" + col
    mca = prince.MCA(n_components=10, random_state=42).fit(act)
    rc = mca.row_coordinates(act)
    Z = linkage(rc.iloc[:, :5].values, method="ward")
    lab = fcluster(Z, t=k, criterion="maxclust")
    return g.assign(_cl=lab), len(g)


def main():
    v2 = pd.read_csv(V2)
    v2["cuit"] = v2["cuit"].astype(str)
    v2_sizes = sorted(v2["cluster"].value_counts().values, reverse=True)
    print("v2 target sizes (k=6):", v2_sizes, "N=", len(v2))
    print("manuscript clusters 1-4: 767, 3444, 1676, 6317; 5+6 -> InstServ 1701\n")

    geo = load_base()
    specs = {
        "4act canonical (tipo,subtipo,era,estado_bin) k6": (
            ["tipo", "subtipo", "era", "estado_bin"], 6),
        "4act estado_raw k6": (["tipo", "subtipo", "era", "estado_raw"], 6),
        "5act +dept (deposited 04) estado_raw k6": (
            ["tipo", "subtipo", "era", "departamento", "estado_raw"], 6),
        "5act +dept estado_bin k6": (
            ["tipo", "subtipo", "era", "departamento", "estado_bin"], 6),
        "4act canonical k5": (["tipo", "subtipo", "era", "estado_bin"], 5),
    }
    for name, (cols, k) in specs.items():
        try:
            g, n = run_spec(geo, cols, k)
            sizes = sorted(pd.Series(g["_cl"]).value_counts().values, reverse=True)
            m = g[["cuit", "_cl"]].merge(v2, on="cuit", how="inner")
            agr = best_agreement(m["_cl"].values, m["cluster"].values) if len(m) else 0
            tag = "  <== MATCH" if sizes == v2_sizes else ""
            print(f"[{name}]")
            print(f"  N={n}  sizes={sizes}{tag}")
            print(f"  v2 label agreement (best perm, shared={len(m)}): {agr:.4f}\n")
        except Exception as e:
            print(f"[{name}] FAILED: {e}\n")


if __name__ == "__main__":
    main()
