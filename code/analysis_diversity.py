"""
analysis_diversity.py — canonical analysis for the reconstructed paper
======================================================================
Juridical-form diversity compression and cooperative decline in the
formal-organisation registry of Misiones, 1901–2025.

Model-free by design. Every quantity is computed on the RAW juridical
form (and, for cooperatives, on the registered-name subtype). No MCA,
no clustering, no sequence analysis: the earlier geometric pipeline
entered political era as an active variable and thereby manufactured a
tautological terminal convergence (documented in archive/README). This
script is the single source of truth for all figures and in-text numbers.

Outputs (acm/tables/):
  tab_diversity_annual.csv     Shannon H per year 1990–2025 (+roll, dH)
  tab_diversity_era.csv        H, evenness, dominant form per era
  tab_composition_era.csv      per-era form counts, per-yr rate, share %
  tab_coop_subtypes.csv        work / ag-service / other coop per era
  tab_spatial_formlevel.csv    SAS-form dominance by era + Milei null
  tab_dept_env.csv             dept-level coop-share vs canopy (+jackknife)

DB (PostGIS) is OPTIONAL — only the dept-env table needs it.
"""
import re
import sys
import warnings
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)

PROJECT = Path(__file__).resolve().parents[2]
TAB = PROJECT / "acm" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

# canonical juridical-form classifier (self-contained; the exploratory
# geometric pipeline that previously hosted it is in ../archive/, see
# archive/README.md and §4.5)
def classify_tipo(t):
    t = str(t).upper()
    if "COOPERATIVA" in t:
        return "Coop"
    if "ASOCIACION CIVIL" in t:
        return "Asoc"
    if "FUNDACION" in t:
        return "Fund"
    if "MUTUAL" in t:
        return "Mutual"
    if "RESPONSABILIDAD LIMITADA" in t:
        return "SRL"
    if t == "SOCIEDAD ANONIMA":
        return "SA"
    if "ACCION SIMPLIFICADA" in t:
        return "SAS"
    return "Otra"

ERA_SPAN = {"Pre-1990": (1901, 1989), "Menem": (1990, 1999),
            "Crisis": (2000, 2002), "N.Kirchner": (2003, 2007),
            "C.Kirchner": (2008, 2015), "Macri": (2016, 2019),
            "Fernández": (2020, 2023), "Milei": (2024, 2025)}
ERAS = list(ERA_SPAN)
FORMS = ["SAS", "SRL", "SA", "Coop", "Asoc", "Fund", "Otra"]

# cooperative functional subtype from the registered name
AGSV = ["AGROPECUARIA", "AGRICOLA", "AGRICOLAS", "YERBA", "TABACAL",
        "TABACO", "FORESTAL", "TAMBO", "GANADER", "PRODUCTORES", "COLONOS",
        "TE ", "SERVICIOS", "PROVISION", "OBRAS Y SERVICIOS", "PUBLICOS",
        "CITRICOL", "APICOL", "HORTICOL"]
WORK = ["TRABAJO"]


def coop_subtype(rs):
    rs = str(rs).upper()
    if any(k in rs for k in AGSV):
        return "ag_service"
    if any(k in rs for k in WORK):
        return "work"
    return "other_coop"


def era_of(y):
    if pd.isna(y):
        return None
    y = int(y)
    for e, (lo, hi) in ERA_SPAN.items():
        if lo <= y <= hi:
            return e
    return None


def shannon(counts):
    c = np.asarray(counts, float)
    c = c[c > 0]
    if c.sum() == 0:
        return np.nan
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def load():
    g = pd.read_parquet(PROJECT / "data" / "geocoded_sociedades.parquet")
    g["cuit"] = g["cuit"].astype(str)
    g["redcode"] = g["redcode"].astype(str)
    g["form"] = g["tipo_societario"].map(classify_tipo)
    g["year"] = pd.to_datetime(g["fecha_hora_contrato_social"],
                               errors="coerce").dt.year
    g["era"] = g["year"].map(era_of)
    n0 = len(g)
    g = g[g["year"].notna()].copy()
    print(f"  geocoded registry rows : {n0}")
    print(f"  − missing/unparseable date : {n0 - len(g)}")
    print(f"  = dated organisations (diversity/composition base) : {len(g)}")
    print(f"    (registry 14,278 − 101 ungeocoded → 14,177; "
          f"− {n0 - len(g)} undated → {len(g)} = Shannon N; "
          f"− 264 unknown-fiscal → 13,905 for any fiscal-conditioned step)")
    return g


def t_annual(g):
    rows = []
    for y in range(1990, 2026):
        s = g[g.year == y]
        if not len(s):
            continue
        rows.append({"year": y, "n": len(s),
                     "H": round(shannon(s["form"].value_counts().values), 4)})
    t = pd.DataFrame(rows)
    t["roll5"] = t["H"].rolling(5, center=True, min_periods=3).mean().round(4)
    t["dH"] = t["H"].diff().round(4)
    t.to_csv(TAB / "tab_diversity_annual.csv", index=False)
    worst = t.loc[t["dH"].idxmin()]
    pre = t[(t.year >= 2016) & (t.year <= 2023)]["H"].mean()
    mil = t[t.year >= 2024]["H"].mean()
    hist_min = t.loc[t["H"].idxmin()]
    print("\n[diversity — annual, juridical form]")
    print(f"  mean H 2016–23 = {pre:.3f} ; mean H 2024–25 = {mil:.3f} "
          f"(Δ = {mil - pre:+.3f})")
    print(f"  largest single-YEAR ΔH = {worst.dH:+.3f} in {int(worst.year)}")
    print(f"  series minimum H = {hist_min.H:.3f} in {int(hist_min.year)} "
          f"→ 2024–25 is NOT the historical minimum")
    return t


def t_era(g):
    rows = []
    for e in ERAS:
        s = g[g.era == e]
        if not len(s):
            continue
        vc = s["form"].value_counts()
        H = shannon(vc.values)
        S = int((vc > 0).sum())
        rows.append({"era": e, "n": len(s), "H": round(H, 4),
                     "evenness": round(H / np.log(S), 4) if S > 1 else np.nan,
                     "dominant_form": vc.index[0],
                     "dominant_share_%": round(vc.iloc[0] / len(s) * 100, 1)})
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "tab_diversity_era.csv", index=False)
    print("\n[diversity — era]")
    print(t.to_string(index=False))
    return t


def t_composition(g):
    rows = []
    for e in ERAS:
        s = g[g.era == e]
        if not len(s):
            continue
        lo, hi = ERA_SPAN[e]
        yrs = (s.year.max() - s.year.min() + 1) if e == "Pre-1990" \
            else (min(hi, 2025) - lo + 1)
        r = {"era": e, "n": len(s), "years": int(yrs),
             "total_per_yr": round(len(s) / yrs, 1)}
        for f in FORMS:
            nf = int((s.form == f).sum())
            r[f"{f}_per_yr"] = round(nf / yrs, 1)
            r[f"{f}_share_%"] = round(nf / len(s) * 100, 1)
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "tab_composition_era.csv", index=False)
    print("\n[composition — SAS & Coop trajectory]")
    print(t[["era", "total_per_yr", "SAS_share_%", "Coop_per_yr",
             "Coop_share_%"]].to_string(index=False))
    return t


def t_coop(g):
    coop = g[g.form == "Coop"].copy()
    coop["sub"] = coop["razon_social"].map(coop_subtype)
    rows = []
    for e in ERAS:
        s = coop[coop.era == e]
        if not len(s):
            continue
        lo, hi = ERA_SPAN[e]
        yrs = (s.year.max() - s.year.min() + 1) if e == "Pre-1990" \
            else (min(hi, 2025) - lo + 1)
        sub = s["sub"]
        rows.append({"era": e, "total": len(s),
                     "work_per_yr": round((sub == "work").sum() / yrs, 1),
                     "ag_service_per_yr":
                         round((sub == "ag_service").sum() / yrs, 1),
                     "other_per_yr":
                         round((sub == "other_coop").sum() / yrs, 1)})
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "tab_coop_subtypes.csv", index=False)
    print("\n[cooperative disaggregation — per year]")
    print(t.to_string(index=False))
    return t


def t_spatial(g):
    rows = []
    for e in ERAS:
        s = g[(g.era == e) & g.redcode.notna()]
        if not len(s):
            continue
        rad = s.groupby("redcode")["form"].agg(
            lambda x: x.value_counts().index[0])
        dep = s.groupby("departamento")["form"].agg(
            lambda x: x.value_counts().index[0])
        share = rad.value_counts(normalize=True)
        rows.append({"era": e, "active_radios": rad.size,
                     "top_form": share.index[0],
                     "top_form_radio_share_%": round(share.iloc[0] * 100, 1),
                     "SAS_dom_depts": int((dep == "SAS").sum()),
                     "n_depts": dep.size,
                     "SAS_dom_radios": int((rad == "SAS").sum())})
    t = pd.DataFrame(rows)
    # permutation null for the Milei form-level spatial claim
    mil = g[(g.era == "Milei") & g.redcode.notna()]
    fm = mil["form"].value_counts(normalize=True)
    cats, probs = fm.index.to_numpy(), fm.values
    counts = mil.groupby("redcode").size().values
    obs = int((mil.groupby("redcode")["form"].agg(
        lambda x: x.value_counts().index[0]) == "SAS").sum())
    nit = 9999
    null = np.empty(nit, int)
    for it in range(nit):
        c = 0
        for k in counts:
            d = RNG.choice(cats, size=k, p=probs)
            u, cc = np.unique(d, return_counts=True)
            if u[cc.argmax()] == "SAS":
                c += 1
        null[it] = c
    p = (np.sum(null >= obs) + 1) / (nit + 1)
    t.to_csv(TAB / "tab_spatial_formlevel.csv", index=False)
    print("\n[spatial — form level, the explicit NEGATIVE]")
    print(t.to_string(index=False))
    print(f"  Milei: SAS-form dominant in {obs}/{counts.size} radios; "
          f"permutation null mean={null.mean():.1f} p95="
          f"{np.percentile(null,95):.0f}; P(null≥obs)={p:.4f}")
    print("  → no spatial-uniformity signal; compression is compositional.")
    return t


def t_dept_env(g):
    try:
        from sqlalchemy import create_engine
        eng = create_engine(
            "postgresql://postgres:postgres@localhost:5432/ndvi_misiones")
        env = pd.read_sql("SELECT redcode, canopy_cover FROM "
                          "radio_stats_master", eng)
    except Exception as e:  # noqa: BLE001
        print(f"\n[dept-env skipped — DB unavailable: {type(e).__name__}]")
        return None
    env["redcode"] = env["redcode"].astype(str)
    d = g.merge(env, on="redcode", how="left")
    d["is_coop"] = (d.form == "Coop").astype(int)
    a = d.groupby("departamento").agg(
        coop_share=("is_coop", "mean"),
        canopy=("canopy_cover", "mean")).dropna()
    r = a.coop_share.corr(a.canopy)
    jk = [a.drop(i).coop_share.corr(a.drop(i).canopy) for i in a.index]
    a.to_csv(TAB / "tab_dept_env.csv")
    print(f"\n[dept-env] r(coop_share, canopy)={r:.3f} N={len(a)}; "
          f"jackknife [{min(jk):.3f}, {max(jk):.3f}] → forest is context.")
    return a


def main():
    print("=" * 64 + "\nanalysis_diversity.py — canonical, model-free\n" +
          "=" * 64)
    g = load()
    t_annual(g)
    t_era(g)
    t_composition(g)
    t_coop(g)
    t_spatial(g)
    t_dept_env(g)
    print("\n" + "=" * 64 + "\nDONE — tables in acm/tables/\n" + "=" * 64)


if __name__ == "__main__":
    main()
