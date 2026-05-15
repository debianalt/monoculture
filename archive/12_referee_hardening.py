"""
12 — Referee-hardening analyses (critiques #1, #2, #5, #6, #7, #8, #3)
=====================================================================
Self-contained. Reads the FROZEN canonical partition
(acm/data/org_cluster_assignments_v2.csv) so every number is computed on
exactly the artifact the manuscript reports. Console + acm/tables/tab_R*.csv.

  #1  permutation null for the spatial-uniformity claim (180/180 radios)
  #2  within-data decomposition: absolute vs relative, all-form collapse
  #5  cooperative disaggregation: work vs agricultural/service
  #6  fiscal-status-supplementary robustness MCA (endogeneity)
  #7  optimal-matching cost-scheme sensitivity (k=2/k=3 stability)
  #8  duration-normalised diversity (annual + rolling, not era bins)
  #3  department-level environmental correlation + jackknife (needs DB)
"""
import importlib.util
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
import prince
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)

PROJECT = Path(__file__).resolve().parents[2]
ACM = PROJECT / "acm"
TAB = ACM / "tables"
TAB.mkdir(parents=True, exist_ok=True)
V2 = ACM / "data" / "org_cluster_assignments_v2.csv"

NAME = {1: "SAS", 2: "Assoc", 3: "Coop", 4: "Commercial", 5: "Services",
        6: "Services"}
ERAS = ["pre1990", "menem", "crisis", "n_kirchner", "c_kirchner", "macri",
        "fernandez", "milei"]
ERA_SPAN = {"pre1990": (1901, 1989), "menem": (1990, 1999),
            "crisis": (2000, 2002), "n_kirchner": (2003, 2007),
            "c_kirchner": (2008, 2015), "macri": (2016, 2019),
            "fernandez": (2020, 2023), "milei": (2024, 2025)}


def era_of(y):
    if pd.isna(y):
        return None
    y = int(y)
    for e, (lo, hi) in ERA_SPAN.items():
        if lo <= y <= hi:
            return e
    return None


def load():
    g = pd.read_parquet(PROJECT / "data" / "geocoded_sociedades.parquet")
    g["cuit"] = g["cuit"].astype(str)
    g["redcode"] = g["redcode"].astype(str)
    g["year"] = pd.to_datetime(g["fecha_hora_contrato_social"],
                               errors="coerce").dt.year
    g["era"] = g["year"].map(era_of)
    g["razon_social"] = g["razon_social"].astype(str)
    g["tipo_societario"] = g["tipo_societario"].astype(str)
    v2 = pd.read_csv(V2)
    v2["cuit"] = v2["cuit"].astype(str)
    df = g.merge(v2[["cuit", "cluster"]], on="cuit", how="inner")
    df["cl"] = df["cluster"].map(NAME)
    return df


# ── #1 ───────────────────────────────────────────────────────────────────────
def c1_permutation_null(df, n_iter=9999):
    print("\n" + "=" * 64 + "\n#1  Spatial-uniformity permutation null\n" + "=" * 64)
    mil = df[df["era"] == "milei"].copy()
    per_radio = mil.groupby("redcode").size()
    n_radios = per_radio.size
    obs_dom = (mil.groupby("redcode")["cl"]
               .agg(lambda s: s.value_counts().index[0]))
    obs_sas = int((obs_dom == "SAS").sum())
    print(f"  radios with >=1 Milei creation : {n_radios}")
    print(f"  observed SAS-dominant radios   : {obs_sas}/{n_radios}")

    # Null model: SAS is simply the modal national form in 2024-25.
    # For each radio draw its k_i creations i.i.d. from the empirical
    # province-wide Milei cluster distribution; recompute SAS-dominant count.
    p = mil["cl"].value_counts(normalize=True)
    cats, probs = p.index.to_numpy(), p.values
    sas_share = float(p.get("SAS", 0.0))
    counts = per_radio.values
    null = np.empty(n_iter, dtype=int)
    for it in range(n_iter):
        s = 0
        for k in counts:
            draw = RNG.choice(cats, size=k, p=probs)
            vals, cnt = np.unique(draw, return_counts=True)
            if vals[cnt.argmax()] == "SAS":
                s += 1
        null[it] = s
    p_emp = (np.sum(null >= obs_sas) + 1) / (n_iter + 1)
    print(f"  province-wide Milei SAS share  : {sas_share:.3f}")
    print(f"  null SAS-dominant radios       : mean={null.mean():.1f} "
          f"p5={np.percentile(null,5):.0f} p95={np.percentile(null,95):.0f} "
          f"max={null.max()}")
    print(f"  P(null >= observed)            : {p_emp:.4f}")
    verdict = ("MECHANICAL — null reproduces ~100% routinely; the spatial-"
               "uniformity claim is not distinguishable from sparse draws."
               if null.mean() >= 0.95 * obs_sas else
               "INFORMATIVE — observed exceeds the null expectation.")
    print(f"  VERDICT: {verdict}")
    pd.DataFrame({"metric": ["n_radios", "obs_sas_dominant",
                             "milei_sas_share", "null_mean", "null_p95",
                             "null_max", "p_empirical"],
                  "value": [n_radios, obs_sas, round(sas_share, 4),
                            round(float(null.mean()), 2),
                            float(np.percentile(null, 95)),
                            int(null.max()), round(p_emp, 4)]}
                 ).to_csv(TAB / "tab_R1_spatial_null.csv", index=False)


# ── #2 ───────────────────────────────────────────────────────────────────────
def c2_decomposition(df):
    print("\n" + "=" * 64 + "\n#2  Within-data decomposition (abs vs relative)\n" + "=" * 64)
    rows = []
    for e in ERAS:
        sub = df[df["era"] == e]
        if not len(sub):
            continue
        lo, hi = ERA_SPAN[e]
        yrs = (min(hi, 2025) - lo + 1) if e != "pre1990" else (
            sub["year"].max() - sub["year"].min() + 1)
        tot = len(sub)
        sas = (sub["cl"] == "SAS").sum()
        coop = (sub["cl"] == "Coop").sum()
        rows.append({"era": e, "n": tot, "years": yrs,
                     "total_per_yr": round(tot / yrs, 1),
                     "sas_per_yr": round(sas / yrs, 1),
                     "coop_per_yr": round(coop / yrs, 1),
                     "sas_share_%": round(sas / tot * 100, 1),
                     "coop_share_%": round(coop / tot * 100, 1)})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    t.to_csv(TAB / "tab_R2_decomposition.csv", index=False)
    mac, fer, mil = (t[t.era == x].iloc[0] for x in
                     ["macri", "fernandez", "milei"])
    print(f"\n  total creations/yr: Macri={mac.total_per_yr} "
          f"Fernandez={fer.total_per_yr} Milei={mil.total_per_yr}  "
          f"(general registration collapse?)")
    print(f"  SAS share: Macri={mac['sas_share_%']}% "
          f"Fernandez={fer['sas_share_%']}% Milei={mil['sas_share_%']}%  "
          f"(secular S-curve vs targeted?)")
    print("  → If total/yr collapses AND SAS share already rising pre-Milei, "
          "the mechanism reads 'associated with', not 'produced by'.")


# ── #5 ───────────────────────────────────────────────────────────────────────
WORK = ["TRABAJO"]
AGSV = ["AGROPECUARIA", "AGRICOLA", "AGRICOLAS", "YERBA", "TABACAL", "TABACO",
        "FORESTAL", "TAMBO", "GANADER", "PRODUCTORES", "COLONOS", "TE ",
        "SERVICIOS", "PROVISION", "OBRAS Y SERVICIOS", "PUBLICOS",
        "CITRICOL", "APICOL", "HORTICOL"]


def coop_subtype(rs):
    rs = rs.upper()
    if any(k in rs for k in AGSV):
        return "ag_service"
    if any(k in rs for k in WORK):
        return "work"
    return "other_coop"


def c5_coop_disaggregation(df):
    print("\n" + "=" * 64 + "\n#5  Cooperative disaggregation\n" + "=" * 64)
    coop = df[df["cl"] == "Coop"].copy()
    coop["sub"] = coop["razon_social"].map(coop_subtype)
    yr = {"n_kirchner": 5, "macri": 4, "fernandez": 4, "milei": 2}
    rows = []
    for e in ["n_kirchner", "c_kirchner", "macri", "fernandez", "milei"]:
        s = coop[coop["era"] == e]
        span = (8 if e == "c_kirchner" else yr.get(e, 1))
        rows.append({"era": e,
                     "work/yr": round((s["sub"] == "work").sum() / span, 1),
                     "ag_service/yr": round(
                         (s["sub"] == "ag_service").sum() / span, 1),
                     "other/yr": round(
                         (s["sub"] == "other_coop").sum() / span, 1),
                     "total/yr": round(len(s) / span, 1)})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    t.to_csv(TAB / "tab_R5_coop_subtypes.csv", index=False)
    f = t[t.era == "fernandez"].iloc[0]
    m = t[t.era == "milei"].iloc[0]
    print(f"\n  Fernandez->Milei: work {f['work/yr']}->{m['work/yr']} | "
          f"ag/service {f['ag_service/yr']}->{m['ag_service/yr']} /yr.")
    print("  → headline '114->1' must be split: governance argument rests "
          "on the ag/service subset, not the work-coop programme vehicles.")


# ── #6 ───────────────────────────────────────────────────────────────────────
def c6_fiscal_supplementary():
    print("\n" + "=" * 64 + "\n#6  Fiscal-status-supplementary robustness MCA\n" + "=" * 64)
    spec = importlib.util.spec_from_file_location(
        "_c", Path(__file__).with_name("04_run_acm.py"))
    canon = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canon)
    geo = canon.load_active()
    v2 = pd.read_csv(V2); v2["cuit"] = v2["cuit"].astype(str)

    def run(active_cols):
        a = geo[active_cols].copy()
        for c in a.columns:
            vc = a[c].value_counts(); r = vc[vc < 30].index
            if len(r):
                a.loc[a[c].isin(r), c] = "other_" + c
        m = prince.MCA(n_components=10, random_state=42).fit(a)
        rc = m.row_coordinates(a).reset_index(drop=True)
        ev = np.asarray(m.eigenvalues_)
        Z = linkage(rc.iloc[:, :5].values, method="ward")
        lab = canon.relabel_clusters(geo, fcluster(Z, t=6, criterion="maxclust"))
        return rc, ev, lab

    rc_a, ev_a, lab_a = run(["tipo", "subtipo", "era", "estado"])   # canonical
    rc_s, ev_s, lab_s = run(["tipo", "subtipo", "era"])             # fiscal OUT
    ba = canon.benzecri_correction(ev_a, 4)[1][:3]
    bs = canon.benzecri_correction(ev_s, 3)[1][:3]
    print(f"  Benzecri% canonical (fiscal active) : "
          f"{[round(x,1) for x in ba]}")
    print(f"  Benzecri% fiscal supplementary      : "
          f"{[round(x,1) for x in bs]}")
    for ax in range(3):
        r = abs(np.corrcoef(rc_a.iloc[:, ax], rc_s.iloc[:, ax])[0, 1])
        print(f"  axis {ax+1}: |r|(active vs fiscal-out) = {r:.4f}")
    da = pd.Series(lab_a).map(NAME).value_counts()
    ds = pd.Series(lab_s).map(NAME).value_counts()
    print(f"  cluster sizes canonical (fiscal active): {da.to_dict()}")
    print(f"  cluster sizes fiscal supplementary     : {ds.to_dict()}")
    agr = (pd.Series(lab_a) == pd.Series(lab_s)).mean()
    print(f"  per-record label agreement active vs fiscal-out: {agr:.3f}")
    print(f"  SAS / Coop clusters still recovered without fiscal status: "
          f"SAS n={int(ds.get('SAS',0))}, Coop n={int(ds.get('Coop',0))}  "
          f"(canonical: SAS {int(da.get('SAS',0))}, Coop "
          f"{int(da.get('Coop',0))})")
    pd.DataFrame({"axis": [1, 2, 3],
                  "benz_canonical": [round(x, 1) for x in ba],
                  "benz_fiscal_supp": [round(x, 1) for x in bs]}
                 ).to_csv(TAB / "tab_R6_fiscal_supp.csv", index=False)


# ── #7 ───────────────────────────────────────────────────────────────────────
def c7_om_cost_sensitivity(df):
    print("\n" + "=" * 64 + "\n#7  OM cost-scheme sensitivity\n" + "=" * 64)
    spec = importlib.util.spec_from_file_location(
        "_om", Path(__file__).with_name("06b_om_sequences.py"))
    om = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(om)
    seq = om.load_sequences()
    depts = list(seq.index)
    S = [list(seq.loc[d, om.ERA_ORDER]) for d in depts]

    def matrix(sub, indel):
        n = len(S); M = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = om.nw_distance(S[i], S[j], sub_cost=sub, indel_cost=indel)
                M[i, j] = M[j, i] = d
        return M

    def part(M, k):
        Z = linkage(squareform(M, checks=False), method="ward")
        return fcluster(Z, t=k, criterion="maxclust")

    base = matrix(2.0, 1.0)
    b2, b3 = part(base, 2), part(base, 3)
    print("  scheme                         ARI(k=2)  ARI(k=3)  sil(k=2)")
    rows = []
    for tag, sub, ind in [("baseline sub2/indel1", 2.0, 1.0),
                          ("sub2/indel0.5", 2.0, 0.5),
                          ("sub2/indel2", 2.0, 2.0),
                          ("sub1/indel1", 1.0, 1.0),
                          ("Hamming(no indel~sub2/indel9)", 2.0, 9.0)]:
        M = matrix(sub, ind)
        a2, a3 = part(M, 2), part(M, 3)
        ari2 = adjusted_rand_score(b2, a2)
        ari3 = adjusted_rand_score(b3, a3)
        sil2 = silhouette_score(M, a2, metric="precomputed")
        print(f"  {tag:30s}  {ari2:+.3f}    {ari3:+.3f}    {sil2:.3f}")
        rows.append({"scheme": tag, "ARI_k2": round(ari2, 3),
                     "ARI_k3": round(ari3, 3), "sil_k2": round(sil2, 3)})
    pd.DataFrame(rows).to_csv(TAB / "tab_R7_om_costs.csv", index=False)
    print("  → ARI≈1 ⇒ the 6-dept cooperative-presence split is cost-robust.")


# ── #8 ───────────────────────────────────────────────────────────────────────
def c8_duration_normalised(df):
    print("\n" + "=" * 64 + "\n#8  Duration-normalised diversity\n" + "=" * 64)
    d = df[df["year"].notna()].copy()
    s = importlib.util.spec_from_file_location(
        "_a", Path(__file__).with_name("04_run_acm.py"))
    cn = importlib.util.module_from_spec(s); s.loader.exec_module(cn)
    d["form"] = d["tipo_societario"].map(cn.classify_tipo)

    def H(series):
        p = series.value_counts(normalize=True).values
        return float(-(p * np.log(p)).sum())

    ann = (d[d.year >= 1990].groupby("year")["form"]
           .apply(H).rename("H").reset_index())
    ann["roll5"] = ann["H"].rolling(5, center=True, min_periods=3).mean()
    ann["dH"] = ann["H"].diff()
    print(ann.tail(12).to_string(index=False))
    worst = ann.loc[ann["dH"].idxmin()]
    milei_mean = ann[ann.year >= 2024]["H"].mean()
    pre = ann[(ann.year >= 2016) & (ann.year <= 2023)]["H"].mean()
    print(f"\n  largest single-YEAR ΔH drop: {worst['dH']:.3f} at "
          f"{int(worst['year'])}")
    print(f"  mean annual H 2016-23={pre:.3f}  vs  2024-25={milei_mean:.3f}  "
          f"(Δ={milei_mean-pre:+.3f})")
    print("  → 'steepest single-era decline' must be restated on a "
          "duration-comparable (annual) basis.")
    ann.to_csv(TAB / "tab_R8_annual_shannon.csv", index=False)


# ── #3 ───────────────────────────────────────────────────────────────────────
def c3_dept_env_jackknife(df):
    print("\n" + "=" * 64 + "\n#3  Dept-level env correlation + jackknife\n" + "=" * 64)
    try:
        from sqlalchemy import create_engine
        eng = create_engine(
            "postgresql://postgres:postgres@localhost:5432/ndvi_misiones")
        env = pd.read_sql("SELECT redcode, canopy_cover, "
                          "deforest_pressure_score, dist_nearest_anp_km "
                          "FROM radio_stats_master", eng)
    except Exception as e:
        print(f"  [skipped — DB unavailable: {type(e).__name__}]")
        return
    env["redcode"] = env["redcode"].astype(str)
    d = df.merge(env, on="redcode", how="left")
    d["coop_share"] = (d["cl"] == "Coop").astype(int)
    g = d.groupby("departamento").agg(
        coop_share=("coop_share", "mean"),
        canopy=("canopy_cover", "mean")).dropna()
    r_full = g["coop_share"].corr(g["canopy"])
    jk = [g.drop(idx)["coop_share"].corr(g.drop(idx)["canopy"])
          for idx in g.index]
    print(f"  dept-level r(coop_share, canopy) = {r_full:.3f}  (N={len(g)})")
    print(f"  leave-one-dept-out range: [{min(jk):.3f}, {max(jk):.3f}]")
    print("  → confirms aggregation-gain / MAUP fragility: report at dept "
          "scale with jackknife, do not over-claim micro alignment.")
    pd.DataFrame({"departamento": g.index, "coop_share": g.coop_share,
                  "canopy": g.canopy}).to_csv(
        TAB / "tab_R3_dept_env.csv", index=False)


def main():
    df = load()
    print(f"loaded {len(df)} orgs (frozen v2 partition)")
    c1_permutation_null(df)
    c2_decomposition(df)
    c5_coop_disaggregation(df)
    c8_duration_normalised(df)
    c3_dept_env_jackknife(df)
    c7_om_cost_sensitivity(df)
    c6_fiscal_supplementary()
    print("\nDONE — tables in acm/tables/tab_R*.csv")


if __name__ == "__main__":
    main()
