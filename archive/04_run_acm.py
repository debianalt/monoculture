"""
04 — MCA of the organisational field of Misiones  (CANONICAL spec)
==================================================================
Reproduces the partition consumed by the manuscript
(``acm/data/org_cluster_assignments_v2.csv``).

Specification (manuscript §4.4–§4.5):
  Unit            : individual organisation, N = 13,905
  Active (4)      : juridical form (tipo), functional subtype (subtipo),
                    political era (era), fiscal status (estado: active /
                    cancelled)
  Supplementary   : department (projected, NOT active — avoids constructing
                    the territorial gradient by design); + 13 continuous
                    environmental covariates (require the PostGIS DB; the
                    active pipeline and the cluster export run WITHOUT it).
  Exclusions      : 109 missing incorporation dates + 264 unknown fiscal
                    status  →  N = 13,905
  Clustering      : Ward on the first 5 MCA axes, k = 6; the two
                    institutional-services clusters (foundations +
                    archaic / state-linked forms) are reported merged as a
                    single "Institutional services" cluster (k = 5 in text),
                    following §4.5.

Reproducibility note
--------------------
``org_cluster_assignments_v2.csv`` is the FROZEN canonical derived dataset;
every downstream table and figure is computed from it and therefore
reproduces exactly. Regeneration from raw under the pinned environment
(``requirements.txt``) matches the frozen file at ≈99.4% of records; the
residual is library-version sensitivity in the MCA SVD and Ward tie-breaking.
The headline spatial-uniformity claims (17/17 departments and 180/180
census radios SAS-dominant under Milei) are invariant to this residual.
Department-as-active is a separate specification — see
``04b_sensitivity_dept_active.py``; it leaves axes 1–2 substantially
unchanged (|r| ≈ 0.99, 0.92) but does alter the finer k = 6 membership,
which is why department is projected as supplementary here and the
dominant-cluster results rely on modal assignment (§4.5).

``org_cluster_assignments_v2.csv`` is treated as IMMUTABLE here: it is the
frozen artifact every downstream script consumes. This script regenerates a
*verification copy* and reports agreement against the frozen file; it never
overwrites it. To intentionally re-freeze (only when the spec itself changes),
copy the regenerated file over the canonical name by hand.

Output (DB-free):
    acm/data/org_cluster_assignments_v2.regenerated.csv  (verification copy)
    acm/tables/tab_eigenvalues.csv
    acm/tables/tab_test_values.csv
    acm/tables/tab_dept_supplementary.csv        (department as supplementary)
Output (only if the PostGIS DB is reachable):
    acm/tables/tab_cluster_profiles.csv
    acm/figures/fig_mca_biplot_12.png  (+ _23, dendrogram, cluster map)
"""

import re
import sys
import warnings
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prince
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.stats import norm as scipy_norm
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[2]
ACM_DIR = PROJECT / "acm"
DATA_DIR = ACM_DIR / "data"
FIG_DIR = ACM_DIR / "figures"
TAB_DIR = ACM_DIR / "tables"
for d in (DATA_DIR, FIG_DIR, TAB_DIR):
    d.mkdir(parents=True, exist_ok=True)

DB_URL = "postgresql://postgres:postgres@localhost:5432/ndvi_misiones"
RANDOM_STATE = 42
N_CLUSTERS = 6           # k=6; clusters 5+6 reported merged (§4.5)
N_AXES_HAC = 5           # Ward on first 5 MCA axes

# ═════════════════════════════════════════════════════════════════════════════
# Classification (juridical form, functional subtype, political era)
# ═════════════════════════════════════════════════════════════════════════════

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
    r"MISIONERA\s+S[.\s]|MISIONERO\s+S[.\s]|MISIONERAS\s+SA|ARENERA|AGUAS\s+MISION",
    re.IGNORECASE,
)


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


# ═════════════════════════════════════════════════════════════════════════════
# Data loading (parquet only — no database needed for the canonical partition)
# ═════════════════════════════════════════════════════════════════════════════

def load_active():
    """Return the N=13,905 frame with the 4 active variables + keys.

    Exclusions, in order, matching §4.2:
      * missing incorporation date  (era == 'unknown')   → 109
      * unknown fiscal status       (actividad_estado NA) → 264
    """
    geo = pd.read_parquet(PROJECT / "data" / "geocoded_sociedades.parquet")
    enr = pd.read_parquet(PROJECT / "data" / "enriquecido_arca_v2.parquet")
    geo["cuit"] = geo["cuit"].astype(str)
    geo["redcode"] = geo["redcode"].astype(str)
    enr["cuit"] = enr["cuit"].astype(str)

    geo["tipo"] = geo["tipo_societario"].map(classify_tipo)
    geo["subtipo"] = geo["razon_social"].map(classify_subtipo)
    geo["year"] = pd.to_datetime(
        geo["fecha_hora_contrato_social"], errors="coerce").dt.year
    geo["era"] = geo["year"].map(political_era)

    enr["viva_arca"] = enr["error"].isna()
    geo = geo.merge(enr[["cuit", "viva_arca", "es_empleador", "categoria_iva"]],
                    on="cuit", how="left")

    n0 = len(geo)
    geo = geo[geo["era"] != "unknown"]
    n1 = len(geo)
    geo = geo[geo["actividad_estado"].notna()]
    n2 = len(geo)
    # fiscal status as a 2-category active variable (active vs cancelled)
    geo["estado"] = np.where(geo["actividad_estado"] == "BD",
                             "cancelled", "active")
    geo["departamento"] = geo["departamento"].fillna("unknown")

    print(f"  registry rows (geocoded)        : {n0}")
    print(f"  − missing incorporation date    : {n0 - n1}")
    print(f"  − unknown fiscal status         : {n1 - n2}")
    print(f"  = N (analysis)                  : {n2}")
    return geo.reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# Benzécri correction & test-values  (unchanged from the original deposit)
# ═════════════════════════════════════════════════════════════════════════════

def benzecri_correction(eigenvalues, n_active_vars):
    K = n_active_vars
    threshold = 1.0 / K
    corrected = [((K / (K - 1)) * (lam - threshold)) ** 2
                 for lam in eigenvalues if lam > threshold]
    total = sum(corrected)
    return corrected, [c / total * 100 for c in corrected]


def compute_test_values(active_df, row_coords, n_axes=5):
    N = len(active_df)
    out = []
    for col in active_df.columns:
        for cat in active_df[col].unique():
            mask = (active_df[col] == cat).values
            n_k = int(mask.sum())
            if n_k < 2 or n_k > N - 2:
                continue
            for ax in range(min(n_axes, row_coords.shape[1])):
                col_ax = row_coords.iloc[:, ax].values
                std_all = col_ax.std()
                if std_all == 0:
                    continue
                vtest = ((col_ax[mask].mean() - col_ax.mean())
                         / (std_all * np.sqrt((N - n_k) / (n_k * (N - 1)))))
                out.append({"variable": col, "category": cat, "axis": ax + 1,
                            "n": n_k, "vtest": vtest,
                            "p_value": 2 * (1 - scipy_norm.cdf(abs(vtest)))})
    return pd.DataFrame(out)


# ═════════════════════════════════════════════════════════════════════════════
# Content-based, version-stable cluster labelling
# ═════════════════════════════════════════════════════════════════════════════

def relabel_clusters(geo, raw_labels):
    """Map the 6 Ward clusters to stable IDs by *content*, not Ward order.

    1 = SAS-dominated    2 = civil-association-dominated
    3 = cooperative-dominated   4 = commercial (SRL/SA) dominated
    5,6 = the two residual institutional-services clusters (foundations /
          archaic / state-linked); both map downstream to "Services"
          (V2_TO_NAME). 5 = smaller, 6 = larger, to mirror the frozen file.

    Content rules make the export invariant to Ward's arbitrary numbering
    across library versions.
    """
    df = pd.DataFrame({"raw": raw_labels, "tipo": geo["tipo"].values})
    share = (df.groupby("raw")["tipo"].value_counts(normalize=True)
               .unstack(fill_value=0.0))
    for f in ["SAS", "Coop", "Asoc", "SRL", "SA"]:
        if f not in share.columns:
            share[f] = 0.0
    share["commercial"] = share["SRL"] + share["SA"]

    mapping, taken = {}, set()

    def claim(metric_col, target_id):
        ranked = share[metric_col].sort_values(ascending=False)
        for raw in ranked.index:
            if raw not in taken:
                mapping[raw] = target_id
                taken.add(raw)
                return

    claim("SAS", 1)
    claim("Asoc", 2)
    claim("Coop", 3)
    claim("commercial", 4)
    # remaining two → Services, smaller=5, larger=6
    rest = [r for r in share.index if r not in taken]
    rest_sorted = sorted(rest, key=lambda r: (df["raw"] == r).sum())
    for r, cid in zip(rest_sorted, (5, 6)):
        mapping[r] = cid
    return np.array([mapping[r] for r in raw_labels])


# ═════════════════════════════════════════════════════════════════════════════
# Optional environmental supplementary projection + figures (need PostGIS)
# ═════════════════════════════════════════════════════════════════════════════

def try_environmental_projection(geo, row_coords, clusters, benz_pcts):
    try:
        from sqlalchemy import create_engine
        engine = create_engine(DB_URL)
        env = pd.read_sql("SELECT * FROM radio_stats_master", engine)
        poi = pd.read_sql(
            "SELECT redcode, dist_nearest_bank, dist_nearest_supermarket "
            "FROM osm_poi_commercial", engine)
    except Exception as exc:  # noqa: BLE001 — DB optional by design
        print(f"\n  [env projection skipped — DB unavailable: "
              f"{type(exc).__name__}]")
        print("  Canonical partition + eigen/test-value tables were still "
              "written; environmental Fig.2/Table S4 require the PostGIS DB.")
        return

    env["redcode"] = env["redcode"].astype(str)
    poi["redcode"] = poi["redcode"].astype(str)
    supp_cols = ["canopy_cover", "viirs_mean_radiance", "densidad_hab_km2",
                 "pct_nbi", "pct_universitario", "travel_min_posadas",
                 "dist_nearest_anp_km", "vulnerability_score",
                 "hansen_total_loss", "deforest_pressure_score",
                 "road_density_km_per_km2", "elev_mean", "pct_agua_red"]
    env_sel = env[["redcode"] + [c for c in supp_cols if c in env.columns]]
    g = geo.merge(env_sel, on="redcode", how="left").merge(
        poi, on="redcode", how="left")
    keep = [c for c in supp_cols if c in g.columns]
    z = (g.assign(cluster=clusters).groupby("cluster")[keep].mean()
         - g[keep].mean()) / g[keep].std()
    z.to_csv(TAB_DIR / "tab_cluster_profiles.csv")
    print(f"\n  env projection OK — wrote tab_cluster_profiles.csv "
          f"({len(keep)} covariates)")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("04 — Canonical MCA (4 active; department supplementary)")
    print("=" * 64)

    geo = load_active()
    active = geo[["tipo", "subtipo", "era", "estado"]].copy()
    for col in active.columns:
        vc = active[col].value_counts()
        rare = vc[vc < 30].index
        if len(rare):
            active.loc[active[col].isin(rare), col] = "other_" + col
            print(f"  {col}: merged {len(rare)} rare (<30) categories")

    print("\nRunning MCA (n_components=10)...")
    mca = prince.MCA(n_components=10, random_state=RANDOM_STATE).fit(active)
    row_coords = mca.row_coordinates(active).reset_index(drop=True)

    raw_eigen = np.asarray(mca.eigenvalues_)
    benz_eigen, benz_pcts = benzecri_correction(raw_eigen, active.shape[1])
    eigen_df = pd.DataFrame({
        "axis": range(1, len(raw_eigen) + 1),
        "raw_eigenvalue": raw_eigen,
        "raw_pct": [e / raw_eigen.sum() * 100 for e in raw_eigen],
    })
    for i, (be, bp) in enumerate(zip(benz_eigen, benz_pcts)):
        eigen_df.loc[i, "benzecri_eigenvalue"] = be
        eigen_df.loc[i, "benzecri_pct"] = bp
    eigen_df["benzecri_cumulative"] = eigen_df["benzecri_pct"].cumsum()
    eigen_df.to_csv(TAB_DIR / "tab_eigenvalues.csv", index=False)
    print("  Benzécri %: " + ", ".join(
        f"Ax{i+1}={p:.1f}" for i, p in enumerate(benz_pcts[:5])))

    tv = compute_test_values(active, row_coords, n_axes=5)
    tv.to_csv(TAB_DIR / "tab_test_values.csv", index=False)

    # Department as SUPPLEMENTARY: category coordinates = mean of member rows
    dept = geo["departamento"].values
    sup = (pd.DataFrame(row_coords.iloc[:, :5].values,
                        columns=[f"axis{i+1}" for i in range(5)])
           .assign(departamento=dept)
           .groupby("departamento").mean())
    sup.to_csv(TAB_DIR / "tab_dept_supplementary.csv")

    print(f"\nWard clustering on first {N_AXES_HAC} axes, k={N_CLUSTERS}...")
    Z = linkage(row_coords.iloc[:, :N_AXES_HAC].values, method="ward")
    raw_labels = fcluster(Z, t=N_CLUSTERS, criterion="maxclust")
    clusters = relabel_clusters(geo, raw_labels)

    try:
        sil = silhouette_score(
            row_coords.iloc[:, :N_AXES_HAC].sample(
                min(10000, len(row_coords)), random_state=RANDOM_STATE).values,
            pd.Series(raw_labels).sample(
                min(10000, len(raw_labels)), random_state=RANDOM_STATE).values)
        print(f"  silhouette (k={N_CLUSTERS}, sample) = {sil:.3f}")
    except Exception:
        pass

    sizes = pd.Series(clusters).value_counts().sort_index()
    print("  cluster sizes (canonical IDs):")
    names = {1: "SAS", 2: "Assoc", 3: "Coop", 4: "Commercial",
             5: "Services", 6: "Services"}
    for cid, n in sizes.items():
        print(f"    {cid} {names[cid]:11s} n={n}")
    serv = int(sizes.get(5, 0) + sizes.get(6, 0))
    print(f"    (5+6 reported merged → Institutional services n={serv})")

    out = pd.DataFrame({"cuit": geo["cuit"].values,
                        "redcode": geo["redcode"].values,
                        "cluster": clusters})
    regen_path = DATA_DIR / "org_cluster_assignments_v2.regenerated.csv"
    out.to_csv(regen_path, index=False)
    print(f"\n  wrote verification copy {regen_path} (N={len(out)})")

    # Self-check against the frozen canonical artifact (never overwritten here)
    frozen = DATA_DIR / "org_cluster_assignments_v2.csv"
    if frozen.exists():
        f = pd.read_csv(frozen); f["cuit"] = f["cuit"].astype(str)
        m = out.merge(f, on="cuit", suffixes=("_regen", "_frozen"))
        agr = (m["cluster_regen"] == m["cluster_frozen"]).mean()
        print(f"  agreement vs frozen org_cluster_assignments_v2.csv: "
              f"{agr:.4%}  (expected ≈99.4% under pinned env;")
        print(f"  headline 17/17 & 180/180 invariant — see "
              f"00b_phase0_position_checks.py)")

    try_environmental_projection(geo, row_coords, clusters, benz_pcts)
    print("\n" + "=" * 64 + "\nDONE\n" + "=" * 64)


if __name__ == "__main__":
    main()
