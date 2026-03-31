"""
03 — MCA of the organisational field of Misiones
==================================================
Unit: formal organisation (N~14,177 full / N~6,652 ARCA-enriched)
Active: categorical organisational attributes (juridical form, function,
        political era, employment status, fiscal regime)
Supplementary: continuous environmental/physical variables of each
               organisation's census radio

Corrections: Benzécri eigenvalue adjustment, silhouette for optimal k,
             correlation filter on supplementary variables, test-values.

Output:
    acm/figures/fig_mca_biplot_12.png
    acm/figures/fig_mca_biplot_23.png
    acm/figures/fig_mca_dendrogram.png
    acm/figures/fig_mca_cluster_map.png
    acm/figures/fig_mca_cluster_profile.png
    acm/figures/fig_mca_silhouette.png
    acm/tables/tab_eigenvalues.csv
    acm/tables/tab_contributions.csv
    acm/tables/tab_test_values.csv
    acm/tables/tab_cluster_profiles.csv
"""

import re
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prince
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.stats import norm as scipy_norm
from sklearn.metrics import silhouette_score, silhouette_samples
from sqlalchemy import create_engine

warnings.filterwarnings("ignore")

ACM_DIR = Path(__file__).parent
DATA_DIR = ACM_DIR / "data"
FIG_DIR = ACM_DIR / "figures"
TAB_DIR = ACM_DIR / "tables"
PROJECT = ACM_DIR.parent
DB_URL = "postgresql://postgres:postgres@localhost:5432/ndvi_misiones"

# ═════════════════════════════════════════════════════════════════════════════
# Data loading & classification
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


def load_and_prepare():
    """Load organisations, enrich, classify, merge environment."""
    geo = pd.read_parquet(PROJECT / "data" / "geocoded_sociedades.parquet")
    enr = pd.read_parquet(PROJECT / "data" / "enriquecido_arca_v2.parquet")
    engine = create_engine(DB_URL)
    env = pd.read_sql("SELECT * FROM radio_stats_master", engine)
    poi = pd.read_sql("SELECT redcode, dist_nearest_bank, dist_nearest_supermarket FROM osm_poi_commercial", engine)

    # Keys
    geo["cuit"] = geo["cuit"].astype(str)
    geo["redcode"] = geo["redcode"].astype(str)
    enr["cuit"] = enr["cuit"].astype(str)
    env["redcode"] = env["redcode"].astype(str)
    poi["redcode"] = poi["redcode"].astype(str)

    # Classify
    geo["tipo"] = geo["tipo_societario"].map(classify_tipo)
    geo["subtipo"] = geo["razon_social"].map(classify_subtipo)
    geo["fecha"] = pd.to_datetime(geo["fecha_hora_contrato_social"], errors="coerce")
    geo["year"] = geo["fecha"].dt.year
    geo["era"] = geo["year"].map(political_era)

    # Merge ARCA
    enr["viva_arca"] = enr["error"].isna()
    arca = enr[enr["viva_arca"]][["cuit", "es_empleador", "categoria_iva"]].copy()
    geo = geo.merge(arca, on="cuit", how="left")
    geo["has_arca"] = geo["es_empleador"].notna()

    # Merge environment (supplementary)
    supp_cols = [
        "redcode", "canopy_cover", "viirs_mean_radiance", "densidad_hab_km2",
        "pct_nbi", "pct_universitario", "travel_min_posadas", "travel_min_cabecera",
        "dist_nearest_anp_km", "vulnerability_score", "hansen_total_loss",
        "deforest_pressure_score", "road_density_km_per_km2", "elev_mean",
        "pct_agua_red", "pct_hacinamiento", "building_density_per_km2",
    ]
    env_sel = env[[c for c in supp_cols if c in env.columns]]
    geo = geo.merge(env_sel, on="redcode", how="left")
    geo = geo.merge(poi, on="redcode", how="left")

    return geo


def filter_supplementary_correlations(supp_df, threshold=0.7):
    """Remove highly correlated supplementary variables."""
    corr = supp_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        high = upper.index[upper[col] > threshold].tolist()
        if high:
            to_drop.add(col)
    if to_drop:
        print(f"  Supplementary dropped (|r|>{threshold}): {to_drop}", flush=True)
    return supp_df.drop(columns=to_drop)


# ═════════════════════════════════════════════════════════════════════════════
# Benzécri correction
# ═════════════════════════════════════════════════════════════════════════════

def benzecri_correction(eigenvalues, n_active_vars):
    """Benzécri (1979) correction for MCA eigenvalues."""
    K = n_active_vars
    threshold = 1.0 / K
    corrected = []
    for lam in eigenvalues:
        if lam > threshold:
            c = ((K / (K - 1)) * (lam - threshold)) ** 2
            corrected.append(c)
    total = sum(corrected)
    pcts = [c / total * 100 for c in corrected]
    return corrected, pcts


# ═════════════════════════════════════════════════════════════════════════════
# Test-values
# ═════════════════════════════════════════════════════════════════════════════

def compute_test_values(active_df, row_coords, n_axes=5):
    """Compute test-values (v.test) for each category on each axis."""
    N = len(active_df)
    results = []
    for col in active_df.columns:
        for cat in active_df[col].unique():
            mask = active_df[col] == cat
            n_k = mask.sum()
            if n_k < 2 or n_k > N - 2:
                continue
            for ax in range(min(n_axes, row_coords.shape[1])):
                mean_k = row_coords.iloc[:, ax][mask].mean()
                mean_all = row_coords.iloc[:, ax].mean()
                std_all = row_coords.iloc[:, ax].std()
                if std_all == 0:
                    continue
                vtest = (mean_k - mean_all) / (std_all * np.sqrt((N - n_k) / (n_k * (N - 1))))
                results.append({
                    "variable": col, "category": cat, "axis": ax + 1,
                    "n": n_k, "mean_cat": mean_k, "mean_all": mean_all,
                    "vtest": vtest, "p_value": 2 * (1 - scipy_norm.cdf(abs(vtest))),
                })
    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════════════════
# Silhouette analysis
# ═════════════════════════════════════════════════════════════════════════════

def find_optimal_k(coords, k_range=range(3, 9)):
    """Compute silhouette scores for different k values."""
    scores = {}
    Z = linkage(coords, method="ward")
    for k in k_range:
        labels = fcluster(Z, t=k, criterion="maxclust")
        s = silhouette_score(coords, labels)
        scores[k] = s
        print(f"  k={k}: silhouette={s:.3f}", flush=True)
    best_k = max(scores, key=scores.get)
    print(f"  Best k={best_k} (silhouette={scores[best_k]:.3f})", flush=True)
    return scores, best_k, Z


# ═════════════════════════════════════════════════════════════════════════════
# Figures
# ═════════════════════════════════════════════════════════════════════════════

def plot_biplot(row_coords, col_coords, clusters, supp_df, eigen_pcts,
                axes=(0, 1), filename="fig_mca_biplot_12.png", n_top_cats=25):
    ax1, ax2 = axes
    fig, ax = plt.subplots(figsize=(14, 11))

    # Cluster colours
    n_clusters = clusters.nunique()
    cmap = plt.cm.Set2(np.linspace(0, 1, n_clusters))

    # Plot organisations (sample if too many)
    plot_df = row_coords.copy()
    plot_df["cluster"] = clusters.values
    if len(plot_df) > 3000:
        plot_sample = plot_df.sample(3000, random_state=42)
    else:
        plot_sample = plot_df

    for i, cl in enumerate(sorted(clusters.unique())):
        mask = plot_sample["cluster"] == cl
        ax.scatter(plot_sample.loc[mask].iloc[:, ax1], plot_sample.loc[mask].iloc[:, ax2],
                   c=[cmap[i]], s=4, alpha=0.15, label=f"Cl.{cl} (N={int((clusters==cl).sum())})")

    # Plot category points — only top contributors
    contrib_ax1 = (col_coords.iloc[:, ax1] ** 2)
    contrib_ax2 = (col_coords.iloc[:, ax2] ** 2)
    combined = contrib_ax1 + contrib_ax2
    top_cats = combined.nlargest(n_top_cats).index

    for idx in col_coords.index:
        x, y = col_coords.loc[idx].iloc[ax1], col_coords.loc[idx].iloc[ax2]
        if idx in top_cats:
            ax.annotate(str(idx), (x, y), fontsize=7, fontweight="bold", color="black",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="wheat", ec="gray", alpha=0.85))
        ax.plot(x, y, "ks", markersize=3, alpha=0.4)

    # Supplementary arrows (correlation)
    for col in supp_df.columns:
        vals = supp_df[col].dropna()
        common = vals.index.intersection(row_coords.index)
        if len(common) < 100:
            continue
        v = vals.loc[common]
        v_norm = (v - v.mean()) / v.std()
        rc = row_coords.loc[common]
        corr_x = np.corrcoef(v_norm, rc.iloc[:, ax1])[0, 1]
        corr_y = np.corrcoef(v_norm, rc.iloc[:, ax2])[0, 1]
        mag = np.sqrt(corr_x**2 + corr_y**2)
        if mag > 0.15:
            scale = 1.2
            ax.annotate("", xy=(corr_x * scale, corr_y * scale), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color="steelblue", lw=1.3))
            label = col.replace("_", "\n").replace("pct\n", "% ")
            ax.text(corr_x * scale * 1.1, corr_y * scale * 1.1, label,
                    fontsize=6, color="steelblue", ha="center", style="italic")

    ax.axhline(0, color="gray", lw=0.4, ls="--")
    ax.axvline(0, color="gray", lw=0.4, ls="--")
    ax.set_xlabel(f"Axis {ax1+1} ({eigen_pcts[ax1]:.1f}% corrected inertia)")
    ax.set_ylabel(f"Axis {ax2+1} ({eigen_pcts[ax2]:.1f}% corrected inertia)")
    ax.set_title(f"MCA of the organisational field — Misiones (N={len(row_coords):,} organisations)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, markerscale=3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=250)
    plt.close()
    print(f"  Saved {filename}", flush=True)


def plot_silhouette(scores, filename="fig_mca_silhouette.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = sorted(scores.keys())
    vals = [scores[k] for k in ks]
    ax.bar(ks, vals, color="steelblue", alpha=0.8)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Optimal number of clusters — silhouette analysis")
    best = max(scores, key=scores.get)
    ax.bar(best, scores[best], color="darkorange")
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=200)
    plt.close()
    print(f"  Saved {filename}", flush=True)


def plot_dendrogram(Z, filename="fig_mca_dendrogram.png"):
    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(Z, truncate_mode="lastp", p=40, leaf_rotation=90, leaf_font_size=8, ax=ax)
    ax.set_title("Ward hierarchical clustering on MCA coordinates")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=200)
    plt.close()
    print(f"  Saved {filename}", flush=True)


def plot_cluster_map(geo_df, clusters, filename="fig_mca_cluster_map.png"):
    """Map showing dominant cluster per radio."""
    engine = create_engine(DB_URL)
    radios = gpd.read_postgis("SELECT redcode, geom FROM radios_misiones", engine, geom_col="geom")
    radios["redcode"] = radios["redcode"].astype(str)

    # Dominant cluster per radio (mode)
    geo_df = geo_df.copy()
    geo_df["cluster"] = clusters.values
    dom = geo_df.groupby("redcode")["cluster"].agg(lambda x: x.mode().iloc[0]).reset_index()
    dom.columns = ["redcode", "cluster"]

    gdf = radios.merge(dom, on="redcode", how="left")

    fig, ax = plt.subplots(figsize=(8, 12))
    gdf[gdf["cluster"].isna()].plot(ax=ax, color="#f0f0f0", edgecolor="#ddd", linewidth=0.15)
    gdf[gdf["cluster"].notna()].plot(ax=ax, column="cluster", categorical=True,
                                      cmap="Set2", edgecolor="gray", linewidth=0.15,
                                      legend=True, legend_kwds={"title": "Cluster", "fontsize": 8})
    ax.set_title(f"Dominant organisational cluster per radio")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=200)
    plt.close()
    print(f"  Saved {filename}", flush=True)


def plot_cluster_profiles(geo_df, clusters, supp_cols, filename="fig_mca_cluster_profile.png"):
    geo_df = geo_df.copy()
    geo_df["cluster"] = clusters.values
    means = geo_df.groupby("cluster")[supp_cols].mean()
    g_mean = geo_df[supp_cols].mean()
    g_std = geo_df[supp_cols].std()
    z = (means - g_mean) / g_std

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(z.T, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                linewidths=0.5, ax=ax, vmin=-1.0, vmax=1.0,
                cbar_kws={"label": "Z-score"})
    ax.set_title("Cluster profiles — supplementary environmental variables (z-scores)")
    ax.set_xlabel("Cluster")
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=200)
    plt.close()
    z.to_csv(TAB_DIR / "tab_cluster_profiles.csv")
    print(f"  Saved {filename}", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("MCA — Organisational field of Misiones", flush=True)
    print("=" * 60, flush=True)

    # ── Load & prepare ──────────────────────────────────────────────────
    print("\n1. Loading and preparing data...", flush=True)
    geo = load_and_prepare()

    # ── Build active variables (organisation-level) ─────────────────────
    # Use FULL dataset (14K) with available categorical variables
    active_cols = ["tipo", "subtipo", "era"]

    # For ARCA-enriched subset, add fiscal variables
    # Decision: use full dataset for tipo/subtipo/era, project ARCA as supplementary category
    active = geo[active_cols].copy()

    # Add departamento (coarsened to reduce categories)
    active["departamento"] = geo["departamento"].fillna("unknown")

    # Add actividad_estado
    active["estado"] = geo["actividad_estado"].fillna("unknown")

    # Drop rows with unknown era
    active = active[active["era"] != "unknown"]
    geo_aligned = geo.loc[active.index]

    # Drop rare categories (< 30 occurrences in any variable)
    print(f"\n  Raw: {len(active)} organisations, {len(active.columns)} active variables", flush=True)
    for col in active.columns:
        counts = active[col].value_counts()
        rare = counts[counts < 30].index
        if len(rare) > 0:
            active.loc[active[col].isin(rare), col] = "other_" + col
            print(f"  {col}: merged {len(rare)} rare categories into 'other_{col}'", flush=True)

    print(f"\n  Active variable profiles:", flush=True)
    for col in active.columns:
        cats = active[col].value_counts()
        print(f"  {col} ({len(cats)} cats): {', '.join(f'{c}={n}' for c, n in cats.head(6).items())}", flush=True)

    # ── Supplementary variables ─────────────────────────────────────────
    supp_candidates = [
        "canopy_cover", "viirs_mean_radiance", "densidad_hab_km2",
        "pct_nbi", "pct_universitario", "travel_min_posadas",
        "dist_nearest_anp_km", "vulnerability_score", "hansen_total_loss",
        "deforest_pressure_score", "road_density_km_per_km2", "elev_mean",
        "pct_agua_red", "pct_hacinamiento", "building_density_per_km2",
        "dist_nearest_bank", "dist_nearest_supermarket", "travel_min_cabecera",
    ]
    supp_cols = [c for c in supp_candidates if c in geo_aligned.columns]
    supp_df = geo_aligned[supp_cols].copy()

    # Filter correlated supplementary variables
    print("\n2. Filtering correlated supplementary variables...", flush=True)
    supp_df = filter_supplementary_correlations(supp_df, threshold=0.7)
    supp_cols = list(supp_df.columns)
    print(f"  Kept {len(supp_cols)} supplementary variables", flush=True)

    # ── Run MCA ─────────────────────────────────────────────────────────
    print("\n3. Running MCA...", flush=True)
    n_components = 10
    mca = prince.MCA(n_components=n_components, random_state=42)
    mca = mca.fit(active)

    row_coords = mca.row_coordinates(active)
    row_coords.index = active.index
    col_coords = mca.column_coordinates(active)
    contrib = mca.column_contributions_

    # Eigenvalues + Benzécri
    raw_eigen = mca.eigenvalues_
    n_active = len(active.columns)
    benz_eigen, benz_pcts = benzecri_correction(raw_eigen, n_active)

    eigen_df = pd.DataFrame({
        "axis": range(1, len(raw_eigen) + 1),
        "raw_eigenvalue": raw_eigen,
        "raw_pct": [e / sum(raw_eigen) * 100 for e in raw_eigen],
    })
    # Add Benzécri columns (fewer axes)
    for i, (be, bp) in enumerate(zip(benz_eigen, benz_pcts)):
        if i < len(eigen_df):
            eigen_df.loc[i, "benzecri_eigenvalue"] = be
            eigen_df.loc[i, "benzecri_pct"] = bp
    eigen_df["benzecri_cumulative"] = eigen_df["benzecri_pct"].cumsum()
    eigen_df.to_csv(TAB_DIR / "tab_eigenvalues.csv", index=False)
    contrib.to_csv(TAB_DIR / "tab_contributions.csv")

    print(f"\n  Eigenvalues (raw vs Benzécri corrected):", flush=True)
    for _, r in eigen_df.head(6).iterrows():
        raw_str = f"raw={r['raw_pct']:.1f}%"
        benz_str = f"Benz={r['benzecri_pct']:.1f}%" if pd.notna(r.get("benzecri_pct")) else ""
        cum_str = f"cum={r['benzecri_cumulative']:.1f}%" if pd.notna(r.get("benzecri_cumulative")) else ""
        print(f"  Axis {r['axis']:.0f}: {raw_str:12s} {benz_str:12s} {cum_str}", flush=True)

    # ── Test-values ─────────────────────────────────────────────────────
    print("\n4. Computing test-values...", flush=True)
    tv = compute_test_values(active, row_coords, n_axes=5)
    tv.to_csv(TAB_DIR / "tab_test_values.csv", index=False)
    # Show significant categories per axis
    for ax_n in range(1, 4):
        sig = tv[(tv["axis"] == ax_n) & (tv["vtest"].abs() > 2.58)].sort_values("vtest")
        print(f"\n  Axis {ax_n} — significant categories (|v.test| > 2.58):", flush=True)
        for _, r in sig.head(5).iterrows():
            print(f"    {r['category']:25s} v={r['vtest']:+.2f} (n={r['n']:.0f})", flush=True)
        print(f"    ...", flush=True)
        for _, r in sig.tail(5).iterrows():
            print(f"    {r['category']:25s} v={r['vtest']:+.2f} (n={r['n']:.0f})", flush=True)

    # ── Silhouette analysis ─────────────────────────────────────────────
    print("\n5. Silhouette analysis for optimal k...", flush=True)
    n_axes_hac = min(5, len(benz_eigen))
    coords_hac = row_coords.iloc[:, :n_axes_hac].values

    # Sample for silhouette if too large
    if len(coords_hac) > 10000:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(coords_hac), 10000, replace=False)
        coords_sample = coords_hac[sample_idx]
    else:
        coords_sample = coords_hac

    sil_scores, best_k, Z = find_optimal_k(coords_sample, k_range=range(3, 9))
    plot_silhouette(sil_scores)

    # ── HAC with optimal k ──────────────────────────────────────────────
    print(f"\n6. Running HAC with k={best_k}...", flush=True)
    Z_full = linkage(coords_hac, method="ward")
    clusters = pd.Series(fcluster(Z_full, t=best_k, criterion="maxclust"),
                         index=active.index, name="cluster")
    for c in sorted(clusters.unique()):
        n = (clusters == c).sum()
        print(f"  Cluster {c}: {n:,} orgs ({n/len(clusters)*100:.1f}%)", flush=True)

    # ── Figures ─────────────────────────────────────────────────────────
    print("\n7. Generating figures...", flush=True)
    benz_pcts_full = eigen_df["benzecri_pct"].fillna(0).tolist()

    supp_df.index = active.index
    plot_biplot(row_coords, col_coords, clusters, supp_df, benz_pcts_full,
                axes=(0, 1), filename="fig_mca_biplot_12.png")
    plot_biplot(row_coords, col_coords, clusters, supp_df, benz_pcts_full,
                axes=(1, 2), filename="fig_mca_biplot_23.png")
    plot_dendrogram(Z_full)
    plot_cluster_map(geo_aligned, clusters)
    plot_cluster_profiles(geo_aligned, clusters, supp_cols)

    # ── Active variable profiles per cluster ────────────────────────────
    print("\n8. Active variable profiles per cluster...", flush=True)
    geo_aligned_cl = geo_aligned.copy()
    geo_aligned_cl["cluster"] = clusters.values

    for col in active.columns:
        ct = pd.crosstab(geo_aligned_cl["cluster"], active[col], normalize="index") * 100
        print(f"\n  {col}:", flush=True)
        print(ct.round(1).to_string(), flush=True)

    # Save cluster assignments
    geo_aligned_cl[["cuit", "redcode", "cluster"]].to_csv(
        DATA_DIR / "org_cluster_assignments.csv", index=False
    )

    print("\n" + "=" * 60, flush=True)
    print("DONE", flush=True)
    print("=" * 60, flush=True)
