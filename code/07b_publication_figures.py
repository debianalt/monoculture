"""
07b — Publication-quality figures
==================================
300 DPI, TIFF + PNG. Consistent palette, font, style across all figures.
No titles inside figures — captions provided separately.

Figures:
    Fig1 — MCA biplot axes 1-2
    Fig2 — Era trajectory in MCA space
    Fig3 — Creation rates by cluster × era
    Fig4 — Mortality heatmap by cluster × era
    Fig5 — Spatial map of clusters
"""

import re
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import prince
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, linkage
from sqlalchemy import create_engine

warnings.filterwarnings("ignore")

ACM_DIR = Path(__file__).parent
PROJECT = ACM_DIR.parent
FIG_DIR = ACM_DIR / "figures"
DB_URL = "postgresql://postgres:postgres@localhost:5432/ndvi_misiones"

# ── Publication style ────────────────────────────────────────────────────────
DPI = 300
FONT_SIZE = 9
FONT_FAMILY = "Arial"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT_FAMILY, "Helvetica", "DejaVu Sans"],
    "font.size": FONT_SIZE,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Cluster names and consistent palette ─────────────────────────────────────
CLUSTER_NAMES = {
    1: "SAS (Milei/Fernández)",
    2: "Civil associations (cemetery)",
    3: "Cooperatives (frontier)",
    4: "Commercial SRL/SA",
    5: "Institutional services",
}
CLUSTER_COLORS = {
    1: "#e41a1c",   # red — SAS
    2: "#377eb8",   # blue — dead associations
    3: "#4daf4a",   # green — cooperatives/frontier
    4: "#ff7f00",   # orange — commercial
    5: "#984ea3",   # purple — services
}

ERAS_ORDERED = [
    "1. pre1990", "2. menem", "3. crisis", "4. n_kirchner",
    "5. c_kirchner", "6. macri", "7. fernandez", "8. milei",
]
ERA_LABELS = [
    "Pre-1990", "Menem", "Crisis", "N. Kirchner",
    "C. Kirchner", "Macri", "Fernández", "Milei",
]
ERA_YEARS = {
    "1. pre1990": 7, "2. menem": 10, "3. crisis": 3, "4. n_kirchner": 5,
    "5. c_kirchner": 8, "6. macri": 4, "7. fernandez": 4, "8. milei": 2,
}

# ── Data loading (same pipeline as 03/04) ────────────────────────────────────

SUBTIPO_KW = {
    "agro": ["AGROPECUAR", "AGRO ", "AGRICOL", "GANADER", "YERBA", "TABAC",
             "FORESTAL", "MADERA", "ASERRADERO"],
    "religious": ["IGLESIA", "EVANGELICA", "PASTORAL", "PARROQUIA", "TEMPLO",
                  "CRISTIANA", "ADVENTISTA", "BAUTISTA", "PENTECOSTAL",
                  "ASAMBLEA DE DIOS", "METODISTA", "LUTERANA", "MENONITA",
                  "CONGREGACION", "CULTO"],
    "sports": ["CLUB ", "DEPORTIV", "FUTBOL", "ATLETICO"],
    "education": ["ESCUELA", "COLEGIO", "INSTITUTO", "EDUCACI", "BIBLIOTECA"],
    "health": ["HOSPITAL", "CLINICA", "SANATORIO", "SALUD", "MEDIC", "FARMAC"],
    "transport": ["TRANSPORT", "REMIS", "TAXI", "COLECTIV", "CAMION"],
    "construction": ["CONSTRUC", "INMOBILIAR", "INMUEBLE", "VIVIENDA"],
    "commerce": ["COMERCI", "MERCADO", "SUPERMERCADO", "DISTRIBUID"],
    "tourism": ["TURIS", "HOTEL", "HOSTEL", "CABANA", "ALOJAMIENTO"],
}
RELIG_EXCL = re.compile(r"MISIONERA\s+S[.\s]|MISIONERO\s+S[.\s]|MISIONERAS\s+SA|ARENERA|AGUAS\s+MISION", re.I)


def classify_tipo(t):
    t = str(t).upper()
    if "COOPERATIVA" in t: return "Coop"
    if "ASOCIACION CIVIL" in t: return "Asoc"
    if "FUNDACION" in t: return "Fund"
    if "RESPONSABILIDAD LIMITADA" in t: return "SRL"
    if t == "SOCIEDAD ANONIMA": return "SA"
    if "ACCION SIMPLIFICADA" in t: return "SAS"
    return "Otra"


def classify_sub(rs):
    rs = str(rs).upper()
    for sub, kws in SUBTIPO_KW.items():
        if any(kw in rs for kw in kws):
            if sub == "religious" and RELIG_EXCL.search(rs): continue
            return sub
    return "other"


def political_era(y):
    if pd.isna(y): return "unknown"
    y = int(y)
    if y <= 1989: return "1. pre1990"
    if y <= 1999: return "2. menem"
    if y <= 2002: return "3. crisis"
    if y <= 2007: return "4. n_kirchner"
    if y <= 2015: return "5. c_kirchner"
    if y <= 2019: return "6. macri"
    if y <= 2023: return "7. fernandez"
    return "8. milei"


def load_mca():
    geo = pd.read_parquet(PROJECT / "data" / "geocoded_sociedades.parquet")
    geo["cuit"] = geo["cuit"].astype(str)
    geo["redcode"] = geo["redcode"].astype(str)
    geo["fecha"] = pd.to_datetime(geo["fecha_hora_contrato_social"], errors="coerce")
    geo["year"] = geo["fecha"].dt.year
    geo["era"] = geo["year"].map(political_era)
    geo["tipo"] = geo["tipo_societario"].map(classify_tipo)
    geo["subtipo"] = geo["razon_social"].map(classify_sub)
    geo["estado"] = geo["actividad_estado"].fillna("unknown")
    geo = geo[geo["era"] != "unknown"]

    act = geo[["tipo", "subtipo", "era", "departamento", "estado"]].copy()
    for col in act.columns:
        counts = act[col].value_counts()
        rare = counts[counts < 30].index
        if len(rare) > 0:
            act.loc[act[col].isin(rare), col] = "other_" + col

    mca = prince.MCA(n_components=5, random_state=42)
    mca = mca.fit(act)
    row_coords = mca.row_coordinates(act)
    row_coords.index = geo.index
    col_coords = mca.column_coordinates(act)

    Z = linkage(row_coords.values, method="ward")
    clusters = pd.Series(fcluster(Z, t=5, criterion="maxclust"), index=geo.index, name="cluster")
    geo["cluster"] = clusters

    # Benzécri
    raw_eigen = mca.eigenvalues_
    K = len(act.columns)
    threshold = 1.0 / K
    benz = [((K / (K - 1)) * (l - threshold)) ** 2 for l in raw_eigen if l > threshold]
    total_benz = sum(benz)
    benz_pcts = [b / total_benz * 100 for b in benz]

    return geo, row_coords, col_coords, clusters, benz_pcts


# ═════════════════════════════════════════════════════════════════════════════
# Fig 1 — MCA biplot
# ═════════════════════════════════════════════════════════════════════════════

def fig1_biplot(row_coords, col_coords, clusters, benz_pcts):
    fig, ax = plt.subplots(figsize=(7.5, 7))

    # Cloud: sample per cluster
    for cl in sorted(clusters.unique()):
        mask = clusters == cl
        sample = row_coords.loc[mask].sample(min(1200, mask.sum()), random_state=42)
        ax.scatter(sample.iloc[:, 0], sample.iloc[:, 1],
                   c=CLUSTER_COLORS[cl], s=3, alpha=0.08, rasterized=True)

    # Category labels — color-coded by variable type
    LABEL_COLORS = {
        "tipo": "#b30000",          # dark red
        "subtipo": "#006600",       # dark green
        "era": "#000099",           # dark blue
        "departamento": "#555555",  # grey
        "estado": "#993300",        # brown
    }

    def detect_var(label):
        s = str(label)
        tipos = ["SRL", "SA", "SAS", "Coop", "Asoc", "Fund", "Otra", "Mutual"]
        eras = ["pre1990", "menem", "crisis", "n_kirchner", "c_kirchner", "macri", "fernandez", "milei"]
        estados = ["AC", "BD", "unknown"]
        if any(s.endswith(t) for t in tipos): return "tipo"
        if any(s.endswith(e) for e in eras): return "era"
        if any(s.endswith(e) for e in estados): return "estado"
        if "sub" in s: return "subtipo"
        return "departamento"

    def clean_label(s):
        s = str(s)
        for p in ["tipo_", "subtipo_", "era_", "departamento_", "estado_", "other_", "sub_"]:
            s = s.replace(p, "")
        renames = {"n_kirchner": "N.Kirchner", "c_kirchner": "C.Kirchner",
                   "fernandez": "Fernandez", "milei": "Milei", "menem": "Menem",
                   "pre1990": "Pre-1990", "construction": "construction",
                   "commerce": "commerce", "transport": "transport",
                   "education": "education", "religious": "religious",
                   "sports": "sports", "tourism": "tourism", "health": "health"}
        for old, new in renames.items():
            s = s.replace(old, new)
        return s

    contrib = col_coords.iloc[:, 0] ** 2 + col_coords.iloc[:, 1] ** 2
    top = contrib.nlargest(20).index

    for idx in col_coords.index:
        x, y = col_coords.loc[idx].iloc[0], col_coords.loc[idx].iloc[1]
        vt = detect_var(idx)
        color = LABEL_COLORS.get(vt, "#333")

        if idx in top:
            label = clean_label(idx)
            # Manual offsets for overlapping labels
            ox, oy = 0, 0
            if "sports" in label.lower():
                oy = 0.15
            if "educ" in label.lower():
                oy = -0.10

            ax.annotate(label, (x + ox, y + oy), fontsize=7, fontweight="bold",
                        color=color, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=color, alpha=0.85, lw=0.6),
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.4, alpha=0.5) if (ox != 0 or oy != 0) else None,
                        xytext=(x + ox, y + oy) if (ox == 0 and oy == 0) else None)
        ax.plot(x, y, "s", color=color, markersize=3, alpha=0.5)

    ax.axhline(0, color="#ddd", lw=0.5)
    ax.axvline(0, color="#ddd", lw=0.5)
    ax.set_xlabel(f"Axis 1 ({benz_pcts[0]:.1f}% corrected inertia)")
    ax.set_ylabel(f"Axis 2 ({benz_pcts[1]:.1f}% corrected inertia)")

    # Legend below plot — two rows
    from matplotlib.lines import Line2D
    cluster_handles = [mpatches.Patch(color=CLUSTER_COLORS[c], label=CLUSTER_NAMES[c])
                       for c in sorted(CLUSTER_COLORS)]
    var_handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#b30000", markersize=7, label="Juridical form"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#006600", markersize=7, label="Function"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#000099", markersize=7, label="Political era"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#555555", markersize=7, label="Department"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#993300", markersize=7, label="Fiscal status"),
    ]

    ax.legend(handles=cluster_handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=3, fontsize=7, framealpha=0.95, edgecolor="#ccc", title="Clusters", title_fontsize=7.5)

    plt.subplots_adjust(bottom=0.16)
    plt.savefig(FIG_DIR / "Fig1_biplot.tiff", dpi=DPI, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(FIG_DIR / "Fig1_biplot.png", dpi=DPI)
    plt.close()
    print("  Fig1_biplot OK", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# Fig 2 — Era trajectory
# ═════════════════════════════════════════════════════════════════════════════

def fig2_trajectory(geo, row_coords, clusters, benz_pcts):
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Background cloud
    for cl in sorted(clusters.unique()):
        mask = clusters == cl
        sample = row_coords.loc[mask].sample(min(1000, mask.sum()), random_state=42)
        ax.scatter(sample.iloc[:, 0], sample.iloc[:, 1],
                   c=CLUSTER_COLORS[cl], s=2, alpha=0.05, rasterized=True)

    # Compute centres
    centers = []
    for e in ERAS_ORDERED:
        mask = geo["era"] == e
        if mask.sum() > 0:
            centers.append({
                "era": e, "x": row_coords.loc[mask, 0].mean(), "y": row_coords.loc[mask, 1].mean(),
                "sx": row_coords.loc[mask, 0].std(), "sy": row_coords.loc[mask, 1].std(), "n": mask.sum(),
            })
    cdf = pd.DataFrame(centers)

    # Trajectory line + arrows
    xs, ys = cdf["x"].values, cdf["y"].values
    ax.plot(xs, ys, "k-", lw=2, alpha=0.5, zorder=5)
    for i in range(len(xs) - 1):
        ax.annotate("", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, alpha=0.4), zorder=6)

    # Points + ellipses
    cmap_era = plt.cm.plasma(np.linspace(0.1, 0.9, len(cdf)))
    offsets = [
        (0.05, 0.07), (0.05, -0.09), (-0.20, 0.07), (0.05, 0.07),
        (0.05, -0.09), (-0.20, -0.09), (0.05, 0.07), (0.05, -0.09),
    ]
    for i, (_, row) in enumerate(cdf.iterrows()):
        ax.scatter(row["x"], row["y"], c=[cmap_era[i]], s=120, zorder=10, edgecolor="black", linewidth=1.2)
        ell = mpatches.Ellipse((row["x"], row["y"]), row["sx"] * 1.2, row["sy"] * 1.2,
                                alpha=0.10, color=cmap_era[i], zorder=3)
        ax.add_patch(ell)
        dx, dy = offsets[i]
        ax.annotate(ERA_LABELS[i], (row["x"] + dx, row["y"] + dy), fontsize=8, fontweight="bold",
                    color=cmap_era[i], zorder=11,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#aaa", alpha=0.9, lw=0.5))

    ax.axhline(0, color="#ccc", lw=0.5)
    ax.axvline(0, color="#ccc", lw=0.5)
    ax.set_xlabel(f"Axis 1 ({benz_pcts[0]:.1f}% corrected inertia)")
    ax.set_ylabel(f"Axis 2 ({benz_pcts[1]:.1f}% corrected inertia)")
    plt.savefig(FIG_DIR / "Fig2_trajectory.tiff", dpi=DPI, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(FIG_DIR / "Fig2_trajectory.png", dpi=DPI)
    plt.close()
    print("  Fig2_trajectory OK", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# Fig 3 — Creation rates
# ═════════════════════════════════════════════════════════════════════════════

def fig3_creation(geo, clusters):
    ct = pd.crosstab(clusters, geo["era"])
    for e in ERAS_ORDERED:
        if e in ct.columns:
            ct[e] = ct[e] / ERA_YEARS.get(e, 1)
    ct = ct[[e for e in ERAS_ORDERED if e in ct.columns]]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(ERAS_ORDERED))
    width = 0.65
    bottoms = np.zeros(len(ERAS_ORDERED))

    for cl in sorted(clusters.unique()):
        vals = [ct.loc[cl, e] if e in ct.columns else 0 for e in ERAS_ORDERED]
        ax.bar(x, vals, width, bottom=bottoms, color=CLUSTER_COLORS[cl],
               label=CLUSTER_NAMES[cl], alpha=0.9, edgecolor="white", linewidth=0.3)
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(ERA_LABELS, rotation=45, ha="right")
    ax.set_ylabel("Organisations created per year")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.95, edgecolor="#ccc")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.savefig(FIG_DIR / "Fig3_creation.tiff", dpi=DPI, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(FIG_DIR / "Fig3_creation.png", dpi=DPI)
    plt.close()
    print("  Fig3_creation OK", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# Fig 4 — Mortality heatmap
# ═════════════════════════════════════════════════════════════════════════════

def fig4_mortality(geo, clusters):
    geo = geo.copy()
    geo["is_dead"] = geo["actividad_estado"] == "BD"
    geo["cluster"] = clusters
    mort = geo.groupby(["cluster", "era"]).agg(n=("cuit", "size"), dead=("is_dead", "sum")).reset_index()
    mort["mort_rate"] = mort["dead"] / mort["n"] * 100
    pivot = mort.pivot(index="cluster", columns="era", values="mort_rate").fillna(0)
    ordered = [e for e in ERAS_ORDERED if e in pivot.columns]
    pivot = pivot[ordered]
    pivot.index = [CLUSTER_NAMES.get(c, str(c)) for c in pivot.index]
    pivot.columns = ERA_LABELS[:len(ordered)]

    fig, ax = plt.subplots(figsize=(7, 3))
    sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".0f", ax=ax,
                linewidths=0.8, linecolor="white",
                cbar_kws={"label": "% fiscally cancelled", "shrink": 0.8},
                annot_kws={"size": 8, "weight": "bold"})
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.savefig(FIG_DIR / "Fig4_mortality.tiff", dpi=DPI, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(FIG_DIR / "Fig4_mortality.png", dpi=DPI)
    plt.close()
    print("  Fig4_mortality OK", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# Fig 5 — Cluster map
# ═════════════════════════════════════════════════════════════════════════════

def fig5_map(geo, clusters):
    engine = create_engine(DB_URL)
    radios = gpd.read_postgis("SELECT redcode, geom FROM radios_misiones", engine, geom_col="geom")
    deptos = gpd.read_postgis("SELECT nombre, geom FROM departamentos_misiones", engine, geom_col="geom")
    env = pd.read_sql("SELECT redcode, densidad_hab_km2, viirs_mean_radiance FROM radio_stats_master", engine)
    radios["redcode"] = radios["redcode"].astype(str)
    env["redcode"] = env["redcode"].astype(str)

    # Dominant cluster per radio
    geo_cl = geo.copy()
    geo_cl["cluster"] = clusters.values
    dom = geo_cl.groupby("redcode")["cluster"].agg(lambda x: x.mode().iloc[0]).reset_index()
    dom.columns = ["redcode", "cluster"]
    gdf = radios.merge(dom, on="redcode", how="left")

    # Exclude uninhabited radios (river islands, >10 km2 with density < 5)
    gdf = gdf.merge(env, on="redcode", how="left")
    island_mask = (gdf["densidad_hab_km2"] < 5) & (gdf.geometry.area > 0.002)  # ~20 km2 in degrees
    gdf.loc[island_mask, "cluster"] = pd.NA

    fig, ax = plt.subplots(figsize=(5, 7.5))

    # Background (all radios)
    gdf[gdf["cluster"].isna()].plot(ax=ax, color="#f5f5f5", edgecolor="#ddd", linewidth=0.1)

    # Coloured radios
    for cl in sorted(CLUSTER_COLORS):
        sub = gdf[gdf["cluster"] == cl]
        if len(sub) > 0:
            sub.plot(ax=ax, color=CLUSTER_COLORS[cl], edgecolor="#999", linewidth=0.15,
                     label=CLUSTER_NAMES[cl], alpha=0.85)

    # Department boundaries
    deptos.boundary.plot(ax=ax, color="#666", linewidth=0.6, linestyle="-")

    # Department labels
    for _, d in deptos.iterrows():
        c = d.geom.centroid
        name = d["nombre"]
        if name in ("Capital", "Iguazu", "Eldorado", "Obera", "San Pedro", "Guarani"):
            ax.annotate(name, (c.x, c.y), fontsize=6, ha="center", color="#333", style="italic")

    # Scale bar (approximate)
    ax.plot([-55.8, -55.4], [-28.0, -28.0], "k-", lw=1.5)
    ax.text(-55.6, -28.05, "~40 km", ha="center", fontsize=6)

    # North arrow
    ax.annotate("N", xy=(-53.7, -25.8), fontsize=9, fontweight="bold", ha="center")
    ax.annotate("", xy=(-53.7, -25.65), xytext=(-53.7, -25.8),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))

    ax.legend(fontsize=6, loc="lower left", framealpha=0.95, edgecolor="#ccc")
    ax.set_axis_off()
    plt.savefig(FIG_DIR / "Fig5_map.tiff", dpi=DPI, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(FIG_DIR / "Fig5_map.png", dpi=DPI)
    plt.close()
    print("  Fig5_map OK", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating publication figures (300 DPI, TIFF+PNG)...", flush=True)
    geo, row_coords, col_coords, clusters, benz_pcts = load_mca()

    fig1_biplot(row_coords, col_coords, clusters, benz_pcts)
    fig2_trajectory(geo, row_coords, clusters, benz_pcts)
    fig3_creation(geo, clusters)
    fig4_mortality(geo, clusters)
    fig5_map(geo, clusters)

    print("\nAll figures saved to acm/figures/", flush=True)
    print("  Fig1_biplot.tiff / .png")
    print("  Fig2_trajectory.tiff / .png")
    print("  Fig3_creation.tiff / .png")
    print("  Fig4_mortality.tiff / .png")
    print("  Fig5_map.tiff / .png")
