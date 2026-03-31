"""
04 — Temporal extension of MCA: era trajectories + cluster dynamics
====================================================================
1. Centre of gravity per political era in MCA space
2. Concentration ellipses per era
3. Creation rates by cluster × era
4. Mortality rates by cluster × era
"""

import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import prince
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, linkage

warnings.filterwarnings("ignore")

ACM_DIR = Path(__file__).parent
PROJECT = ACM_DIR.parent
FIG_DIR = ACM_DIR / "figures"
TAB_DIR = ACM_DIR / "tables"

ERAS_ORDERED = [
    "1. pre1990", "2. menem", "3. crisis", "4. n_kirchner",
    "5. c_kirchner", "6. macri", "7. fernandez", "8. milei",
]
ERA_LABELS = [
    "Pre-1990", "Menem", "Crisis", "N.Kirchner",
    "C.Kirchner", "Macri", "Fernandez", "Milei",
]
ERA_YEARS = {
    "1. pre1990": 7, "2. menem": 10, "3. crisis": 3, "4. n_kirchner": 5,
    "5. c_kirchner": 8, "6. macri": 4, "7. fernandez": 4, "8. milei": 2,
}

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
RELIG_EXCL = re.compile(
    r"MISIONERA\s+S[.\s]|MISIONERO\s+S[.\s]|MISIONERAS\s+SA|ARENERA|AGUAS\s+MISION", re.I
)


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
            if sub == "religious" and RELIG_EXCL.search(rs):
                continue
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


def load_and_run_mca():
    """Rebuild MCA at organisation level (same as 03)."""
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

    Z = linkage(row_coords.values, method="ward")
    clusters = pd.Series(fcluster(Z, t=5, criterion="maxclust"), index=geo.index, name="cluster")
    geo["cluster"] = clusters

    return geo, row_coords, clusters, mca


# ═════════════════════════════════════════════════════════════════════════════
# 1. Era trajectory in MCA space
# ═════════════════════════════════════════════════════════════════════════════

def era_trajectory(geo, row_coords, clusters):
    print("\n=== CENTRES OF GRAVITY BY POLITICAL ERA ===", flush=True)

    centers = []
    for e in ERAS_ORDERED:
        mask = geo["era"] == e
        if mask.sum() > 0:
            cx = row_coords.loc[mask, 0].mean()
            cy = row_coords.loc[mask, 1].mean()
            n = mask.sum()
            sx = row_coords.loc[mask, 0].std()
            sy = row_coords.loc[mask, 1].std()
            centers.append({"era": e, "x": cx, "y": cy, "n": n, "sx": sx, "sy": sy})
            print(f"  {e:15s} N={n:5d} center=({cx:+.3f}, {cy:+.3f})", flush=True)

    centers_df = pd.DataFrame(centers)
    centers_df.to_csv(TAB_DIR / "tab_era_centres.csv", index=False)

    # ── Figure ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 9))

    # Background cloud
    cmap = plt.cm.Set2(np.linspace(0, 1, 5))
    for i, cl in enumerate(sorted(clusters.unique())):
        mask = clusters == cl
        sample_idx = row_coords.loc[mask].sample(min(1500, mask.sum()), random_state=42).index
        ax.scatter(row_coords.loc[sample_idx, 0], row_coords.loc[sample_idx, 1],
                   c=[cmap[i]], s=2, alpha=0.06)

    # Trajectory
    xs, ys = centers_df["x"].values, centers_df["y"].values
    ax.plot(xs, ys, "k-", lw=2.5, alpha=0.6, zorder=5)

    # Arrows
    for i in range(len(xs) - 1):
        ax.annotate("", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8, alpha=0.5), zorder=6)

    # Points + ellipses + labels
    colors_era = plt.cm.plasma(np.linspace(0.1, 0.9, len(centers_df)))
    offsets = [(0.04, 0.06), (0.04, -0.07), (-0.18, 0.06), (0.04, 0.06),
              (0.04, -0.07), (-0.18, -0.07), (0.04, 0.06), (0.04, -0.07)]

    for i, (_, row) in enumerate(centers_df.iterrows()):
        ax.scatter(row["x"], row["y"], c=[colors_era[i]], s=180, zorder=10,
                   edgecolor="black", linewidth=1.5)
        ellipse = mpatches.Ellipse(
            (row["x"], row["y"]), row["sx"] * 1.5, row["sy"] * 1.5,
            alpha=0.12, color=colors_era[i], zorder=3,
        )
        ax.add_patch(ellipse)
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(ERA_LABELS[i], (row["x"] + dx, row["y"] + dy),
                    fontsize=10, fontweight="bold", color=colors_era[i], zorder=11,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85))

    ax.axhline(0, color="gray", lw=0.4, ls="--")
    ax.axvline(0, color="gray", lw=0.4, ls="--")
    ax.set_xlabel("Axis 1 — institutional vitality (active/new vs dead/old)")
    ax.set_ylabel("Axis 2 — juridical form (SRL-SA/established vs Coop-SAS/emerging)")
    ax.set_title("Trajectory of the organisational field across political eras\nMisiones, 1983-2025")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_mca_era_trajectory.png", dpi=250)
    plt.close()
    print("  Saved fig_mca_era_trajectory.png", flush=True)

    return centers_df


# ═════════════════════════════════════════════════════════════════════════════
# 2. Creation rates by cluster × era
# ═════════════════════════════════════════════════════════════════════════════

def cluster_timeseries(geo, clusters):
    print("\n=== CREATION RATES BY CLUSTER AND ERA ===", flush=True)

    ct = pd.crosstab(clusters, geo["era"])
    # Annual rate
    for e in ERAS_ORDERED:
        if e in ct.columns:
            ct[e] = ct[e] / ERA_YEARS.get(e, 1)

    ct = ct[[e for e in ERAS_ORDERED if e in ct.columns]]
    print(ct.round(0).to_string(), flush=True)
    ct.to_csv(TAB_DIR / "tab_cluster_creation_rates.csv")

    # ── Figure: stacked bars ────────────────────────────────────────────
    cmap = plt.cm.Set2(np.linspace(0, 1, 5))
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(ERAS_ORDERED))
    bottoms = np.zeros(len(ERAS_ORDERED))

    for i, cl in enumerate(sorted(clusters.unique())):
        vals = [ct.loc[cl, e] if e in ct.columns else 0 for e in ERAS_ORDERED]
        ax.bar(x, vals, bottom=bottoms, color=cmap[i], label=f"Cluster {cl}", alpha=0.85, width=0.75)
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(ERA_LABELS, rotation=45, ha="right")
    ax.set_ylabel("Organisations created per year")
    ax.set_title("Annual creation rate by organisational cluster and political era")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_mca_cluster_timeseries.png", dpi=200)
    plt.close()
    print("  Saved fig_mca_cluster_timeseries.png", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Mortality by cluster × era
# ═════════════════════════════════════════════════════════════════════════════

def mortality_analysis(geo, clusters):
    print("\n=== MORTALITY RATE (% BD) BY CLUSTER AND ERA ===", flush=True)

    geo = geo.copy()
    geo["is_dead"] = geo["actividad_estado"] == "BD"
    geo["cluster"] = clusters

    mort = geo.groupby(["cluster", "era"]).agg(
        n=("cuit", "size"), dead=("is_dead", "sum")
    ).reset_index()
    mort["mort_rate"] = mort["dead"] / mort["n"] * 100

    pivot = mort.pivot(index="cluster", columns="era", values="mort_rate").fillna(0)
    ordered = [e for e in ERAS_ORDERED if e in pivot.columns]
    pivot = pivot[ordered]
    print(pivot.round(1).to_string(), flush=True)
    pivot.to_csv(TAB_DIR / "tab_mortality_by_cluster_era.csv")

    # ── Figure ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(
        pivot.rename(columns=dict(zip(ERAS_ORDERED, ERA_LABELS))),
        cmap="YlOrRd", annot=True, fmt=".0f", ax=ax, linewidths=0.5,
        cbar_kws={"label": "% dead (BD)"},
    )
    ax.set_title("Organisational mortality rate (% baja definitiva) by cluster and political era")
    ax.set_ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_mca_mortality_heatmap.png", dpi=200)
    plt.close()
    print("  Saved fig_mca_mortality_heatmap.png", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("Temporal extension of MCA", flush=True)
    print("=" * 60, flush=True)

    geo, row_coords, clusters, mca = load_and_run_mca()
    era_trajectory(geo, row_coords, clusters)
    cluster_timeseries(geo, clusters)
    mortality_analysis(geo, clusters)

    print("\n" + "=" * 60, flush=True)
    print("DONE", flush=True)
    print("=" * 60, flush=True)
