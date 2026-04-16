"""Generate Fig4: Small-multiples choropleth of dominant regime by department × era,
with hatching patterns for grayscale compatibility and a South America orientation inset."""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sqlalchemy

ACM_DIR = Path(__file__).parent
PROJECT = ACM_DIR.parent
FIG_DIR = PROJECT / "figures"
DPI = 300

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "hatch.linewidth": 1.5,
})

ERA_ORDER = ["1.Pre1990", "2.Menem", "3.Crisis", "4.NK", "5.CK", "6.Macri", "7.Fern", "8.Milei"]
ERA_TITLES = ["Pre-1990", "Menem\n1990–99", "Crisis\n2000–02", "N. Kirchner\n2003–07",
              "C. Kirchner\n2008–15", "Macri\n2016–19", "Fernández\n2020–23", "Milei\n2024–25"]

REGIME_COLORS = {
    "Commercial": "#E69F00",
    "Coop": "#009E73",
    "Assoc": "#56B4E9",
    "Services": "#CC79A7",
    "SAS": "#D55E00",
    "None": "#CCCCCC",
}

REGIME_HATCHES = {
    "Commercial": "",
    "Coop":       "...",
    "Assoc":      "///",
    "Services":   "\\\\\\",
    "SAS":        "xxx",
    "None":       "",
}

REGIME_LABELS = {
    "Commercial": "Commercial",
    "Coop": "Cooperatives",
    "Assoc": "Associations",
    "Services": "Inst. services",
    "SAS": "SAS",
    "None": "No creations",
}

# Plot order: None first (background), then regimes on top
REGIME_PLOT_ORDER = ["None", "Commercial", "Assoc", "Coop", "Services", "SAS"]


def load_sequences():
    assignments = pd.read_csv(ACM_DIR / "data" / "org_cluster_assignments.csv")
    geo = pd.read_parquet(PROJECT / "data" / "geocoded_sociedades.parquet")
    geo["cuit"] = geo["cuit"].astype(str)
    assignments["cuit"] = assignments["cuit"].astype(str)
    merged = geo.merge(assignments[["cuit", "cluster"]], on="cuit", how="inner")
    merged["fecha"] = pd.to_datetime(merged["fecha_hora_contrato_social"], errors="coerce")
    merged["year"] = merged["fecha"].dt.year

    def era(y):
        if pd.isna(y): return None
        y = int(y)
        if y <= 1989: return "1.Pre1990"
        if y <= 1999: return "2.Menem"
        if y <= 2002: return "3.Crisis"
        if y <= 2007: return "4.NK"
        if y <= 2015: return "5.CK"
        if y <= 2019: return "6.Macri"
        if y <= 2023: return "7.Fern"
        return "8.Milei"

    cluster_names = {1: "SAS", 2: "Assoc", 3: "Coop", 4: "Commercial", 5: "Services"}
    merged["era"] = merged["year"].map(era)
    merged = merged[merged["era"].notna()]
    merged["regime"] = merged["cluster"].map(cluster_names)

    dept_era = merged.groupby(["departamento", "era", "regime"]).size().reset_index(name="n")
    idx = dept_era.groupby(["departamento", "era"])["n"].idxmax()
    dominant = dept_era.loc[idx, ["departamento", "era", "regime"]].copy()
    seq = dominant.pivot(index="departamento", columns="era", values="regime").fillna("None")
    seq = seq[[e for e in ERA_ORDER if e in seq.columns]]
    return seq


def load_geodata_for_inset():
    """Load South America land + country boundaries for the orientation inset.

    Tries three sources in order:
    1. Natural Earth 110m admin-0 countries (URL download — provides country borders)
    2. geodatasets naturalearth.land clipped to SA bbox (fallback, no country borders)
    Returns (sa_land, argentina_gdf_or_None, land_gdf).
    """
    from shapely.geometry import box as shp_box
    SA_BBOX = shp_box(-82, -56, -34, 13)

    # --- Attempt 1: countries with named polygons ---
    countries = None
    try:
        # Cache file locally so re-runs are instant
        import tempfile, os, urllib.request, zipfile
        cache_dir = Path.home() / "AppData" / "Local" / "geodatasets" / "geodatasets" / "Cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        countries_zip = cache_dir / "ne_110m_admin_0_countries.zip"
        if not countries_zip.exists():
            print("  Downloading Natural Earth countries...", flush=True)
            url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
            urllib.request.urlretrieve(url, countries_zip)
        countries = gpd.read_file(countries_zip)
    except Exception as e:
        print(f"  Countries download skipped: {e}", flush=True)

    if countries is not None:
        sa = countries[countries["CONTINENT"] == "South America"].copy()
        argentina = countries[countries["ADMIN"] == "Argentina"].copy()
        return sa, argentina

    # --- Fallback: land polygons only ---
    try:
        import geodatasets
        land = gpd.read_file(geodatasets.get_path("naturalearth.land"))
        sa = land.clip(SA_BBOX)
        return sa, None
    except Exception as e:
        print(f"  Warning: could not load land data for inset: {e}", flush=True)
        return None, None


def add_orientation_inset(fig, depts):
    """Add a polished South America orientation inset (lower-left of figure)."""
    sa, argentina = load_geodata_for_inset()
    if sa is None:
        print("  Inset skipped — no geodata available.", flush=True)
        return

    # Misiones province outline
    try:
        misiones_geom = depts.geometry.union_all()
    except AttributeError:
        misiones_geom = depts.geometry.unary_union
    misiones_s = gpd.GeoSeries([misiones_geom], crs=depts.crs).to_crs("EPSG:4326")
    mis_centroid = misiones_s.geometry.iloc[0].centroid

    # ── Inset axes: lower-left, overlays empty corner of bottom-left panel ──
    # Size chosen to be clearly legible at 300 DPI without covering map content
    ax_in = fig.add_axes([0.005, 0.085, 0.215, 0.345])

    # Ocean background
    ax_in.set_facecolor("#cfe2f0")

    # Land / countries
    if argentina is not None:
        # Other SA countries: light neutral gray
        other_sa = sa[sa["ADMIN"] != "Argentina"]
        other_sa.plot(ax=ax_in, color="#d8d8d8", edgecolor="#aaaaaa", linewidth=0.35)
        # Argentina: slightly warmer gray to distinguish it
        argentina.plot(ax=ax_in, color="#b8c4b8", edgecolor="#888888", linewidth=0.5)
    else:
        # Land-only fallback (no country borders)
        sa.plot(ax=ax_in, color="#d4d4d4", edgecolor="#aaaaaa", linewidth=0.35)

    # Misiones province: highlighted in SAS/frontier orange
    misiones_s.plot(ax=ax_in, color="#D55E00", edgecolor="#333333", linewidth=1.2, zorder=5)

    # Country / region labels
    ax_in.text(-64.5, -36.5, "Argentina", fontsize=9, ha="center", va="center",
               color="#333333", style="italic")
    ax_in.text(-53, -12, "Brazil", fontsize=7.5, ha="center", va="center",
               color="#777777", style="italic")

    # Misiones annotation with leader line
    ax_in.annotate(
        "Misiones",
        xy=(mis_centroid.x, mis_centroid.y),
        xytext=(mis_centroid.x - 16, mis_centroid.y - 9),
        fontsize=9, fontweight="bold", color="#111111",
        ha="right", va="center",
        arrowprops=dict(arrowstyle="-|>", color="#333333",
                        lw=0.9, mutation_scale=9),
        zorder=6,
    )

    # Map extent
    ax_in.set_xlim(-82, -34)
    ax_in.set_ylim(-56, 13)
    ax_in.set_aspect("equal")

    # Hide ticks/labels but keep spines for the border frame
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    for spine in ax_in.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("#555555")

    # North arrow (top-right corner of inset)
    ax_in.annotate("", xy=(0.90, 0.26), xytext=(0.90, 0.12),
                   xycoords="axes fraction", textcoords="axes fraction",
                   arrowprops=dict(arrowstyle="-|>", color="#333333",
                                   lw=1.0, mutation_scale=9))
    ax_in.text(0.90, 0.30, "N", transform=ax_in.transAxes,
               fontsize=9, fontweight="bold", ha="center", va="bottom", color="#333333")

    # Title inside the inset frame
    ax_in.set_title("Study area", fontsize=9, pad=4,
                    loc="center", color="#222222", fontweight="normal")


def main():
    print("Loading department boundaries...", flush=True)
    engine = sqlalchemy.create_engine("postgresql://postgres@localhost:5432/ndvi_misiones")
    depts = gpd.read_postgis("SELECT nombre, geom FROM departamentos_misiones", engine, geom_col="geom")

    print("Loading sequences...", flush=True)
    seq = load_sequences()

    # Normalize names for matching
    def norm(s):
        return s.lower().replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u")

    name_map = {}
    for db_name in depts["nombre"].values:
        for seq_name in seq.index:
            if norm(seq_name) in norm(db_name) or norm(db_name) in norm(seq_name):
                name_map[db_name] = seq_name

    depts["dept_key"] = depts["nombre"].map(name_map)
    unmatched = depts[depts["dept_key"].isna()]["nombre"].tolist()
    if unmatched:
        print(f"  Unmatched departments: {unmatched}")

    # Plot: 2 rows × 4 cols
    fig, axes = plt.subplots(2, 4, figsize=(12, 7))
    axes = axes.flatten()

    for i, (era, title) in enumerate(zip(ERA_ORDER, ERA_TITLES)):
        ax = axes[i]

        # Assign regime for this era to each department row
        def get_regime(row):
            key = row["dept_key"]
            if pd.isna(key) or key not in seq.index or era not in seq.columns:
                return "None"
            return seq.loc[key, era]

        depts["current_regime"] = depts.apply(get_regime, axis=1)

        # Plot per-regime to support hatching
        for regime in REGIME_PLOT_ORDER:
            sub = depts[depts["current_regime"] == regime]
            if len(sub) == 0:
                continue
            sub.plot(
                ax=ax,
                color=REGIME_COLORS.get(regime, "#CCCCCC"),
                hatch=REGIME_HATCHES.get(regime, ""),
                edgecolor="#333333",
                linewidth=0.4,
            )

        ax.set_title(title, fontsize=11, fontweight="bold", pad=3)
        ax.set_axis_off()
        ax.set_aspect("equal")

    # Legend
    regime_order = ["Commercial", "Assoc", "Coop", "Services", "SAS", "None"]
    used = set()
    for era in ERA_ORDER:
        if era in seq.columns:
            used.update(seq[era].unique())
    legend_regimes = [r for r in regime_order if r in used]

    with plt.rc_context({"hatch.linewidth": 1.2}):
        handles = [
            mpatches.Patch(
                facecolor=REGIME_COLORS[r],
                hatch=REGIME_HATCHES.get(r, ""),
                edgecolor="#555555",
                label=REGIME_LABELS.get(r, r),
            )
            for r in legend_regimes
        ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=len(legend_regimes), fontsize=11, framealpha=0.95, edgecolor="#ccc")

    plt.subplots_adjust(hspace=0.08, wspace=0.02, bottom=0.10)

    # Save as Fig4 (main text figure) and keep FigS1 as alias
    plt.savefig(FIG_DIR / "Fig4.png", dpi=DPI)
    plt.savefig(FIG_DIR / "Fig4.tiff", dpi=DPI, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(FIG_DIR / "FigS1.png", dpi=DPI)
    plt.savefig(ACM_DIR / "figures" / "FigS1_choropleth.png", dpi=DPI)
    plt.close()
    print(f"  Fig4 OK -> figures/Fig4.png (+ FigS1 alias)", flush=True)


if __name__ == "__main__":
    main()
