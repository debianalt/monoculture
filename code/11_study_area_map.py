"""Generate Fig1: Study area map — Misiones in regional and continental context.

Two-panel figure:
  A (left)  — South America locator with bounding-box rectangle around Misiones.
               set_aspect("equal") so the continent is not stretched.
               Zoom connector lines link the bbox corners to Panel B.
  B (right) — Regional detail at 10m resolution: country borders, admin-1 province
               outlines, Misiones province (highlighted), UPAF boundary, departments.
"""
import warnings
warnings.filterwarnings("ignore")

import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import ConnectionPatch
import numpy as np
import sqlalchemy
from shapely.geometry import box as shp_box

# ── Paths ─────────────────────────────────────────────────────────────────────
ACM_DIR = Path(__file__).parent
PROJECT = ACM_DIR.parent
FIG_DIR = PROJECT / "figures"
CACHE   = Path.home() / "AppData" / "Local" / "geodatasets" / "geodatasets" / "Cache"
CACHE.mkdir(parents=True, exist_ok=True)
DPI = 300

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":       9,
    "savefig.dpi":     DPI,
    "savefig.bbox":    "tight",
})

# ── Color palette ─────────────────────────────────────────────────────────────
C_OCEAN      = "#d0e8f5"   # ocean / water background
C_LAND_OTHER = "#e4e0d8"   # neighbouring countries
C_ARGENTINA  = "#d4cfc2"   # Argentina (slightly warmer)
C_MISIONES   = "#D55E00"   # Misiones — consistent with all other figures
C_UPAF_FILL  = "#b5d9a8"   # UPAF region fill
C_UPAF_EDGE  = "#2d7a2d"   # UPAF outline
C_BORDER     = "#999999"   # country borders
C_ADM1       = "#bbbbbb"   # admin-1 (provincial) borders — lighter than country
C_DEPT       = "#c8c0b0"   # department lines within Misiones
C_PANEL_BDR  = "#555555"   # panel frame


# ── Data loaders ──────────────────────────────────────────────────────────────

def cached_download(url: str, fname: str) -> Path:
    dest = CACHE / fname
    if not dest.exists():
        print(f"  Downloading {fname} ...", flush=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def load_countries_110m():
    """110m Natural Earth countries — Panel A locator only."""
    f = cached_download(
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip",
        "ne_110m_admin_0_countries.zip",
    )
    return gpd.read_file(f)


def load_countries_10m():
    """10m Natural Earth countries — Panel B detail."""
    f = cached_download(
        "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip",
        "ne_10m_admin_0_countries.zip",
    )
    return gpd.read_file(f)


def load_admin1_10m():
    """10m admin-1 states/provinces — provincial borders in Panel B."""
    try:
        f = cached_download(
            "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip",
            "ne_10m_admin_1_states_provinces.zip",
        )
        adm1 = gpd.read_file(f)
        # Keep only countries visible in the Panel B extent
        return adm1[adm1["admin"].isin(["Argentina", "Brazil", "Paraguay"])]
    except Exception as e:
        print(f"  Admin-1 load failed: {e}", flush=True)
        return None


def load_upaf():
    try:
        f = cached_download(
            "https://storage.googleapis.com/teow2016/Ecoregions2017.zip",
            "Ecoregions2017.zip",
        )
        eco = gpd.read_file(f)
        mask = eco["ECO_NAME"].str.contains("Alto Paran|Upper Paran", na=False, case=False)
        upaf = eco[mask]
        if len(upaf) == 0:
            print("  UPAF not found — skipping.", flush=True)
            return None
        return upaf.to_crs("EPSG:4326")
    except Exception as e:
        print(f"  UPAF load failed: {e}", flush=True)
        return None


def load_postgis():
    engine = sqlalchemy.create_engine("postgresql://postgres@localhost:5432/ndvi_misiones")
    depts = gpd.read_postgis(
        "SELECT nombre, geom FROM departamentos_misiones", engine, geom_col="geom"
    )
    return depts


# ── Helpers ───────────────────────────────────────────────────────────────────

def style_frame(ax):
    """Thin, clean panel border; no ticks."""
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.9)
        sp.set_color(C_PANEL_BDR)


def add_scale_bar(ax, x0, y0, length_deg, label, color="#333333"):
    """Scale bar with label above the bar, fully inside the panel."""
    ax.plot([x0, x0 + length_deg], [y0, y0],
            color=color, lw=2.0, solid_capstyle="butt", zorder=11)
    tick_h = 0.055
    for xv in [x0, x0 + length_deg]:
        ax.plot([xv, xv], [y0 - tick_h, y0 + tick_h], color=color, lw=1.5, zorder=11)
    ax.text(x0 + length_deg / 2, y0 + tick_h + 0.07, label,
            ha="center", va="bottom", fontsize=8, color=color, zorder=11)


def add_north_arrow(ax, x_frac=0.92, y_frac=0.10, size=0.07):
    kw = dict(xycoords="axes fraction", textcoords="axes fraction")
    ax.annotate("", xy=(x_frac, y_frac + size), xytext=(x_frac, y_frac),
                arrowprops=dict(arrowstyle="-|>", color="#333333",
                                lw=1.2, mutation_scale=10), **kw)
    ax.text(x_frac, y_frac + size + 0.03, "N",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#333333")


# ── Panel A — South America locator ──────────────────────────────────────────

def draw_panel_a(ax, countries, mis_env):
    """South America with correct aspect ratio and dashed bbox around Misiones.

    Returns the bbox corner coordinates (bx0, by0, bx1, by1) in data CRS so the
    caller can draw ConnectionPatch zoom lines to Panel B.
    """
    ax.set_facecolor(C_OCEAN)

    sa  = countries[countries["CONTINENT"] == "South America"]
    arg = countries[countries["ADMIN"] == "Argentina"]

    sa.plot(ax=ax,  color=C_LAND_OTHER, edgecolor="#bbbbbb", linewidth=0.3)
    arg.plot(ax=ax, color=C_ARGENTINA,  edgecolor=C_BORDER,  linewidth=0.55)

    # Dashed rectangle around Misiones
    minx, miny, maxx, maxy = mis_env.bounds
    pad_x, pad_y = 0.55, 0.45
    bx0 = minx - pad_x
    by0 = miny - pad_y
    bx1 = maxx + pad_x
    by1 = maxy + pad_y
    rect = mpatches.FancyBboxPatch(
        (bx0, by0),
        (bx1 - bx0),
        (by1 - by0),
        boxstyle="square,pad=0",
        linewidth=2.0, edgecolor=C_MISIONES,
        facecolor="none", linestyle="--", zorder=8,
    )
    ax.add_patch(rect)

    ax.set_xlim(-82, -34)
    ax.set_ylim(-56, 13)
    ax.set_aspect("equal")   # correct continental proportions — no stretching
    style_frame(ax)

    ax.text(0.5, 1.025, "A", transform=ax.transAxes,
            fontsize=13, fontweight="bold", ha="center", va="bottom")

    return bx0, by0, bx1, by1   # bbox corners for ConnectionPatch


# ── Panel B — Regional detail ─────────────────────────────────────────────────

def draw_panel_b(ax, countries_10m, admin1, depts, upaf):
    """Regional zoom at 10m resolution: Misiones in geographic and ecological context."""
    minx, miny, maxx, maxy = depts.total_bounds

    pad_lr, pad_tb = 1.3, 1.1
    X0 = minx - pad_lr
    X1 = maxx + pad_lr
    Y0 = miny - pad_tb
    Y1 = maxy + pad_tb

    ax.set_facecolor(C_OCEAN)
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.set_aspect("equal")

    # ── Country fill (10m) ────────────────────────────────────────────────────
    clip  = shp_box(X0 - 0.5, Y0 - 0.5, X1 + 0.5, Y1 + 0.5)
    sa    = countries_10m[countries_10m["CONTINENT"] == "South America"].clip(clip)
    arg   = sa[sa["ADMIN"] == "Argentina"]
    other = sa[sa["ADMIN"] != "Argentina"]

    other.plot(ax=ax, color=C_LAND_OTHER, edgecolor=C_BORDER, linewidth=0.5, zorder=1)
    arg.plot(ax=ax,   color=C_ARGENTINA,  edgecolor=C_BORDER, linewidth=0.6, zorder=2)

    # ── Admin-1 provincial/state borders (10m) ────────────────────────────────
    if admin1 is not None:
        try:
            adm1_clip = admin1.clip(shp_box(X0, Y0, X1, Y1))
            if len(adm1_clip) > 0:
                adm1_clip.plot(ax=ax, color="none", edgecolor=C_ADM1,
                               linewidth=0.45, linestyle="--", zorder=2)
        except Exception:
            pass

    # ── UPAF (clipped to panel) ───────────────────────────────────────────────
    if upaf is not None:
        upaf_clip = upaf.clip(shp_box(X0, Y0, X1, Y1))
        upaf_clip.plot(ax=ax, color=C_UPAF_FILL, edgecolor=C_UPAF_EDGE,
                       linewidth=0.9, alpha=0.65, zorder=3)

    # ── Misiones (province fill + dept lines) ────────────────────────────────
    depts.dissolve().plot(ax=ax, color=C_MISIONES, edgecolor="#222222",
                          linewidth=1.3, zorder=4)
    depts.plot(ax=ax, color="none", edgecolor=C_DEPT, linewidth=0.55, zorder=5)

    # ── Country labels ────────────────────────────────────────────────────────
    for name, (lx, ly) in [
        ("Brazil",    (-53.4, -24.85)),
        ("Paraguay",  (-56.85, -26.20)),
        ("Argentina", (-55.40, -29.00)),
    ]:
        if X0 < lx < X1 and Y0 < ly < Y1:
            ax.text(lx, ly, name, fontsize=9, ha="center", va="center",
                    color="#555555", style="italic", zorder=6)

    # ── "Misiones" label — black, at TRUE centroid of province polygon ────────
    province = depts.dissolve()
    centroid  = province.geometry.iloc[0].centroid
    ax.text(centroid.x, centroid.y, "Misiones",
            fontsize=11, fontweight="bold", ha="center", va="center",
            color="#111111", zorder=7)

    # ── UPAF label — centred inside the visible UPAF area in Brazil ──────────
    if upaf is not None:
        label_x = min(X1 - 0.6, -53.0)
        label_y = max(Y0 + 0.5, -25.2)
        ax.text(label_x, label_y,
                "Upper Paraná\nAtlantic Forest",
                fontsize=7.5, ha="center", va="center",
                color=C_UPAF_EDGE, style="italic",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
                zorder=8)

    # ── Scale bar — fully inside panel ───────────────────────────────────────
    sb_x = X0 + 0.35
    sb_y = Y0 + 0.55
    add_scale_bar(ax, sb_x, sb_y, 0.95, "100 km")

    # ── North arrow ───────────────────────────────────────────────────────────
    add_north_arrow(ax, x_frac=0.935, y_frac=0.07, size=0.075)

    style_frame(ax)
    ax.text(0.5, 1.025, "B", transform=ax.transAxes,
            fontsize=13, fontweight="bold", ha="center", va="bottom")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...", flush=True)
    countries_110m = load_countries_110m()
    countries_10m  = load_countries_10m()
    admin1         = load_admin1_10m()
    upaf           = load_upaf()
    depts          = load_postgis()

    try:
        mis_env = depts.geometry.union_all().envelope
    except AttributeError:
        mis_env = depts.geometry.unary_union.envelope

    print("Composing figure...", flush=True)
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=(13, 6.5),
        gridspec_kw={"width_ratios": [1, 1]},
        layout="constrained",
    )

    bbox_a = draw_panel_a(ax_a, countries_110m, mis_env)
    draw_panel_b(ax_b, countries_10m, admin1, depts, upaf)

    # ── Zoom connector lines: right corners of bbox in A → left edge of B ────
    # Both panels share the same CRS (EPSG:4326 / geographic degrees), so
    # coordsA/B="data" resolves correctly across axes.
    bx0, by0, bx1, by1 = bbox_a
    pb_x0 = ax_b.get_xlim()[0]
    pb_y0 = ax_b.get_ylim()[0]
    pb_y1 = ax_b.get_ylim()[1]

    for ya, yb in [(by1, pb_y1), (by0, pb_y0)]:
        con = ConnectionPatch(
            xyA=(bx1, ya), coordsA="data", axesA=ax_a,
            xyB=(pb_x0, yb), coordsB="data", axesB=ax_b,
            color=C_MISIONES, lw=0.85, linestyle="-",
            clip_on=False, zorder=20,
        )
        fig.add_artist(con)

    out_png  = FIG_DIR / "Fig1.png"
    out_tiff = FIG_DIR / "Fig1.tiff"
    plt.savefig(out_png,  dpi=DPI)
    plt.savefig(out_tiff, dpi=DPI, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.close()
    print(f"  Fig1 OK -> {out_png}", flush=True)


if __name__ == "__main__":
    main()
