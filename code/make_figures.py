"""
make_figures.py — figures for the reconstructed (model-free) paper
==================================================================
Reads the canonical tables written by analysis_diversity.py and produces
Fig2–Fig5 (Fig1 study-area map is generated separately). Consistent style
across all figures per the manuscript figure conventions: identical sans
fonts and sizes, a single form→colour map, identical state-project order
and labels.

Colour: one fixed form→colour assignment used by every figure. The seven
categorical hues were validated for colour-vision deficiency with a
palette validator (worst adjacent pair in the Fig. 2 stacking order:
ΔE 21.2 under protanopia, target ≥ 12). "Other"/"Residual" are drawn in a
deliberately recessive grey — they are residual classes, not entities —
which is a documented exception to the chroma floor. The two hues below
3:1 contrast on white (aqua, yellow) rely on the legend and on the
supplementary tables, per the relief rule.

Outputs → <repo>/figures/Fig{2,3,4,5}.png  (300 dpi, for the manuscript)
        → <repo>/figures/Fig{2,3,4,5}.pdf  (vector, for typesetting:
          the journal requires EPS/PDF/TIFF post-acceptance)
"""
import sys
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

# Repository root: reads the canonical tables written by analysis_diversity.py
# and writes the publication figures back into the deposit.
REPO = Path(__file__).resolve().parents[1]
TAB = REPO / "tables"
FIGS = REPO / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300, "savefig.dpi": 300,
    "axes.linewidth": 0.6, "pdf.fonttype": 42,
})

INK = "#0b0b0b"        # primary ink (marks, tick labels)
SECONDARY = "#52514e"  # secondary ink (panel letters, period labels)
GRID = "#e1e0d9"       # hairline gridlines
BASELINE = "#c3c2b7"   # axis/baseline

ERAS = ["Pre-1990", "Menem", "Crisis", "N.Kirchner", "C.Kirchner",
        "Macri", "Fernández", "Milei"]
# Display labels: the CSV keys above are compact; axis ticks must read exactly
# as the period names do in the manuscript tables.
ERA_LABEL = {"N.Kirchner": "N. Kirchner", "C.Kirchner": "C. Kirchner"}
ERA_TICKS = [ERA_LABEL.get(e, e) for e in ERAS]
FORMS = ["SAS", "SRL", "SA", "Coop", "Asoc", "Fund", "Otra"]
FORM_LABEL = {"SAS": "SAS", "SRL": "SRL", "SA": "SA", "Coop": "Cooperative",
              "Asoc": "Civil association", "Fund": "Foundation",
              "Otra": "Other"}
# CVD-validated assignment (see module docstring). Grey = residual exception.
FORM_COLOR = {"SAS": "#2a78d6", "SRL": "#eb6834", "SA": "#4a3aa7",
              "Coop": "#e34948", "Asoc": "#1baf7a", "Fund": "#eda100",
              "Otra": "#898781"}


def _style(ax, grid_axis="y"):
    """Recessive chrome: hairline grid below the marks, no top/right spines."""
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK, which="both", labelcolor=INK)


def _panel_letter(ax, letter):
    ax.text(-0.075, 1.02, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", color=INK, va="bottom", ha="left")


def _save(fig, name):
    out = FIGS / name
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out}  (+ .pdf)")


def fig2_composition():
    d = pd.read_csv(TAB / "tab_composition_era.csv").set_index("era").reindex(ERAS)
    x = np.arange(len(ERAS))
    fig, (ax_rate, ax_share) = plt.subplots(
        2, 1, figsize=(7.2, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 2.4], "hspace": 0.12})

    # Panel A — registration rate with exact Poisson 95% CIs (one measure,
    # one axis: the former twin-axis overlay is split out, not rescaled).
    yerr = np.vstack([d["total_per_yr"] - d["total_per_yr_lo95"],
                      d["total_per_yr_hi95"] - d["total_per_yr"]])
    ax_rate.errorbar(x, d["total_per_yr"], yerr=yerr, fmt="-o", color=INK,
                     lw=1.1, ms=3.5, ecolor=INK, elinewidth=0.8, capsize=2.5)
    ax_rate.set_ylabel("New registrations\nper year")
    ax_rate.set_ylim(0, float(d["total_per_yr_hi95"].max()) * 1.12)
    _style(ax_rate)
    _panel_letter(ax_rate, "A")

    # Panel B — juridical-form shares, stacked to 100% with surface gaps.
    bottom = np.zeros(len(ERAS))
    for f in FORMS:
        sh = d[f"{f}_share_%"].values
        ax_share.bar(x, sh, bottom=bottom, color=FORM_COLOR[f],
                     label=FORM_LABEL[f], width=0.62,
                     edgecolor="white", linewidth=0.6)
        bottom += sh
    ax_share.set_ylabel("Share of new registrations (%)")
    ax_share.set_ylim(0, 100)
    ax_share.set_xticks(x)
    ax_share.set_xticklabels(ERA_TICKS, rotation=30, ha="right")
    _style(ax_share)
    _panel_letter(ax_share, "B")

    # Single legend for the whole figure, in stacking order (bottom→top).
    handles, labels = ax_share.get_legend_handles_labels()
    ax_rate.legend(handles=handles, labels=labels, ncol=4, loc="lower center",
                   bbox_to_anchor=(0.5, 1.12), frameon=False,
                   columnspacing=1.4, handlelength=1.4)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.86, bottom=0.13)
    _save(fig, "Fig2.png")


def fig3_annual_shannon():
    a = pd.read_csv(TAB / "tab_diversity_annual.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    # One quantity at two smoothings: one hue, two weights.
    ax.plot(a["year"], a["H"], color="#898781", lw=0.9, marker="o", ms=2.5,
            label="Annual $H$")
    ax.plot(a["year"], a["roll5"], color="#1c5cab", lw=2.2,
            label="5-year rolling mean")
    # State-project boundaries as solid hairlines, each period labelled so the
    # figure reads without the caption.
    bounds = [2000, 2003, 2008, 2016, 2020, 2024]
    for b in bounds:
        ax.axvline(b, color="#d9d8d2", ls="-", lw=0.7, zorder=0)
    spans = [(1990, 2000, "Menem"), (2000, 2003, "Crisis"),
             (2003, 2008, "N. Kirchner"), (2008, 2016, "C. Kirchner"),
             (2016, 2020, "Macri"), (2020, 2024, "Fernández"),
             (2024, 2025, "Milei")]
    for lo, hi, name in spans:
        ax.text((lo + hi) / 2, 0.975, name, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6.5, color=SECONDARY,
                clip_on=False)
    ax.set_xlabel("Year")
    ax.set_ylabel("Shannon diversity $H$")
    ax.set_xlim(a["year"].min(), a["year"].max())
    _style(ax)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    _save(fig, "Fig3.png")


def fig4_coop_subtypes():
    c = pd.read_csv(TAB / "tab_coop_subtypes.csv").set_index("era").reindex(ERAS)
    x = np.arange(len(ERAS))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    # Work keeps the cooperative red (it is the dominant cooperative subtype);
    # agricultural/service takes aqua — validated against red (ΔE 42.9 tritan,
    # 21.2 protan) — and the residual class recedes in grey.
    ax.bar(x - w, c["work_per_yr"], w, label="Work", color="#e34948",
           edgecolor="white", linewidth=0.6)
    ax.bar(x, c["ag_service_per_yr"], w, label="Agricultural / service",
           color="#1baf7a", edgecolor="white", linewidth=0.6)
    ax.bar(x + w, c["other_per_yr"], w, label="Residual", color="#898781",
           edgecolor="white", linewidth=0.6)
    ax.set_ylabel("Cooperative registrations per year")
    ax.set_xticks(x)
    ax.set_xticklabels(ERA_TICKS, rotation=30, ha="right")
    _style(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "Fig4.png")


def fig5_spatial():
    s = pd.read_csv(TAB / "tab_spatial_formlevel.csv").set_index("era").reindex(ERAS)
    x = np.arange(len(ERAS))
    colors = [FORM_COLOR.get(f, "#333") for f in s["top_form"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x, s["top_form_radio_share_%"], color=colors, width=0.6,
           edgecolor="white", linewidth=0.6)
    # Milei permutation null band (mean 87.3 of 190 active radios -> %)
    mil_active = float(s.loc["Milei", "active_radios"])
    null_mean_pct = 87.3 / mil_active * 100
    null_p95_pct = 98.0 / mil_active * 100
    mi = list(ERAS).index("Milei")
    # Bar labels use the same form names as the Fig. 2 legend. The Milei label
    # clears the error-bar cap rather than sitting under it.
    for i, (f, v) in enumerate(zip(s["top_form"], s["top_form_radio_share_%"])):
        top = max(v, null_p95_pct) if i == mi else v
        ax.text(i, top + 1.5, FORM_LABEL.get(f, f),
                ha="center", va="bottom", fontsize=7, color=INK)
    ax.errorbar([mi], [s.loc["Milei", "top_form_radio_share_%"]],
                yerr=[[s.loc["Milei", "top_form_radio_share_%"] - null_mean_pct],
                      [null_p95_pct - s.loc["Milei", "top_form_radio_share_%"]]],
                fmt="none", ecolor=INK, capsize=4, lw=1.2)
    # Point at the left edge of the error bar, so the arrow never crosses the label.
    ax.annotate("Milei permutation\nnull (mean, 95th pct)",
                xy=(mi - 0.34, null_mean_pct), xytext=(mi - 2.9, null_mean_pct + 16),
                fontsize=7, color=SECONDARY,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=SECONDARY))
    ax.set_ylabel("Modal-form share of active census radios (%)")
    ax.set_ylim(0, 75)
    ax.set_xticks(x)
    ax.set_xticklabels(ERA_TICKS, rotation=30, ha="right")
    _style(ax)
    fig.tight_layout()
    _save(fig, "Fig5.png")


def main():
    print("Generating Fig2–Fig5 from canonical tables...")
    fig2_composition()
    fig3_annual_shannon()
    fig4_coop_subtypes()
    fig5_spatial()
    print("Done. Fig1 (study area) is generated separately and unchanged.")


if __name__ == "__main__":
    main()
