"""
make_figures.py — figures for the reconstructed (model-free) paper
==================================================================
Reads the canonical tables written by analysis_diversity.py and produces
Fig2–Fig5 (Fig1 study-area map is generated separately). Consistent style
across all figures per the manuscript figure conventions: identical fonts,
a single form→colour map, identical state-project order and labels.

Outputs → submission_JRS_active/figures/Fig{2,3,4,5}.png  (300 dpi)
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
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300, "savefig.dpi": 300,
})

ERAS = ["Pre-1990", "Menem", "Crisis", "N.Kirchner", "C.Kirchner",
        "Macri", "Fernández", "Milei"]
FORMS = ["SAS", "SRL", "SA", "Coop", "Asoc", "Fund", "Otra"]
FORM_LABEL = {"SAS": "SAS", "SRL": "SRL", "SA": "SA", "Coop": "Cooperative",
              "Asoc": "Civil association", "Fund": "Foundation",
              "Otra": "Other"}
FORM_COLOR = {"SAS": "#1f77b4", "SRL": "#ff7f0e", "SA": "#2ca02c",
              "Coop": "#d62728", "Asoc": "#9467bd", "Fund": "#8c564b",
              "Otra": "#7f7f7f"}


def _save(fig, name):
    fig.tight_layout()
    out = FIGS / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def fig2_composition():
    from matplotlib.lines import Line2D
    d = pd.read_csv(TAB / "tab_composition_era.csv").set_index("era").reindex(ERAS)
    x = np.arange(len(ERAS))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bottom = np.zeros(len(ERAS))
    for f in FORMS:
        sh = d[f"{f}_share_%"].values
        ax.bar(x, sh, bottom=bottom, color=FORM_COLOR[f],
               label=FORM_LABEL[f], width=0.62, edgecolor="white", linewidth=0.3)
        bottom += sh
    ax.set_ylabel("Share of new registrations (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(ERAS, rotation=30, ha="right")
    ax2 = ax.twinx()
    ax2.plot(x, d["total_per_yr"].values, "k-o", lw=1.3, ms=4)
    ax2.set_ylabel("New registrations per year")
    ax2.set_ylim(0, max(d["total_per_yr"]) * 1.15)
    # Combined legend above chart: form bars + line handle
    bar_handles, bar_labels = ax.get_legend_handles_labels()
    line_handle = Line2D([0], [0], color="k", lw=1.3, marker="o", ms=4,
                         label="New registrations / year")
    ax.legend(handles=bar_handles + [line_handle],
              ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    _save(fig, "Fig2.png")


def fig3_annual_shannon():
    a = pd.read_csv(TAB / "tab_diversity_annual.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(a["year"], a["H"], color="#444", lw=0.9, marker="o", ms=3,
            label="Annual $H$")
    ax.plot(a["year"], a["roll5"], color="#d62728", lw=2.0,
            label="5-year rolling mean")
    bounds = [2000, 2003, 2008, 2016, 2020, 2024]
    for b in bounds:
        ax.axvline(b, color="#bbb", ls="--", lw=0.7, zorder=0)
    ax.set_xlabel("Year")
    ax.set_ylabel("Shannon diversity $H$")
    ax.set_xlim(a["year"].min(), a["year"].max())
    ax.legend(loc="lower left", frameon=False)
    _save(fig, "Fig3.png")


def fig4_coop_subtypes():
    c = pd.read_csv(TAB / "tab_coop_subtypes.csv").set_index("era").reindex(ERAS)
    x = np.arange(len(ERAS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - w, c["work_per_yr"], w, label="Work", color="#d62728")
    ax.bar(x, c["ag_service_per_yr"], w, label="Agricultural / service",
           color="#2ca02c")
    ax.bar(x + w, c["other_per_yr"], w, label="Residual", color="#7f7f7f")
    ax.set_ylabel("Cooperative registrations per year")
    ax.set_xticks(x)
    ax.set_xticklabels(ERAS, rotation=30, ha="right")
    ax.legend(frameon=False)
    _save(fig, "Fig4.png")


def fig5_spatial():
    s = pd.read_csv(TAB / "tab_spatial_formlevel.csv").set_index("era").reindex(ERAS)
    x = np.arange(len(ERAS))
    colors = [FORM_COLOR.get(f, "#333") for f in s["top_form"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x, s["top_form_radio_share_%"], color=colors, width=0.6)
    for i, (f, v) in enumerate(zip(s["top_form"], s["top_form_radio_share_%"])):
        ax.text(i, v + 1.5, f, ha="center", va="bottom", fontsize=7)
    # Milei permutation null band (mean 87.3 of 190 active radios -> %)
    mil_active = float(s.loc["Milei", "active_radios"])
    null_mean_pct = 87.3 / mil_active * 100
    null_p95_pct = 98.0 / mil_active * 100
    mi = list(ERAS).index("Milei")
    ax.errorbar([mi], [s.loc["Milei", "top_form_radio_share_%"]],
                yerr=[[s.loc["Milei", "top_form_radio_share_%"] - null_mean_pct],
                      [null_p95_pct - s.loc["Milei", "top_form_radio_share_%"]]],
                fmt="none", ecolor="black", capsize=4, lw=1.2)
    ax.annotate("Milei permutation\nnull (mean, 95th pct)",
                xy=(mi, null_mean_pct), xytext=(mi - 2.4, null_mean_pct + 14),
                fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_ylabel("Modal-form share of active census radios (%)")
    ax.set_ylim(0, 75)
    ax.set_xticks(x)
    ax.set_xticklabels(ERAS, rotation=30, ha="right")
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
