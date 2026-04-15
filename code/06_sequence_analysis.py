"""
06 — Sequence analysis of organisational regimes
==================================================
Defines dominant organisational regime per department × era,
then produces:
  Fig6 — Sequence index plot (17 departments × 8 eras)
  Fig7 — Transition proportion matrix + alluvial flows
"""

import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ACM_DIR = Path(__file__).parent
PROJECT = ACM_DIR.parent
FIG_DIR = ACM_DIR / "figures"
DPI = 300

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "hatch.linewidth": 0.5,
})

ERA_ORDER = ["1.Pre1990", "2.Menem", "3.Crisis", "4.NK", "5.CK", "6.Macri", "7.Fern", "8.Milei"]
ERA_LABELS = ["Pre-1990", "Menem\n1990-99", "Crisis\n2000-02", "N. Kirchner\n2003-07", "C. Kirchner\n2008-15", "Macri\n2016-19", "Fernández\n2020-23", "Milei\n2024-25"]

REGIME_COLORS = {
    "Commercial": "#ff7f00",
    "Coop": "#4daf4a",
    "Assoc": "#377eb8",
    "Services": "#984ea3",
    "SAS": "#e41a1c",
    "None": "#cccccc",
}
REGIME_ORDER = ["Commercial", "Assoc", "Coop", "Services", "SAS"]

REGIME_HATCHES = {
    "Commercial": "",
    "Assoc": "//",
    "Coop": "..",
    "Services": "\\\\",
    "SAS": "xx",
    "None": "",
}


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

    # Dominant regime per department × era
    dept_era = merged.groupby(["departamento", "era", "regime"]).size().reset_index(name="n")
    idx = dept_era.groupby(["departamento", "era"])["n"].idxmax()
    dominant = dept_era.loc[idx, ["departamento", "era", "regime"]].copy()
    seq = dominant.pivot(index="departamento", columns="era", values="regime").fillna("None")
    seq = seq[[e for e in ERA_ORDER if e in seq.columns]]

    return seq, merged


def compute_transitions(seq):
    """Compute transition counts and probabilities."""
    trans_counts = Counter()
    for _, row in seq.iterrows():
        vals = [row[e] for e in ERA_ORDER if e in row.index]
        for i in range(len(vals) - 1):
            trans_counts[(vals[i], vals[i + 1])] += 1

    # Transition proportion matrix
    states = REGIME_ORDER
    n_states = len(states)
    state_idx = {s: i for i, s in enumerate(states)}
    counts = np.zeros((n_states, n_states))
    for (s1, s2), n in trans_counts.items():
        if s1 in state_idx and s2 in state_idx:
            counts[state_idx[s1], state_idx[s2]] += n

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    probs = counts / row_sums

    return trans_counts, counts, probs


def compute_dwell_times(seq):
    """Compute average dwell time (consecutive eras in same regime)."""
    dwells = {r: [] for r in REGIME_ORDER}
    for _, row in seq.iterrows():
        vals = [row[e] for e in ERA_ORDER if e in row.index]
        current = vals[0]
        count = 1
        for v in vals[1:]:
            if v == current:
                count += 1
            else:
                if current in dwells:
                    dwells[current].append(count)
                current = v
                count = 1
        if current in dwells:
            dwells[current].append(count)

    result = {}
    for r, times in dwells.items():
        if times:
            result[r] = {"mean": np.mean(times), "max": max(times), "n": len(times)}
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Fig 6 — Sequence index plot
# ═════════════════════════════════════════════════════════════════════════════

def fig6_sequence_index(seq):
    # Sort departments: frontier (most Coop eras) at top, metropolitan at bottom
    def coop_score(dept):
        row = seq.loc[dept]
        return sum(1 for e in ERA_ORDER if e in row.index and row[e] == "Coop")

    depts_sorted = sorted(seq.index, key=coop_score, reverse=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for i, dept in enumerate(depts_sorted):
        for j, era in enumerate(ERA_ORDER):
            if era in seq.columns:
                regime = seq.loc[dept, era]
                color = REGIME_COLORS.get(regime, "#cccccc")
                hatch = REGIME_HATCHES.get(regime, "")
                ax.barh(i, 1, left=j, height=0.85, color=color, hatch=hatch,
                        edgecolor="black", linewidth=0.3)

    ax.set_yticks(range(len(depts_sorted)))
    ax.set_yticklabels(depts_sorted, fontsize=9.5)
    ax.set_xticks(np.arange(len(ERA_ORDER)) + 0.5)
    ax.set_xticklabels(ERA_LABELS, fontsize=9.5, rotation=45, ha="right")
    ax.set_xlim(0, len(ERA_ORDER))
    ax.set_ylim(-0.5, len(depts_sorted) - 0.5)
    ax.invert_yaxis()

    # Legend below plot — hatches visibles via hatch.linewidth temporal mayor
    LEGEND_NAMES = {"Commercial": "Commercial", "Assoc": "Associations", "Coop": "Cooperatives",
                    "Services": "Services", "SAS": "SAS", "None": "No creations"}
    legend_order = REGIME_ORDER + (["None"] if (seq.values == "None").any() else [])
    with plt.rc_context({"hatch.linewidth": 1.2}):
        handles = [mpatches.Patch(facecolor=REGIME_COLORS[r], hatch=REGIME_HATCHES.get(r, ""),
                                  edgecolor="black", linewidth=0.8,
                                  label=LEGEND_NAMES.get(r, r)) for r in legend_order]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.30),
                  ncol=len(legend_order), fontsize=8, framealpha=0.95, edgecolor="#ccc",
                  handleheight=2.0, handlelength=3.2)

    ax.set_xlabel("Political era", fontsize=10)
    ax.set_ylabel("")
    plt.subplots_adjust(bottom=0.24)

    plt.savefig(FIG_DIR / "Fig6_sequences.tiff", dpi=DPI, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(FIG_DIR / "Fig6_sequences.png", dpi=DPI)
    plt.savefig(PROJECT / "figures" / "Fig2.png", dpi=DPI)
    plt.close()
    print("  Fig6_sequences OK -> figures/Fig2.png", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# Fig 7 — Transition matrix + key metrics
# ═════════════════════════════════════════════════════════════════════════════

def fig7_transitions(counts, probs, dwells):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.2, 1]})

    # A: Transition proportion matrix
    ax = axes[0]
    im = ax.imshow(probs, cmap="YlOrRd", vmin=0, vmax=0.7, aspect="auto")
    ax.set_xticks(range(len(REGIME_ORDER)))
    ax.set_yticks(range(len(REGIME_ORDER)))
    ax.set_xticklabels(REGIME_ORDER, fontsize=11, rotation=45, ha="right")
    ax.set_yticklabels(REGIME_ORDER, fontsize=11)
    ax.set_xlabel("State at t+1", fontsize=12)
    ax.set_ylabel("State at t", fontsize=12)
    ax.text(-0.05, 1.08, "A", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

    for i in range(len(REGIME_ORDER)):
        for j in range(len(REGIME_ORDER)):
            val = probs[i, j]
            if val > 0.03:
                color = "white" if val > 0.4 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=12, color=color,
                        fontweight="bold" if i == j else "normal")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Transition proportion")

    # B: Dwell times + persistence rates
    ax = axes[1]
    regimes = [r for r in REGIME_ORDER if r in dwells]
    x = np.arange(len(regimes))
    dwell_vals = [dwells[r]["mean"] for r in regimes]
    persist = [probs[REGIME_ORDER.index(r), REGIME_ORDER.index(r)] for r in regimes]
    colors = [REGIME_COLORS[r] for r in regimes]

    hatches = [REGIME_HATCHES.get(r, "") for r in regimes]
    bars = ax.bar(x, dwell_vals, color=colors, alpha=0.85, edgecolor="#555555", linewidth=0.5)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, fontsize=11, rotation=45, ha="right")
    ax.set_ylabel("Mean dwell time (eras)", fontsize=12)
    ax.text(-0.05, 1.08, "B", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

    # Annotate with persistence probability
    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                f"p={persist[i]:.2f}\n({dwell_vals[i]:.1f} eras)",
                ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0, max(dwell_vals) * 1.4)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "Fig7_transitions.tiff", dpi=DPI, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(FIG_DIR / "Fig7_transitions.png", dpi=DPI)
    plt.savefig(PROJECT / "figures" / "Fig3.png", dpi=DPI)
    plt.close()
    print("  Fig7_transitions OK -> figures/Fig3.png", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Sequence analysis of organisational regimes", flush=True)
    print("=" * 50, flush=True)

    seq, merged = load_sequences()
    trans_counts, counts, probs = compute_transitions(seq)
    dwells = compute_dwell_times(seq)

    print("\nSequences:", flush=True)
    print(seq.to_string(), flush=True)

    print("\nTransition probabilities:", flush=True)
    prob_df = pd.DataFrame(probs, index=REGIME_ORDER, columns=REGIME_ORDER)
    print(prob_df.round(2).to_string(), flush=True)

    print("\nDwell times:", flush=True)
    for r, d in dwells.items():
        print(f"  {r:12s}: mean={d['mean']:.1f} eras, max={d['max']}, n={d['n']} spells", flush=True)

    print("\nGenerating figures...", flush=True)
    fig6_sequence_index(seq)
    fig7_transitions(counts, probs, dwells)

    print("\nDone.", flush=True)
