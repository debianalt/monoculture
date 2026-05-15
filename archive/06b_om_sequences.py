"""
06b — Optimal Matching sequence analysis (Abbott genuine)
==========================================================
Implements Needleman-Wunsch optimal matching on dominant-cluster sequences
(17 departments x 8 political eras), then Ward hierarchical clustering on
the OM distance matrix to derive trajectory types empirically.

Replaces the narrative metropolitan/frontier/oscillating typology in §5.2
with empirically derived groups.

Outputs:
  acm/tables/tab_S10_om_distance_matrix.csv  (17 x 17 OM distances)
  acm/tables/tab_S11_sequence_clusters.csv   (dept -> trajectory cluster)
  acm/figures/Fig3b_sequence_dendrogram.png  (Ward dendrogram on OM)
  Console: silhouette for k=2..6, chosen k, cluster membership, medoid seqs
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

# Paths
MONO_CODE = Path(__file__).parent
MONO = MONO_CODE.parent
ACM = MONO.parent / "acm"
TAB_DIR = ACM / "tables"
FIG_DIR = ACM / "figures"
TAB_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

DPI = 300

# Cluster v2 -> manuscript-aligned label
V2_TO_NAME = {1: "SAS", 2: "Assoc", 3: "Coop", 4: "Commercial", 5: "Services", 6: "Services"}

ERA_BOUNDS = [
    ("Pre1990", 1900, 1989),
    ("Menem", 1990, 1999),
    ("Crisis", 2000, 2002),
    ("NK", 2003, 2007),
    ("CK", 2008, 2015),
    ("Macri", 2016, 2019),
    ("Fern", 2020, 2023),
    ("Milei", 2024, 2025),
]
ERA_ORDER = [e[0] for e in ERA_BOUNDS]

STATES = ["Commercial", "Assoc", "Coop", "Services", "SAS", "None"]
STATE_IDX = {s: i for i, s in enumerate(STATES)}

# Theory-derived costs (Abbott & Tsay 2000):
#   substitution between any two non-identical states: 2.0 (uniform)
#   identity: 0
#   indel (gap insertion against "None" / absent observation): 1.0
SUB_COST = 2.0
INDEL_COST = 1.0


def assign_era(year):
    if pd.isna(year):
        return None
    y = int(year)
    for name, lo, hi in ERA_BOUNDS:
        if lo <= y <= hi:
            return name
    return None


def load_sequences():
    """Build dept (17) x era (8) dominant-cluster grid from v2 assignments."""
    v2 = pd.read_csv(ACM / "data" / "org_cluster_assignments_v2.csv")
    geo = pd.read_parquet(MONO / "data" / "geocoded_sociedades.parquet")
    v2["cuit"] = v2["cuit"].astype(str)
    geo["cuit"] = geo["cuit"].astype(str)
    m = v2.merge(
        geo[["cuit", "fecha_hora_contrato_social", "departamento"]],
        on="cuit", how="left",
    )
    m["year"] = pd.to_datetime(m["fecha_hora_contrato_social"], errors="coerce").dt.year
    m["era"] = m["year"].map(assign_era)
    m["regime"] = m["cluster"].map(V2_TO_NAME)
    m = m.dropna(subset=["departamento", "era", "regime"])

    # Dominant cluster per dept x era (mode by count)
    counts = m.groupby(["departamento", "era", "regime"]).size().reset_index(name="n")
    idx = counts.groupby(["departamento", "era"])["n"].idxmax()
    dom = counts.loc[idx, ["departamento", "era", "regime"]]
    seq = dom.pivot(index="departamento", columns="era", values="regime")
    seq = seq[ERA_ORDER]
    seq = seq.fillna("None")
    return seq


def nw_distance(s1, s2, sub_cost=SUB_COST, indel_cost=INDEL_COST):
    """Needleman-Wunsch optimal matching distance between two sequences."""
    L1, L2 = len(s1), len(s2)
    D = np.zeros((L1 + 1, L2 + 1))
    for i in range(1, L1 + 1):
        D[i, 0] = i * indel_cost
    for j in range(1, L2 + 1):
        D[0, j] = j * indel_cost
    for i in range(1, L1 + 1):
        for j in range(1, L2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else sub_cost
            D[i, j] = min(
                D[i - 1, j - 1] + cost,
                D[i - 1, j] + indel_cost,
                D[i, j - 1] + indel_cost,
            )
    return D[L1, L2]


def om_matrix(seq_df):
    """Pairwise OM distance matrix (n_depts x n_depts), symmetric."""
    depts = list(seq_df.index)
    n = len(depts)
    M = np.zeros((n, n))
    seqs = [list(seq_df.loc[d, ERA_ORDER]) for d in depts]
    for i in range(n):
        for j in range(i + 1, n):
            d = nw_distance(seqs[i], seqs[j])
            M[i, j] = d
            M[j, i] = d
    return pd.DataFrame(M, index=depts, columns=depts)


def silhouette_over_k(D, k_range=range(2, 7)):
    """Silhouette per k using precomputed distance matrix."""
    dist_vec = squareform(D.values, checks=False)
    Z = linkage(dist_vec, method="ward")
    scores = []
    for k in k_range:
        labels = fcluster(Z, t=k, criterion="maxclust")
        if len(set(labels)) < 2:
            scores.append((k, np.nan))
            continue
        s = silhouette_score(D.values, labels, metric="precomputed")
        scores.append((k, s))
    return scores, Z


def medoid_per_cluster(D, labels):
    """For each cluster, find the dept with minimum total distance to others."""
    medoids = {}
    for c in sorted(set(labels)):
        members = [i for i, l in enumerate(labels) if l == c]
        # restrict to within-cluster distances
        sub = D.values[np.ix_(members, members)]
        idx = members[sub.sum(axis=1).argmin()]
        medoids[c] = D.index[idx]
    return medoids


def main():
    print("=" * 70)
    print("06b — Optimal Matching sequence analysis")
    print("=" * 70)

    seq = load_sequences()
    print(f"\nSequences (17 depts x {len(seq.columns)} eras):")
    print(seq.to_string())

    # OM distance matrix
    print("\nComputing OM distance matrix (Needleman-Wunsch, sub=2.0, indel=1.0)...")
    D = om_matrix(seq)
    D.to_csv(TAB_DIR / "tab_S10_om_distance_matrix.csv")
    print(f"  Saved: {TAB_DIR / 'tab_S10_om_distance_matrix.csv'}")
    print(f"  Distance range: [{D.values[np.triu_indices_from(D.values, 1)].min():.2f}, "
          f"{D.values[np.triu_indices_from(D.values, 1)].max():.2f}]")

    # Silhouette over k
    print("\nSilhouette over k=2..6:")
    scores, Z = silhouette_over_k(D)
    for k, s in scores:
        print(f"  k={k}: silhouette = {s:.3f}" if not np.isnan(s) else f"  k={k}: degenerate")
    best_k = max([(k, s) for k, s in scores if not np.isnan(s)], key=lambda x: x[1])[0]
    print(f"\nBest k (max silhouette): k={best_k}")

    # Also report k=3 for narrative comparison
    print("\nReporting k=3 explicitly for comparison with narrative typology:")
    labels_k3 = fcluster(Z, t=3, criterion="maxclust")
    out_k3 = pd.DataFrame({"departamento": D.index, "cluster_k3": labels_k3})
    print(out_k3.to_string(index=False))

    # Final clustering at best_k
    labels_best = fcluster(Z, t=best_k, criterion="maxclust")
    medoids = medoid_per_cluster(D, labels_best)
    print(f"\nMedoids per cluster (k={best_k}):")
    for c, m in medoids.items():
        seq_m = list(seq.loc[m, ERA_ORDER])
        print(f"  Cluster {c} (medoid={m}): {' -> '.join(seq_m)}")

    # Save assignments
    out = pd.DataFrame({
        "departamento": D.index,
        "cluster_k3": labels_k3,
        f"cluster_k{best_k}": labels_best,
        "sequence": [" -> ".join(seq.loc[d, ERA_ORDER]) for d in D.index],
    })
    out.to_csv(TAB_DIR / "tab_S11_sequence_clusters.csv", index=False)
    print(f"\n  Saved: {TAB_DIR / 'tab_S11_sequence_clusters.csv'}")

    # Dendrogram
    fig, ax = plt.subplots(figsize=(9, 6))
    dendrogram(Z, labels=list(D.index), leaf_rotation=45, leaf_font_size=9,
               color_threshold=Z[-best_k + 1, 2] if best_k > 1 else 0)
    ax.set_ylabel("OM distance (Ward linkage)")
    ax.set_title(f"Ward clustering of OM-distance sequences (k={best_k}, "
                 f"silhouette={dict(scores)[best_k]:.2f})")
    plt.tight_layout()
    out_png = MONO.parent / "submision JRS - active" / "figures" / "Fig3b.png"
    plt.savefig(out_png, dpi=DPI)
    plt.savefig(FIG_DIR / "Fig3b_sequence_dendrogram.png", dpi=DPI)
    plt.close()
    print(f"\n  Saved figure: {out_png}")

    # Save silhouette table
    sil_df = pd.DataFrame(scores, columns=["k", "silhouette"])
    sil_df.to_csv(TAB_DIR / "tab_S11b_silhouette.csv", index=False)
    print(f"  Saved: {TAB_DIR / 'tab_S11b_silhouette.csv'}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    return best_k, dict(scores), out


if __name__ == "__main__":
    main()
