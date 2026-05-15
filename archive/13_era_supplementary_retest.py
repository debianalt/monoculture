"""
13 — Decisive re-test: MCA with POLITICAL ERA projected SUPPLEMENTARY.
Active = juridical form + functional subtype + fiscal status (era OUT).
If the SAS cluster + spatial uniformity survive when era no longer
constructs the space, the monoculture construct is rescued; if Milei
creations scatter across clusters (as the 43.9% form-level mix predicts),
the spatial-uniformity dimension is confirmed an era-as-active artefact.
"""
import importlib.util
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import pandas as pd
import prince
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import silhouette_score

RNG = np.random.default_rng(42)
PROJECT = Path(__file__).resolve().parents[2]
TAB = PROJECT / "acm" / "tables"
spec = importlib.util.spec_from_file_location(
    "_c", Path(__file__).with_name("04_run_acm.py"))
canon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canon)


def label_by_content(geo, raw):
    """Name each cluster by its plurality juridical form."""
    df = pd.DataFrame({"r": raw, "t": geo["tipo"].values})
    out = {}
    for r, sub in df.groupby("r"):
        vc = sub["t"].value_counts(normalize=True)
        top = vc.index[0]
        nm = {"SAS": "SAS", "Coop": "Coop", "Asoc": "Assoc",
              "SRL": "Commercial", "SA": "Commercial"}.get(top, "Other")
        out[r] = f"{nm}({top} {vc.iloc[0]*100:.0f}%)"
    simple = {}
    for r, sub in df.groupby("r"):
        top = sub["t"].value_counts(normalize=True).index[0]
        simple[r] = {"SAS": "SAS", "Coop": "Coop", "Asoc": "Assoc",
                     "SRL": "Commercial", "SA": "Commercial"}.get(top, "Other")
    return out, simple


def main():
    geo = canon.load_active()
    active = geo[["tipo", "subtipo", "estado"]].copy()   # ERA OUT
    for c in active.columns:
        vc = active[c].value_counts(); r = vc[vc < 30].index
        if len(r):
            active.loc[active[c].isin(r), c] = "other_" + c

    mca = prince.MCA(n_components=10, random_state=42).fit(active)
    rc = mca.row_coordinates(active).reset_index(drop=True)
    ev = np.asarray(mca.eigenvalues_)
    bz = canon.benzecri_correction(ev, active.shape[1])[1][:5]
    print("Active = tipo + subtipo + estado  (era SUPPLEMENTARY)")
    print(f"Benzecri%: {[round(x,1) for x in bz]}")

    # silhouette to choose k (mirror manuscript procedure)
    samp = rc.iloc[:, :5].sample(min(10000, len(rc)), random_state=42).values
    Zs = linkage(samp, method="ward")
    print("silhouette by k:")
    for k in range(3, 9):
        sl = silhouette_score(samp, fcluster(Zs, t=k, criterion="maxclust"))
        print(f"  k={k}: {sl:.3f}")

    Z = linkage(rc.iloc[:, :5].values, method="ward")
    for K in (5, 6):
        raw = fcluster(Z, t=K, criterion="maxclust")
        verbose, simple = label_by_content(geo, raw)
        geo["_cl"] = pd.Series(raw).map(simple).values
        print(f"\n--- k={K} clusters (named by plurality form) ---")
        for r, lab in verbose.items():
            print(f"  raw{r}: n={int((raw==r).sum())}  {lab}")

        mil = geo[geo["year"] >= 2024]
        mix = mil["_cl"].value_counts(normalize=True)
        print(f"  Milei creations cluster mix: "
              f"{ {k: round(v,3) for k,v in mix.items()} }")
        # era as supplementary: centroid of Milei rows on axes 1-2
        ax = rc.loc[geo.index[geo['year'] >= 2024], [0, 1]].mean().round(3)
        print(f"  Milei supplementary centroid (ax1,ax2) = {ax.tolist()}")

        dep = mil.groupby("departamento")["_cl"].agg(
            lambda s: s.value_counts().index[0])
        rad = mil.groupby("redcode")["_cl"].agg(
            lambda s: s.value_counts().index[0])
        sdep = int((dep == "SAS").sum())
        srad = int((rad == "SAS").sum())
        print(f"  SAS-cluster dominant: depts {sdep}/{dep.size}  "
              f"radios {srad}/{rad.size}")

        # permutation null at this cluster level
        cats = mix.index.to_numpy(); probs = mix.values
        counts = mil.groupby("redcode").size().values
        nit = 4999
        null = np.empty(nit, int)
        for it in range(nit):
            s = 0
            for kk in counts:
                d = RNG.choice(cats, size=kk, p=probs)
                u, c = np.unique(d, return_counts=True)
                if u[c.argmax()] == "SAS":
                    s += 1
            null[it] = s
        p = (np.sum(null >= srad) + 1) / (nit + 1)
        print(f"  permutation null radios SAS-dom: mean={null.mean():.1f} "
              f"p95={np.percentile(null,95):.0f}  obs={srad}  p={p:.4f}")

    print("\nReference points:")
    print("  circular 4-active cluster : 17/17 depts, 180/180 radios (p=1.0)")
    print("  raw juridical FORM level  : 7/17 depts, 88/190 radios (p=0.49)")
    print("  → era-supplementary result above is the decisive arbiter.")


if __name__ == "__main__":
    main()
