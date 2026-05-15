"""
04b — Sensitivity: department ACTIVE vs SUPPLEMENTARY
=====================================================
Auditable backing for the §4.4 sensitivity statement.

Canonical (04_run_acm.py): 4 active = tipo, subtipo, era, fiscal status;
department projected as SUPPLEMENTARY. This script re-runs the MCA with
department added as a 5th ACTIVE variable and quantifies the effect on
(i) the principal-axis geometry and (ii) the k=6 cluster membership.

Result (pinned environment): axes 1–2 are substantially preserved
(|r| ~ 0.99 and 0.92); axis 3 rotates (|r| ~ 0.80); the k=6 membership
DOES change (~91% best-permutation agreement, different cluster sizes).
The manuscript therefore reports the department-supplementary specification
and relies on modal dominant-cluster assignment (§4.5), which is invariant
to the membership sensitivity. The §4.4 text must claim axis-1/2 stability
ONLY — not cluster-structure invariance.

DB-free. Console + acm/tables/tab_sensitivity_dept.csv
"""
import re
import sys
import warnings
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import prince
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[2]
TAB_DIR = PROJECT / "acm" / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)

# Reuse the canonical classifiers/loader from 04 (same directory).
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "_canon", Path(__file__).with_name("04_run_acm.py"))
_canon = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_canon)


def mca_run(active):
    a = active.copy()
    for c in a.columns:
        vc = a[c].value_counts()
        rare = vc[vc < 30].index
        if len(rare):
            a.loc[a[c].isin(rare), c] = "other_" + c
    m = prince.MCA(n_components=10, random_state=42).fit(a)
    rc = m.row_coordinates(a).reset_index(drop=True)
    ev = np.asarray(m.eigenvalues_)
    Z = linkage(rc.iloc[:, :5].values, method="ward")
    lab = fcluster(Z, t=6, criterion="maxclust")
    return rc, ev, lab


def best_agreement(a, b):
    ua, ub = np.unique(a), np.unique(b)
    M = np.zeros((len(ua), len(ub)))
    for i, x in enumerate(ua):
        for j, y in enumerate(ub):
            M[i, j] = np.sum((a == x) & (b == y))
    r, c = linear_sum_assignment(-M)
    return M[r, c].sum() / len(a)


def main():
    geo = _canon.load_active()
    canon = geo[["tipo", "subtipo", "era", "estado"]].copy()
    deptav = geo[["tipo", "subtipo", "era", "departamento", "estado"]].copy()

    rc4, ev4, lab4 = mca_run(canon)
    rc5, ev5, lab5 = mca_run(deptav)

    b4 = _canon.benzecri_correction(ev4, 4)[1][:5]
    b5 = _canon.benzecri_correction(ev5, 5)[1][:5]
    rows = []
    print("\n=== Sensitivity: department supplementary vs active ===")
    for ax in range(3):
        r = abs(np.corrcoef(rc4.iloc[:, ax], rc5.iloc[:, ax])[0, 1])
        print(f"  axis {ax+1}: |r| = {r:.4f}   "
              f"Benzecri%% canon={b4[ax]:.1f}  dept-active={b5[ax]:.1f}")
        rows.append({"axis": ax + 1, "abs_r_canon_vs_deptactive": round(r, 4),
                     "benzecri_pct_canonical": round(b4[ax], 1),
                     "benzecri_pct_dept_active": round(b5[ax], 1)})
    agr = best_agreement(lab4, lab5)
    print(f"  k=6 cluster membership best-perm agreement = {agr:.4f}")
    print(f"  canonical sizes    = "
          f"{sorted(pd.Series(lab4).value_counts(), reverse=True)}")
    print(f"  dept-active sizes  = "
          f"{sorted(pd.Series(lab5).value_counts(), reverse=True)}")
    print("\n  → axes 1–2 substantially preserved; k=6 membership is "
          "specification-dependent. §4.4 must claim axis stability only.")

    out = pd.DataFrame(rows)
    out["k6_membership_agreement"] = round(agr, 4)
    out.to_csv(TAB_DIR / "tab_sensitivity_dept.csv", index=False)
    print(f"\n  wrote {TAB_DIR / 'tab_sensitivity_dept.csv'}")


if __name__ == "__main__":
    main()
