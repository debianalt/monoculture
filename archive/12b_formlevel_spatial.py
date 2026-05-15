"""
12b — Non-circular spatial-uniformity at the JURIDICAL-FORM level.
The cluster-level 17/17 & 180/180 are mechanical (era is an active MCA
variable → 100% of Milei creations land in the era-defined SAS cluster).
This recomputes spatial dominance on raw juridical form (tipo), which does
NOT enter any model, plus a permutation null from the Milei form mix.
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

RNG = np.random.default_rng(42)
PROJECT = Path(__file__).resolve().parents[2]
s = importlib.util.spec_from_file_location(
    "_a", Path(__file__).with_name("04_run_acm.py"))
cn = importlib.util.module_from_spec(s); s.loader.exec_module(cn)

g = pd.read_parquet(PROJECT / "data" / "geocoded_sociedades.parquet")
g["cuit"] = g["cuit"].astype(str)
g["redcode"] = g["redcode"].astype(str)
g["year"] = pd.to_datetime(g["fecha_hora_contrato_social"],
                           errors="coerce").dt.year
g["form"] = g["tipo_societario"].map(cn.classify_tipo)
g = g[g["year"].notna()]
mil = g[g["year"] >= 2024]

print(f"Milei creations: {len(mil)}")
fm = mil["form"].value_counts(normalize=True)
print("Milei juridical-FORM mix:")
for k, v in fm.items():
    print(f"  {k:6s} {v*100:5.1f}%")

# observed form-level dominance
dep = mil.groupby("departamento")["form"].agg(lambda x: x.value_counts().index[0])
rad = mil.groupby("redcode")["form"].agg(lambda x: x.value_counts().index[0])
print(f"\nDept  SAS-form dominant: {(dep=='SAS').sum()}/{dep.size}")
print(f"Radio SAS-form dominant: {(rad=='SAS').sum()}/{rad.size}")

# permutation null: draw each radio's k_i creations from the Milei form mix
cats, probs = fm.index.to_numpy(), fm.values
counts = mil.groupby("redcode").size().values
nit = 9999
null = np.empty(nit, int)
for it in range(nit):
    sdom = 0
    for k in counts:
        d = RNG.choice(cats, size=k, p=probs)
        u, c = np.unique(d, return_counts=True)
        if u[c.argmax()] == "SAS":
            sdom += 1
    null[it] = sdom
obs = int((rad == "SAS").sum())
p = (np.sum(null >= obs) + 1) / (nit + 1)
print(f"\nForm-level null SAS-dominant radios: mean={null.mean():.1f} "
      f"p95={np.percentile(null,95):.0f} max={null.max()}  obs={obs}")
print(f"P(null>=obs) = {p:.4f}")
print("\n→ This is the non-circular statistic to report instead of the "
      "cluster-level 180/180. SAS *form* share ≈ {:.0f}% (not 100%).".format(
          fm.get("SAS", 0) * 100))

# prior eras for context: was any single FORM ever this spatially dominant?
print("\nMax single-form radio-dominance share by era (form level):")
for lo, hi, nm in [(1990, 1999, "Menem"), (2003, 2007, "NK"),
                   (2008, 2015, "CK"), (2016, 2019, "Macri"),
                   (2020, 2023, "Fernandez"), (2024, 2025, "Milei")]:
    e = g[(g.year >= lo) & (g.year <= hi)]
    rd = e.groupby("redcode")["form"].agg(lambda x: x.value_counts().index[0])
    sh = rd.value_counts(normalize=True)
    print(f"  {nm:10s} top form={sh.index[0]:5s} "
          f"{sh.iloc[0]*100:5.1f}% of {rd.size} active radios")
