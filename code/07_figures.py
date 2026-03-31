"""Regenerate Fig1 biplot with new MCA (departamento as supplementary)."""
import re
import numpy as np
import pandas as pd
import prince
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from adjustText import adjust_text
from scipy.cluster.hierarchy import linkage, fcluster
from pathlib import Path

PROJECT = Path(__file__).parent.parent
DPI = 300
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": DPI, "savefig.bbox": "tight",
})

SUBTIPO_KW = {
    "agro": ["AGROPECUAR","AGRO ","AGRICOL","GANADER","YERBA","TABAC","FORESTAL","MADERA","ASERRADERO"],
    "religious": ["IGLESIA","EVANGELICA","PASTORAL","PARROQUIA","TEMPLO","CRISTIANA","ADVENTISTA",
                  "BAUTISTA","PENTECOSTAL","ASAMBLEA DE DIOS","METODISTA","LUTERANA","MENONITA","CONGREGACION","CULTO"],
    "sports": ["CLUB ","DEPORTIV","FUTBOL","ATLETICO"],
    "education": ["ESCUELA","COLEGIO","INSTITUTO","EDUCACI","BIBLIOTECA"],
    "health": ["HOSPITAL","CLINICA","SANATORIO","SALUD","MEDIC","FARMAC"],
    "transport": ["TRANSPORT","REMIS","TAXI","COLECTIV","CAMION"],
    "construction": ["CONSTRUC","INMOBILIAR","INMUEBLE","VIVIENDA"],
    "commerce": ["COMERCI","MERCADO","SUPERMERCADO","DISTRIBUID"],
    "tourism": ["TURIS","HOTEL","HOSTEL","CABANA","ALOJAMIENTO"],
}
RELIG_EXCL = re.compile(r"MISIONERA\s+S[.\s]|MISIONERO\s+S[.\s]|MISIONERAS\s+SA|ARENERA|AGUAS\s+MISION", re.I)

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
    for s, kws in SUBTIPO_KW.items():
        if any(kw in rs for kw in kws):
            if s == "religious" and RELIG_EXCL.search(rs): continue
            return s
    return "other"

def political_era(y):
    if pd.isna(y): return "unknown"
    y = int(y)
    if y <= 1989: return "pre1990"
    if y <= 1999: return "menem"
    if y <= 2002: return "crisis"
    if y <= 2007: return "n_kirchner"
    if y <= 2015: return "c_kirchner"
    if y <= 2019: return "macri"
    if y <= 2023: return "fernandez"
    return "milei"

# Load
geo = pd.read_parquet(PROJECT / "data" / "geocoded_sociedades.parquet")
geo["cuit"] = geo["cuit"].astype(str)
geo["fecha"] = pd.to_datetime(geo["fecha_hora_contrato_social"], errors="coerce")
geo["year"] = geo["fecha"].dt.year
geo["era"] = geo["year"].map(political_era)
geo["tipo"] = geo["tipo_societario"].map(classify_tipo)
geo["subtipo"] = geo["razon_social"].map(classify_sub)
geo["estado"] = geo["actividad_estado"].fillna("unknown")
geo = geo[(geo["era"] != "unknown") & (geo["estado"] != "unknown")]

# MCA with 4 active (no departamento)
act = geo[["tipo", "subtipo", "era", "estado"]].copy()
for col in act.columns:
    counts = act[col].value_counts()
    rare = counts[counts < 30].index
    if len(rare) > 0:
        act.loc[act[col].isin(rare), col] = "other_" + col

mca = prince.MCA(n_components=5, random_state=42)
mca = mca.fit(act)
row_coords = mca.row_coordinates(act)
row_coords.index = geo.index
col_coords = mca.column_coordinates(act)

K = len(act.columns)
threshold = 1.0 / K
benz = [((K/(K-1))*(l-threshold))**2 for l in mca.eigenvalues_ if l > threshold]
total_benz = sum(benz)
benz_pcts = [b/total_benz*100 for b in benz]

Z = linkage(row_coords.iloc[:, :5].values, method="ward")
clusters = pd.Series(fcluster(Z, t=5, criterion="maxclust"), index=geo.index)

# Map clusters to colors
cl_dominant = {}
for cl in sorted(clusters.unique()):
    cl_dominant[cl] = geo.loc[clusters == cl, "tipo"].value_counts().index[0]

COLOR_MAP = {"SAS": "#e41a1c", "Asoc": "#377eb8", "Coop": "#4daf4a",
             "SRL": "#ff7f00", "Fund": "#984ea3", "Otra": "#999999"}
HATCH_MAP = {"SAS": "xx", "Asoc": "//", "Coop": "..", "SRL": "", "Fund": "\\\\", "Otra": ""}
NAME_MAP = {"SAS": "SAS", "Asoc": "Associations", "Coop": "Cooperatives",
            "SRL": "Commercial SRL/SA", "Fund": "Inst. services", "Otra": "Other"}

CLUSTER_COLORS = {cl: COLOR_MAP.get(dom, "#999") for cl, dom in cl_dominant.items()}
CLUSTER_HATCHES = {cl: HATCH_MAP.get(dom, "") for cl, dom in cl_dominant.items()}
CLUSTER_NAMES = {cl: NAME_MAP.get(dom, dom) for cl, dom in cl_dominant.items()}

LABEL_COLORS = {"tipo": "#b30000", "subtipo": "#006600", "era": "#000099", "estado": "#993300"}

def detect_var(label):
    s = str(label)
    if any(s.endswith(t) for t in ["SRL","SA","SAS","Coop","Asoc","Fund","Otra","Mutual"]): return "tipo"
    if any(s.endswith(e) for e in ["pre1990","menem","crisis","n_kirchner","c_kirchner","macri","fernandez","milei"]): return "era"
    if any(s.endswith(e) for e in ["AC","BD"]): return "estado"
    return "subtipo"

def clean_label(s):
    s = str(s)
    for p in ["tipo_","subtipo_","era_","estado_","other_","sub_"]:
        s = s.replace(p, "")
    renames = {"n_kirchner":"N.Kirchner","c_kirchner":"C.Kirchner","fernandez":"Fernández",
               "milei":"Milei","macri":"Macri","menem":"Menem","pre1990":"Pre-1990",
               "crisis":"Crisis","BD":"Fiscally dead","AC":"Active"}
    for old, new in renames.items():
        s = s.replace(old, new)
    return s

# Plot
fig, ax = plt.subplots(figsize=(7.5, 7))

for cl in sorted(clusters.unique()):
    mask = clusters == cl
    sample = row_coords.loc[mask].sample(min(1200, mask.sum()), random_state=42)
    ax.scatter(sample.iloc[:, 0], sample.iloc[:, 1], c=CLUSTER_COLORS[cl], s=3, alpha=0.08, rasterized=True)

contrib = col_coords.iloc[:, 0]**2 + col_coords.iloc[:, 1]**2
top = set(contrib.nlargest(20).index)
# Force-include all political eras so they appear on the biplot
for idx in col_coords.index:
    if any(era in str(idx) for era in ["n_kirchner", "c_kirchner", "macri", "crisis"]):
        top.add(idx)

texts = []
for idx in col_coords.index:
    x, y = col_coords.loc[idx].iloc[0], col_coords.loc[idx].iloc[1]
    vt = detect_var(idx)
    color = LABEL_COLORS.get(vt, "#333")
    ax.plot(x, y, "s", color=color, markersize=3, alpha=0.5)
    if idx in top:
        label = clean_label(idx)
        t = ax.text(x, y, label, fontsize=7, fontweight="bold", color=color,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=color, alpha=0.85, lw=0.6))
        texts.append(t)

adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#666", lw=0.4),
            force_text=(0.8, 0.8), force_points=(0.5, 0.5), expand=(1.2, 1.4))

ax.axhline(0, color="#ddd", lw=0.5)
ax.axvline(0, color="#ddd", lw=0.5)
ax.set_xlabel(f"Axis 1 ({benz_pcts[0]:.1f}% corrected inertia)")
ax.set_ylabel(f"Axis 2 ({benz_pcts[1]:.1f}% corrected inertia)")

handles = [mpatches.Patch(color=CLUSTER_COLORS[c], hatch=CLUSTER_HATCHES.get(c, ""),
                          edgecolor="#555555", label=CLUSTER_NAMES[c]) for c in sorted(CLUSTER_COLORS)]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08),
          ncol=len(handles), fontsize=7, framealpha=0.95, edgecolor="#ccc", title="Clusters", title_fontsize=7.5)
plt.subplots_adjust(bottom=0.13)
plt.savefig(PROJECT / "figures" / "Fig1.png", dpi=DPI)
plt.close()
print("Fig1 OK", flush=True)
