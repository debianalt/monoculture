"""
Phase 0 — position-sensitive verification.
Does the canonical-spec regeneration preserve the per-record claims, not just
cluster sizes? Compares v2 vs regenerated (best-perm aligned) on:
  (i)   17x8 dominant-cluster grid (OM input)
  (ii)  radios with >=1 Milei creation that are SAS-dominant (180/180)
  (iii) departments SAS-dominant under Milei (17/17)
  (iv)  Coop-cluster count in the 4 frontier departments
NON-DESTRUCTIVE. Console only.
"""
import re, warnings
from pathlib import Path
import numpy as np, pandas as pd, prince
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
GEO = ROOT / "data" / "geocoded_sociedades.parquet"
ENR = ROOT / "data" / "enriquecido_arca_v2.parquet"
V2 = ROOT / "acm" / "data" / "org_cluster_assignments_v2.csv"

SUB = {
 "agro":["AGROPECUAR","AGRO ","AGRICOL","GANADER","YERBA","TABAC","FORESTAL","MADERA","ASERRADERO","VIVERO","APICOL","CITRICOL"],
 "religious":["IGLESIA","EVANGELICA","PASTORAL","PARROQUIA","TEMPLO","CRISTIANA","ADVENTISTA","BAUTISTA","PENTECOSTAL","ASAMBLEA DE DIOS","METODISTA","LUTERANA","MENONITA","CONGREGACION","MINISTERIO CRISTIAN","CULTO"],
 "sports":["CLUB ","DEPORTIV","FUTBOL","ATLETICO"],
 "indigenous":["COMUNIDAD ABORIGEN","COMUNIDAD INDIGENA","MBYA GUARANI"],
 "education":["ESCUELA","COLEGIO","INSTITUTO","EDUCACI","BIBLIOTECA"],
 "health":["HOSPITAL","CLINICA","SANATORIO","SALUD","MEDIC","FARMAC"],
 "transport":["TRANSPORT","REMIS","TAXI","COLECTIV","CAMION"],
 "construction":["CONSTRUC","INMOBILIAR","INMUEBLE","VIVIENDA"],
 "commerce":["COMERCI","MERCADO","SUPERMERCADO","DISTRIBUID"],
 "tourism":["TURIS","HOTEL","HOSTEL","CABANA","ALOJAMIENTO"]}
RX = re.compile(r"MISIONERA\s+S[.\s]|MISIONERO\s+S[.\s]|MISIONERAS\s+SA|ARENERA|AGUAS\s+MISION", re.I)

def c_tipo(t):
    t=str(t).upper()
    if "COOPERATIVA" in t: return "Coop"
    if "ASOCIACION CIVIL" in t: return "Asoc"
    if "FUNDACION" in t: return "Fund"
    if "MUTUAL" in t: return "Mutual"
    if "RESPONSABILIDAD LIMITADA" in t: return "SRL"
    if t=="SOCIEDAD ANONIMA": return "SA"
    if "ACCION SIMPLIFICADA" in t: return "SAS"
    return "Otra"
def c_sub(rs):
    rs=str(rs).upper()
    for s,ks in SUB.items():
        if any(k in rs for k in ks):
            if s=="religious" and RX.search(rs): continue
            return s
    return "other"
def era_of(y):
    if pd.isna(y): return "unknown"
    y=int(y)
    return ("pre1990" if y<=1989 else "menem" if y<=1999 else "crisis" if y<=2002
            else "n_kirchner" if y<=2007 else "c_kirchner" if y<=2015 else "macri"
            if y<=2019 else "fernandez" if y<=2023 else "milei")
ERAS=["pre1990","menem","crisis","n_kirchner","c_kirchner","macri","fernandez","milei"]
NAME={1:"SAS",2:"Assoc",3:"Coop",4:"Commercial",5:"Services",6:"Services"}
FRONTIER=["Guarani","Cainguas","San Pedro","G. M. Belgrano"]

def regen():
    g=pd.read_parquet(GEO); e=pd.read_parquet(ENR)
    g["cuit"]=g["cuit"].astype(str); e["cuit"]=e["cuit"].astype(str)
    g["tipo"]=g["tipo_societario"].map(c_tipo)
    g["subtipo"]=g["razon_social"].map(c_sub)
    g["year"]=pd.to_datetime(g["fecha_hora_contrato_social"],errors="coerce").dt.year
    g["era"]=g["year"].map(era_of)
    e["viva"]=e["error"].isna()
    g=g.merge(e[["cuit","viva"]],on="cuit",how="left")
    g=g[(g["era"]!="unknown") & g["actividad_estado"].notna()].copy()
    g["estado"]=np.where(g["actividad_estado"]=="BD","cancelled","active")
    act=g[["tipo","subtipo","era","estado"]].copy()
    for c in act.columns:
        vc=act[c].value_counts(); rare=vc[vc<30].index
        if len(rare): act.loc[act[c].isin(rare),c]="other_"+c
    m=prince.MCA(n_components=10,random_state=42).fit(act)
    rc=m.row_coordinates(act)
    Z=linkage(rc.iloc[:,:5].values,method="ward")
    g["_cl"]=fcluster(Z,t=6,criterion="maxclust")
    return g[["cuit","departamento","redcode","era","_cl"]]

def align(regen_lab, v2_lab):
    ua,ub=np.unique(regen_lab),np.unique(v2_lab)
    M=np.zeros((len(ua),len(ub)))
    for i,x in enumerate(ua):
        for j,y in enumerate(ub):
            M[i,j]=np.sum((regen_lab==x)&(v2_lab==y))
    r,c=linear_sum_assignment(-M)
    mp={ua[i]:ub[j] for i,j in zip(r,c)}
    return np.array([mp[x] for x in regen_lab])

def grid(df,colname):
    df=df[df["departamento"].notna()].copy()
    cnt=df.groupby(["departamento","era",colname]).size().reset_index(name="n")
    idx=cnt.groupby(["departamento","era"])["n"].idxmax()
    dom=cnt.loc[idx]
    return dom.pivot(index="departamento",columns="era",values=colname).reindex(columns=ERAS)

def main():
    g=regen()
    v2=pd.read_csv(V2); v2["cuit"]=v2["cuit"].astype(str)
    df=g.merge(v2[["cuit","cluster"]],on="cuit",how="inner")
    print("merged",len(df),"(expect 13905)")
    df["_aln"]=align(df["_cl"].values, df["cluster"].values)
    df["regen"]=df["_aln"].map(NAME)
    df["v2"]=df["cluster"].map(NAME)
    print("record agreement after name-map:",(df.regen==df.v2).mean().round(4))

    # (iii) 17/17 depts SAS-dominant under Milei
    mil=df[df.era=="milei"]
    for tag in ["v2","regen"]:
        dom=mil.groupby("departamento")[tag].agg(lambda s:s.value_counts().index[0])
        print(f"(iii) {tag}: SAS-dominant depts under Milei = {(dom=='SAS').sum()}/{dom.size}")

    # (ii) radios with >=1 Milei creation, SAS-dominant
    for tag in ["v2","regen"]:
        d=mil.groupby("redcode")[tag].agg(lambda s:s.value_counts().index[0])
        print(f"(ii) {tag}: SAS-dominant radios under Milei = {(d=='SAS').sum()}/{d.size}")

    # (i) 17x8 grid equality
    G2,GR=grid(df,"v2"),grid(df,"regen")
    diff=(G2.fillna("NA")!=GR.fillna("NA")).sum().sum()
    print(f"(i) 17x8 dominant-cluster grid: {diff} of {G2.size} cells differ (v2 vs regen)")
    if diff: print(G2.compare(GR))

    # (iv) Coop-cluster counts in frontier depts
    fr=df[df.departamento.isin(FRONTIER)]
    print("(iv) Coop-cluster count, frontier depts:")
    print("   v2   :", fr[fr.v2=="Coop"].groupby("departamento").size().to_dict(),
          "total", (fr.v2=="Coop").sum())
    print("   regen:", fr[fr.regen=="Coop"].groupby("departamento").size().to_dict(),
          "total", (fr.regen=="Coop").sum())

if __name__=="__main__":
    main()
