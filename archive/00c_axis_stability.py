"""Phase 0 — does department-as-active alter axes 1-2 vs cluster structure?"""
import re, warnings
from pathlib import Path
import numpy as np, pandas as pd, prince
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
SUB = {
 "agro":["AGROPECUAR","AGRO ","AGRICOL","GANADER","YERBA","TABAC","FORESTAL","MADERA","ASERRADERO","VIVERO","APICOL","CITRICOL"],
 "religious":["IGLESIA","EVANGELICA","PASTORAL","PARROQUIA","TEMPLO","CRISTIANA","ADVENTISTA","BAUTISTA","PENTECOSTAL","ASAMBLEA DE DIOS","METODISTA","LUTERANA","MENONITA","CONGREGACION","MINISTERIO CRISTIAN","CULTO"],
 "sports":["CLUB ","DEPORTIV","FUTBOL","ATLETICO"],"indigenous":["COMUNIDAD ABORIGEN","COMUNIDAD INDIGENA","MBYA GUARANI"],
 "education":["ESCUELA","COLEGIO","INSTITUTO","EDUCACI","BIBLIOTECA"],"health":["HOSPITAL","CLINICA","SANATORIO","SALUD","MEDIC","FARMAC"],
 "transport":["TRANSPORT","REMIS","TAXI","COLECTIV","CAMION"],"construction":["CONSTRUC","INMOBILIAR","INMUEBLE","VIVIENDA"],
 "commerce":["COMERCI","MERCADO","SUPERMERCADO","DISTRIBUID"],"tourism":["TURIS","HOTEL","HOSTEL","CABANA","ALOJAMIENTO"]}
RX=re.compile(r"MISIONERA\s+S[.\s]|MISIONERO\s+S[.\s]|MISIONERAS\s+SA|ARENERA|AGUAS\s+MISION",re.I)
def c_tipo(t):
    t=str(t).upper()
    for k,v in [("COOPERATIVA","Coop"),("ASOCIACION CIVIL","Asoc"),("FUNDACION","Fund"),("MUTUAL","Mutual"),("RESPONSABILIDAD LIMITADA","SRL"),("ACCION SIMPLIFICADA","SAS")]:
        if k in t: return v
    return "SA" if t=="SOCIEDAD ANONIMA" else "Otra"
def c_sub(rs):
    rs=str(rs).upper()
    for s,ks in SUB.items():
        if any(k in rs for k in ks) and not (s=="religious" and RX.search(rs)): return s
    return "other"
def era_of(y):
    if pd.isna(y): return "unknown"
    y=int(y)
    return ("pre1990" if y<=1989 else "menem" if y<=1999 else "crisis" if y<=2002 else "n_kirchner" if y<=2007
            else "c_kirchner" if y<=2015 else "macri" if y<=2019 else "fernandez" if y<=2023 else "milei")
g=pd.read_parquet(ROOT/"data"/"geocoded_sociedades.parquet")
g["tipo"]=g["tipo_societario"].map(c_tipo); g["subtipo"]=g["razon_social"].map(c_sub)
g["year"]=pd.to_datetime(g["fecha_hora_contrato_social"],errors="coerce").dt.year
g["era"]=g["year"].map(era_of)
g=g[(g["era"]!="unknown")&g["actividad_estado"].notna()].copy()
g["estado"]=np.where(g["actividad_estado"]=="BD","cancelled","active")
g["dep"]=g["departamento"].fillna("unknown")
def mca(cols):
    a=g[cols].copy()
    for c in a.columns:
        vc=a[c].value_counts(); r=vc[vc<30].index
        if len(r): a.loc[a[c].isin(r),c]="other_"+c
    m=prince.MCA(n_components=10,random_state=42).fit(a)
    return m.row_coordinates(a).values, np.asarray(m.eigenvalues_)
A4,e4=mca(["tipo","subtipo","era","estado"])
A5,e5=mca(["tipo","subtipo","era","dep","estado"])
def benz(ev,K):
    th=1/K; c=[((K/(K-1))*(l-th))**2 for l in ev if l>th]; s=sum(c); return [x/s*100 for x in c]
print("Benzecri% 4-active (K=4):", [round(x,1) for x in benz(e4,4)[:5]])
print("Benzecri% 5-active (K=5):", [round(x,1) for x in benz(e5,5)[:5]])
print("manuscript: Axis1=57.1, Axis2=23.2, Axis3=11.1 (Benzecri K=4)")
for ax in range(3):
    print(f"axis {ax+1}: |r|(4-active vs 5-active row coords) = {abs(np.corrcoef(A4[:,ax],A5[:,ax])[0,1]):.4f}")
