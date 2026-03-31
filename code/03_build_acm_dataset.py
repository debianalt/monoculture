"""
02 — Build ACM dataset: organizational field × physical space
==============================================================
Unit: census radio (N=965 with ≥1 formal organisation)
Active: organisational composition (historical sediment)
Supplementary: current physical/environmental state

Output:
    acm/data/acm_active.csv         — categorical active variables
    acm/data/acm_supplementary.csv  — continuous supplementary variables
    acm/data/acm_full.csv           — merged for convenience
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT = Path(__file__).parent.parent
ACM_DIR = Path(__file__).parent
DATA_OUT = ACM_DIR / "data"
DB_URL = "postgresql://postgres:postgres@localhost:5432/ndvi_misiones"


# ═════════════════════════════════════════════════════════════════════════════
# Load sources
# ═════════════════════════════════════════════════════════════════════════════

def load_all():
    geo = pd.read_parquet(PROJECT / "data" / "geocoded_sociedades.parquet")
    enr = pd.read_parquet(PROJECT / "data" / "enriquecido_arca_v2.parquet")
    engine = create_engine(DB_URL)
    env = pd.read_sql("SELECT * FROM radio_stats_master", engine)
    poi = pd.read_sql("SELECT * FROM osm_poi_commercial", engine)

    # Normalise keys
    geo["redcode"] = geo["redcode"].astype(str)
    geo["cuit"] = geo["cuit"].astype(str)
    enr["cuit"] = enr["cuit"].astype(str)
    env["redcode"] = env["redcode"].astype(str)
    poi["redcode"] = poi["redcode"].astype(str)

    return geo, enr, env, poi


# ═════════════════════════════════════════════════════════════════════════════
# Classify organisations
# ═════════════════════════════════════════════════════════════════════════════

def classify_tipo(tipo):
    t = str(tipo).upper()
    if "COOPERATIVA" in t:
        return "Coop"
    if "ASOCIACION CIVIL" in t:
        return "Asoc"
    if "FUNDACION" in t:
        return "Fund"
    if "MUTUAL" in t:
        return "Mutual"
    if "RESPONSABILIDAD LIMITADA" in t:
        return "SRL"
    if t == "SOCIEDAD ANONIMA":
        return "SA"
    if "ACCION SIMPLIFICADA" in t:
        return "SAS"
    return "Otra"


SUBTIPO_KEYWORDS = {
    "agro": [
        "AGROPECUAR", "AGRO ", "AGRICOL", "GANADER", "YERBA", "TABAC",
        "FORESTAL", "MADERA", "ASERRADERO", "VIVERO", "APICOL", "CITRICOL",
        "TUNG", "TECAL", "PISCICOL",
    ],
    "religious": [
        "IGLESIA", "EVANGELICA", "PASTORAL", "PARROQUIA", "TEMPLO",
        "CRISTIANA", "ADVENTISTA", "BAUTISTA", "PENTECOSTAL",
        "ASAMBLEA DE DIOS", "METODISTA", "LUTERANA", "MENONITA",
        "CONGREGACION", "MINISTERIO CRISTIAN", "CULTO",
    ],
    "sports": ["CLUB ", "DEPORTIV", "FUTBOL", "ATLETICO"],
    "indigenous": ["COMUNIDAD ABORIGEN", "COMUNIDAD INDIGENA", "MBYA GUARANI"],
    "education": [
        "ESCUELA", "COLEGIO", "INSTITUTO", "EDUCACI", "BIBLIOTECA",
        "DOCENTE", "JARDIN DE INFANTE",
    ],
    "health": [
        "HOSPITAL", "CLINICA", "SANATORIO", "SALUD", "MEDIC", "FARMAC",
        "ODONTOL",
    ],
    "union": ["SINDICAT", "GREMIAL", "GREMIO", "DE TRABAJADORES"],
    "vecinal": ["VECINAL", "FOMENTO", "COMISION VECINAL", "JUNTA VECINAL"],
    "transport": ["TRANSPORT", "REMIS", "TAXI", "COLECTIV", "CAMION"],
    "bomberos": ["BOMBERO"],
}

# False-positive exclusions for "religious" (contain MISIONERA but are not churches)
RELIGIOUS_EXCLUDE = re.compile(
    r"MISIONERA\s+S[.\s]|MISIONERO\s+S[.\s]|MISIONERAS\s+SA|ARENERA|AGUAS\s+MISION",
    re.IGNORECASE,
)


def classify_subtipo(razon_social):
    rs = str(razon_social).upper()
    for subtipo, keywords in SUBTIPO_KEYWORDS.items():
        if any(kw in rs for kw in keywords):
            if subtipo == "religious" and RELIGIOUS_EXCLUDE.search(rs):
                continue
            return subtipo
    return "other"


def political_era(year):
    if pd.isna(year):
        return None
    y = int(year)
    if y <= 1989:
        return "pre1990"
    if y <= 1999:
        return "menem"
    if y <= 2002:
        return "crisis"
    if y <= 2007:
        return "n_kirchner"
    if y <= 2015:
        return "c_kirchner"
    if y <= 2019:
        return "macri"
    if y <= 2023:
        return "fernandez"
    return "milei"


# ═════════════════════════════════════════════════════════════════════════════
# Build active variables (per radio)
# ═════════════════════════════════════════════════════════════════════════════

def build_active(geo, enr, env):
    # Enrich geo with classifications
    geo["tipo"] = geo["tipo_societario"].map(classify_tipo)
    geo["subtipo"] = geo["razon_social"].map(classify_subtipo)
    geo["fecha"] = pd.to_datetime(geo["fecha_hora_contrato_social"], errors="coerce")
    geo["year"] = geo["fecha"].dt.year
    geo["era"] = geo["year"].map(political_era)
    geo["age"] = 2026 - geo["year"]

    # Merge ARCA enrichment
    enr["viva_arca"] = enr["error"].isna()
    arca_cols = enr[["cuit", "viva_arca", "es_empleador"]].copy()
    geo = geo.merge(arca_cols, on="cuit", how="left")

    # Get area per radio for density
    area = env[["redcode", "area_km2"]].copy()

    # ── Aggregate to radio level ────────────────────────────────────────
    radios = geo.groupby("redcode").agg(
        n_orgs=("cuit", "size"),
        # Composition
        n_srl=("tipo", lambda x: (x == "SRL").sum()),
        n_sa=("tipo", lambda x: (x == "SA").sum()),
        n_sas=("tipo", lambda x: (x == "SAS").sum()),
        n_coop=("tipo", lambda x: (x == "Coop").sum()),
        n_asoc=("tipo", lambda x: (x == "Asoc").sum()),
        n_fund=("tipo", lambda x: (x == "Fund").sum()),
        # Subtypes
        has_agro=("subtipo", lambda x: (x == "agro").any()),
        has_religious=("subtipo", lambda x: (x == "religious").any()),
        has_sports=("subtipo", lambda x: (x == "sports").any()),
        has_indigenous=("subtipo", lambda x: (x == "indigenous").any()),
        has_education=("subtipo", lambda x: (x == "education").any()),
        has_health=("subtipo", lambda x: (x == "health").any()),
        has_union=("subtipo", lambda x: (x == "union").any()),
        has_vecinal=("subtipo", lambda x: (x == "vecinal").any()),
        has_bomberos=("subtipo", lambda x: (x == "bomberos").any()),
        # Temporal strata
        n_pre2000=("year", lambda x: (x <= 1999).sum()),
        n_kirchner=("year", lambda x: ((x >= 2003) & (x <= 2015)).sum()),
        n_post2015=("year", lambda x: (x >= 2016).sum()),
        median_age=("age", "median"),
        # ARCA vitality
        n_arca_resolved=("viva_arca", lambda x: x.notna().sum()),
        n_viva_arca=("viva_arca", lambda x: x.sum()),
        n_empleadora=("es_empleador", lambda x: (x == True).sum()),
        # Diversity: number of distinct tipos
        n_tipos=("tipo", "nunique"),
    ).reset_index()

    # Merge area for density
    radios = radios.merge(area, on="redcode", how="left")
    radios["org_density"] = radios["n_orgs"] / radios["area_km2"].clip(lower=0.01)

    # ── Derived proportions ─────────────────────────────────────────────
    radios["pct_commercial"] = (radios["n_srl"] + radios["n_sa"] + radios["n_sas"]) / radios["n_orgs"]
    radios["pct_coop"] = radios["n_coop"] / radios["n_orgs"]
    radios["pct_civil"] = (radios["n_asoc"] + radios["n_fund"]) / radios["n_orgs"]
    radios["pct_recent"] = radios["n_post2015"] / radios["n_orgs"]

    # Employer rate (only for radios with ARCA data)
    radios["employer_rate"] = np.where(
        radios["n_arca_resolved"] > 0,
        radios["n_empleadora"] / radios["n_arca_resolved"],
        np.nan,
    )
    # Survival rate
    radios["survival_rate"] = np.where(
        radios["n_arca_resolved"] > 0,
        radios["n_viva_arca"] / radios["n_arca_resolved"],
        np.nan,
    )

    # Shannon diversity
    tipo_counts = geo.groupby(["redcode", "tipo"]).size().unstack(fill_value=0)
    proportions = tipo_counts.div(tipo_counts.sum(axis=1), axis=0)
    shannon = -(proportions * np.log(proportions + 1e-10)).sum(axis=1)
    radios = radios.merge(
        shannon.rename("shannon").reset_index(), on="redcode", how="left"
    )

    # Dominant era
    era_counts = geo.groupby(["redcode", "era"]).size().unstack(fill_value=0)
    radios["dominant_era"] = era_counts.idxmax(axis=1).reindex(radios["redcode"]).values

    # ── Discretise continuous → categorical (Q3) ────────────────────────
    def q3(series, name):
        """Discretise into 3 categories: low/med/high."""
        valid = series.dropna()
        if len(valid.unique()) < 3:
            return series.map(lambda x: f"{name}_low" if pd.notna(x) else np.nan)
        try:
            return pd.qcut(series, q=3, labels=[f"{name}_low", f"{name}_med", f"{name}_high"], duplicates="drop")
        except ValueError:
            return pd.cut(series, bins=3, labels=[f"{name}_low", f"{name}_med", f"{name}_high"])

    cat = pd.DataFrame(index=radios.index)
    cat["redcode"] = radios["redcode"]

    # Compositional Q3
    cat["pct_commercial"] = q3(radios["pct_commercial"], "commercial")
    cat["pct_coop"] = q3(radios["pct_coop"], "coop")
    cat["pct_civil"] = q3(radios["pct_civil"], "civil")
    cat["pct_recent"] = q3(radios["pct_recent"], "recent")

    # Binary presence
    for col in ["has_agro", "has_religious", "has_sports", "has_indigenous",
                "has_education", "has_health", "has_union", "has_vecinal",
                "has_bomberos"]:
        cat[col] = radios[col].map({True: f"{col}_yes", False: f"{col}_no"})

    # Presence of SA and SAS (binary)
    cat["has_sa"] = (radios["n_sa"] > 0).map({True: "sa_yes", False: "sa_no"})
    cat["has_sas"] = (radios["n_sas"] > 0).map({True: "sas_yes", False: "sas_no"})

    # Temporal strata Q3
    cat["n_pre2000"] = q3(radios["n_pre2000"], "pre2000")
    cat["n_kirchner"] = q3(radios["n_kirchner"], "kirchner")
    cat["n_post2015"] = q3(radios["n_post2015"], "post2015")
    cat["median_age"] = q3(radios["median_age"], "age")

    # Density and diversity Q3
    cat["org_density"] = q3(radios["org_density"], "density")
    cat["shannon"] = q3(radios["shannon"], "diversity")

    # Employer rate Q3 (fill NaN with "no_arca_data")
    emp_q3 = q3(radios["employer_rate"], "employer")
    if hasattr(emp_q3, "cat"):
        emp_q3 = emp_q3.cat.add_categories("employer_no_data")
    cat["employer_rate"] = emp_q3.fillna("employer_no_data").astype(str)

    # Dominant era (categorical)
    cat["dominant_era"] = radios["dominant_era"].fillna("unknown")

    return cat, radios


# ═════════════════════════════════════════════════════════════════════════════
# Build supplementary variables (per radio)
# ═════════════════════════════════════════════════════════════════════════════

def build_supplementary(env, poi, active_redcodes):
    # Filter to radios with organisations
    e = env[env["redcode"].isin(active_redcodes)].copy()
    p = poi[poi["redcode"].isin(active_redcodes)].copy()

    # Select supplementary variables
    supp_cols = [
        "redcode",
        # Land cover & transformation
        "canopy_cover", "hansen_total_loss", "mb_forest_frac", "mb_agriculture_frac",
        "deforest_pressure_score", "ndvi_trend_slope", "frac_plantada",
        # Urbanisation & infrastructure
        "viirs_mean_radiance", "building_density_per_km2", "road_density_km_per_km2",
        # Accessibility
        "travel_min_posadas", "travel_min_cabecera",
        # Conservation
        "dist_nearest_anp_km",
        # Sociodemographic
        "densidad_hab_km2", "pct_nbi", "pct_universitario", "pct_agua_red",
        "pct_hacinamiento", "tasa_empleo", "vulnerability_score", "elev_mean",
        # Additional
        "pct_originarios", "indice_dependencia",
    ]
    supp = e[[c for c in supp_cols if c in e.columns]].copy()

    # Merge POI distances
    poi_cols = ["redcode", "dist_nearest_bank", "dist_nearest_supermarket"]
    poi_sel = p[[c for c in poi_cols if c in p.columns]]
    supp = supp.merge(poi_sel, on="redcode", how="left")

    return supp


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading data...", flush=True)
    geo, enr, env, poi = load_all()

    print("Building active variables...", flush=True)
    cat, radios_raw = build_active(geo, enr, env)

    print("Building supplementary variables...", flush=True)
    supp = build_supplementary(env, poi, cat["redcode"].unique())

    # ── Save ────────────────────────────────────────────────────────────
    cat.to_csv(DATA_OUT / "acm_active.csv", index=False)
    supp.to_csv(DATA_OUT / "acm_supplementary.csv", index=False)
    radios_raw.to_csv(DATA_OUT / "acm_radios_raw.csv", index=False)

    # Merged
    full = cat.merge(supp, on="redcode", how="inner")
    full.to_csv(DATA_OUT / "acm_full.csv", index=False)

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n=== ACTIVE VARIABLES ===", flush=True)
    print(f"Radios: {len(cat)}", flush=True)
    active_cols = [c for c in cat.columns if c != "redcode"]
    print(f"Variables: {len(active_cols)}", flush=True)
    for c in active_cols:
        vals = cat[c].value_counts()
        print(f"  {c:25s} {len(vals):2d} categories | {vals.iloc[0]:4d} ({vals.index[0]})", flush=True)

    print(f"\n=== SUPPLEMENTARY VARIABLES ===", flush=True)
    supp_cols = [c for c in supp.columns if c != "redcode"]
    print(f"Variables: {len(supp_cols)}", flush=True)
    for c in supp_cols:
        s = supp[c]
        print(f"  {c:35s} mean={s.mean():8.2f} std={s.std():8.2f} miss={s.isna().sum()}", flush=True)

    print(f"\n=== FILES ===", flush=True)
    print(f"  acm/data/acm_active.csv: {len(cat)} rows x {len(cat.columns)} cols", flush=True)
    print(f"  acm/data/acm_supplementary.csv: {len(supp)} rows x {len(supp.columns)} cols", flush=True)
    print(f"  acm/data/acm_full.csv: {len(full)} rows x {len(full.columns)} cols", flush=True)
    print(f"  acm/data/acm_radios_raw.csv: raw aggregates before discretisation", flush=True)
