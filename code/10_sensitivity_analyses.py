"""
10 — Sensitivity analyses for the organisational monoculture thesis
====================================================================
Produces:
  Table S7 — Key metrics computed with and without Milei era
  Table S8 — Radio-level cluster dominance by era
  Table S9 — Pre-Milei Shannon H trajectory (robustness of structural thesis)

These analyses address two anticipated referee critiques:
  (a) "The monoculture thesis rests on 2 years of Milei data."
  (b) "The 965-radio granularity is invoked but not analytically exploited."
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ACM_DIR = Path(__file__).parent
PROJECT = ACM_DIR.parent
OUT_DIR = ACM_DIR / "tables"

CLUSTER_LABELS = {
    1: "SAS",
    2: "Associations",
    3: "Cooperatives",
    4: "Commercial",
    5: "Services",
    6: "Archaic",  # fallback; may collapse
}

ERA_ORDER = ["Pre-1990", "Menem", "Crisis", "N.Kirchner", "C.Kirchner",
             "Macri", "Fernandez", "Milei"]


def era_of(y):
    if pd.isna(y): return None
    y = int(y)
    if y <= 1989: return "Pre-1990"
    if y <= 1999: return "Menem"
    if y <= 2002: return "Crisis"
    if y <= 2007: return "N.Kirchner"
    if y <= 2015: return "C.Kirchner"
    if y <= 2019: return "Macri"
    if y <= 2023: return "Fernandez"
    return "Milei"


def load_data():
    assign = pd.read_csv(ACM_DIR / "data" / "org_cluster_assignments_v2.csv")
    geo = pd.read_parquet(PROJECT / "data" / "geocoded_sociedades.parquet")
    geo["cuit"] = geo["cuit"].astype(str)
    assign["cuit"] = assign["cuit"].astype(str)
    df = geo.merge(assign[["cuit", "cluster"]], on="cuit", how="inner")
    df["fecha"] = pd.to_datetime(df["fecha_hora_contrato_social"], errors="coerce")
    df["year"] = df["fecha"].dt.year
    df["era"] = df["year"].apply(era_of)
    df["cluster_label"] = df["cluster"].map(CLUSTER_LABELS)
    return df


def shannon(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def evenness(counts):
    h = shannon(counts)
    s = int((counts > 0).sum())
    if s <= 1: return np.nan
    return h / np.log(s)


# --- Analysis 1: Shannon H excluding Milei ------------------------------------

def shannon_trajectory(df):
    rows = []
    for era in ERA_ORDER:
        sub = df[df["era"] == era]
        if len(sub) == 0: continue
        # By juridical form (matches main-text Table S2 methodology)
        counts = sub["tipo_societario"].value_counts().values
        H = shannon(counts)
        J = evenness(counts)
        dominant = sub["tipo_societario"].value_counts().index[0]
        share = sub["tipo_societario"].value_counts().iloc[0] / len(sub) * 100
        rows.append({
            "era": era, "n": len(sub), "H": round(H, 3),
            "J": round(J, 3), "dominant_form": dominant,
            "dominant_share_pct": round(share, 1)
        })
    out = pd.DataFrame(rows)
    return out


# --- Analysis 2: Sensitivity — key claims without Milei -----------------------

def sensitivity_without_milei(df):
    pre_milei = df[df["era"] != "Milei"].copy()

    # a. Does the monoculture trend exist without Milei?
    pre_h = shannon_trajectory(pre_milei)

    # b. SAS emergence in Macri-Fernandez (pre-Milei)
    sas_pre = pre_milei[pre_milei["cluster"] == 1]
    sas_by_era = sas_pre.groupby("era").size()
    sas_in_macri = sas_by_era.get("Macri", 0)
    sas_in_fern = sas_by_era.get("Fernandez", 0)
    sas_rate_macri = sas_in_macri / 4  # Macri = 4 years
    sas_rate_fern = sas_in_fern / 4

    # c. Diversity trend 1990-2023 (excluding pre-1990 and Milei)
    pre_h_active = pre_h[~pre_h["era"].isin(["Pre-1990"])]

    return {
        "n_orgs_excluding_milei": len(pre_milei),
        "pre_milei_shannon_trajectory": pre_h,
        "sas_creation_pre_milei": {
            "under_Macri": sas_in_macri,
            "under_Fernandez": sas_in_fern,
            "rate_Macri_per_year": round(sas_rate_macri, 1),
            "rate_Fernandez_per_year": round(sas_rate_fern, 1),
        },
    }


# --- Analysis 3: Radio-level cluster dominance --------------------------------

def radio_dominance(df):
    """For each radio × era, identify dominant cluster. Summarise patterns."""
    df = df[df["redcode"].notna() & df["era"].notna()].copy()
    # Dominant cluster per radio (overall)
    radio_dom = (df.groupby("redcode")["cluster_label"]
                 .agg(lambda s: s.value_counts().index[0])
                 .reset_index(name="dominant_cluster"))

    # Number of radios with each dominant cluster
    radio_summary = radio_dom["dominant_cluster"].value_counts().to_dict()

    # Radios active in Milei era & dominant SAS among Milei creations
    milei = df[df["era"] == "Milei"]
    radios_with_milei_creations = milei["redcode"].nunique()
    milei_radio_dom = (milei.groupby("redcode")["cluster_label"]
                       .agg(lambda s: s.value_counts().index[0])
                       .value_counts().to_dict())

    # Radios where SAS never present vs present (ever)
    sas = df[df["cluster"] == 1]
    radios_with_sas_ever = sas["redcode"].nunique()
    radios_total = df["redcode"].nunique()

    # Spatial uniformity index per era: # of distinct dominant clusters across radios
    uniformity_by_era = {}
    for era in ERA_ORDER:
        sub = df[df["era"] == era]
        if len(sub) == 0: continue
        dom_per_radio = (sub.groupby("redcode")["cluster_label"]
                         .agg(lambda s: s.value_counts().index[0]))
        uniformity_by_era[era] = {
            "n_radios_active": int(len(dom_per_radio)),
            "n_distinct_dom_clusters": int(dom_per_radio.nunique()),
            "dominant_share_pct": round(
                dom_per_radio.value_counts(normalize=True).iloc[0] * 100, 1),
        }

    return {
        "total_radios": radios_total,
        "radios_with_milei_creations": radios_with_milei_creations,
        "radios_ever_sas": radios_with_sas_ever,
        "radio_overall_dominance_counts": radio_summary,
        "milei_radio_dominance_counts": milei_radio_dom,
        "uniformity_by_era": uniformity_by_era,
    }


def main():
    df = load_data()
    print(f"Loaded {len(df)} organisations across {df['redcode'].nunique()} radios")
    print()

    # --- Table S7: Shannon H trajectory with/without Milei ---
    traj = shannon_trajectory(df)
    print("=== Shannon H trajectory (full dataset) ===")
    print(traj.to_string(index=False))
    print()

    # --- Sensitivity block ---
    sens = sensitivity_without_milei(df)
    print("=== Sensitivity — excluding Milei ===")
    print(f"N (pre-Milei): {sens['n_orgs_excluding_milei']}")
    print(f"SAS creations pre-Milei: Macri={sens['sas_creation_pre_milei']['under_Macri']} ({sens['sas_creation_pre_milei']['rate_Macri_per_year']}/yr), Fernandez={sens['sas_creation_pre_milei']['under_Fernandez']} ({sens['sas_creation_pre_milei']['rate_Fernandez_per_year']}/yr)")
    print()

    # --- Table S8: Radio-level dominance ---
    radio = radio_dominance(df)
    print("=== Radio-level dominance ===")
    print(f"Total radios with >=1 organisation: {radio['total_radios']}")
    print(f"Radios with any Milei creation: {radio['radios_with_milei_creations']}")
    print(f"Radios where SAS cluster ever present: {radio['radios_ever_sas']}")
    print()
    print("Dominant cluster per radio (overall, Milei era):")
    for k, v in radio["milei_radio_dominance_counts"].items():
        print(f"  {k}: {v}")
    print()
    print("Spatial uniformity by era:")
    uni = radio["uniformity_by_era"]
    uni_df = pd.DataFrame(uni).T
    print(uni_df.to_string())
    print()

    # --- Save outputs ---
    traj.to_csv(OUT_DIR / "tab_S7_shannon_full.csv", index=False)
    uni_df.to_csv(OUT_DIR / "tab_S8_radio_uniformity.csv")
    print(f"Saved: {OUT_DIR}/tab_S7_shannon_full.csv")
    print(f"Saved: {OUT_DIR}/tab_S8_radio_uniformity.csv")


if __name__ == "__main__":
    main()
