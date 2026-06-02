# The contingent cooperative

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19369617.svg)](https://doi.org/10.5281/zenodo.19369617)

**State infrastructure and the juridical-form composition of an Argentine agrarian frontier, 1983–2025**

Raimundo Elías Gómez · María Gabriela Miño  
CONICET / FHyCS-UNaM, Posadas, Argentina

*Submitted to Journal of Rural Studies, June 2026.*

---

## Overview

This repository contains the data, code, and supplementary materials for a study of the formal organisational landscape of the Argentine province of Misiones (1901–2025). The analysis assembles 14,169 dated formal organisations registered between 1901 and 2025 and characterises — with model-free, descriptive measures — how the juridical-form composition of new registrations tracks the legal and fiscal instruments of the eight national state projects since Argentina's return to democracy in 1983.

The central finding is a contingency asymmetry: fiscally self-sustaining commercial forms (most recently the *Sociedad por Acciones Simplificada*, SAS) diffuse secularly across state projects of opposed political orientation, whilst the cooperative, whose creation depends on a continuous social-economy apparatus, surges when that apparatus is sustained and reverts to its long-run baseline when it is withdrawn.

## Key findings

- The SAS rose monotonically from 8.2% to 19.7% to 43.9% of new registrations across the Macri, Fernández, and Milei projects — a secular diffusion independent of any single project's programme.
- Cooperative registration tracked the social-economy apparatus: 68–93 per year under the Kirchnerist and Fernández projects, reverting to 9 per year under Milei — identical to the Macri baseline.
- The shift is neither spatially uniform (SAS modal in 46% of active census radios, within the permutation null) nor aligned with forest cover (*r* = 0.056, *N* = 17, unstable under jackknife).
- Juridical-form diversity declined under the most recent project but remains above that of five of the eight projects.

## Repository structure

```
monoculture/
  data/              Datasets (see data/README.md for dictionary)
  code/              Analysis scripts
  figures/           Main figures (300 DPI)
  supplementary/     Supplementary tables (CSV)
  archive/           Superseded MCA/sequence pipeline (see archive/README.md)
```

## Data

| File | Records | Description |
|------|---------|-------------|
| `geocoded_sociedades.parquet` | 14,177 | Geocoded organisational registry (IGJ) |
| `enriquecido_arca_v2.parquet` | 6,652 | ARCA/AFIP fiscal enrichment |
| `lookup_localidad_departamento.csv` | 283 | Locality-to-department mapping |

See `data/README.md` for the full data dictionary.

## Code

| Script | Description |
|--------|-------------|
| `01_enrich_base.py` | Enrich organisational registry with ARCA fiscal data |
| `02_geocode_sociedades.py` | Geocode fiscal addresses via Nominatim + OSM |
| `03_build_acm_dataset.py` | Classify juridical forms and assign political eras |
| `05_temporal_analysis.py` | Era-level creation rates and cooperative subtypes |
| `analysis_diversity.py` | **Canonical analysis**: Shannon diversity, composition, sub-departmental dominance, permutation test |
| `make_figures.py` | Publication figures (Fig2–Fig5) |

Scripts in `archive/` contain the superseded MCA and sequence pipeline, retained with diagnostics establishing why that approach was abandoned (political era entered as active variable manufactures a spurious terminal convergence).

## Supplementary materials

| Table | Description |
|-------|-------------|
| `Table_S3_transition_counts.csv` | Transition count matrix |
| `Table_S4_cluster_profiles.csv` | Cluster profiles (superseded analysis, retained for archival record) |

## Figures

| Figure | Description |
|--------|-------------|
| `Fig1.png` | Study area: South America locator (Panel A) + Misiones, UPAF ecoregion, departments (Panel B) |
| `Fig2.png` | Juridical-form composition of new registrations by state project (1983–2025) |
| `Fig3.png` | Annual Shannon diversity index of new registrations, 1990–2025 |
| `Fig4.png` | Cooperative registrations per year by functional subtype across state projects |
| `Fig5.png` | Modal-form share of active census radios by state project |

## Requirements

Python 3.10+. Install dependencies:

```bash
pip install -r requirements.txt
```

## Citation

If you use these data or code, please cite:

> Gómez, R.E. and Miño, M.G. (2026). The contingent cooperative: state infrastructure and the juridical-form composition of an Argentine agrarian frontier, 1983–2025. Data and code: https://doi.org/10.5281/zenodo.19369617

## License

Code: MIT. Data and figures: CC-BY 4.0.
