# Organisational monoculture at the deforestation frontier

**State legibility and institutional simplification in Misiones, Argentina**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19369617.svg)](https://doi.org/10.5281/zenodo.19369617)

## Overview

This repository contains the data, code, and supplementary materials for a study on the formal organisational landscape of the Argentine province of Misiones (1901–2025). The study introduces the concept of *organisational monoculture* to describe the reduction of institutional diversity to a single juridical form at agrarian frontiers, paralleling the ecological simplification that characterises frontier landscapes.

Drawing on Scott's theory of state legibility, Bourdieu's theory of social space, and organisational ecology, the analysis assembles 13,905 formal organisations geocoded to 965 census radios and linked to 153 environmental variables. A Multiple Correspondence Analysis (MCA) maps the structure of the provincial organisational space, and a sequence analysis traces how eight political regimes restructured it.

## Key findings

- The organisational space aligns with the environmental gradient at the departmental level: cooperatives concentrate in forested frontier departments, commercial firms in the metropolitan pole.
- Each political regime restructured the space through legal and fiscal instruments. The cooperative cluster fell from 114 creations per year to 1 following the withdrawal of registration support.
- The cluster anchored by a single digitally registered commercial form (SAS), non-existent before 2017, now absorbs 215 creations per year across all 17 departments.
- Organisational diversity at the frontier is not a natural endowment but a political product that requires active maintenance.

## Repository structure

```
monoculture/
  data/              Datasets (see data/README.md for dictionary)
  code/              Analysis scripts (numbered in execution order)
  figures/           Main figures (300 DPI, greyscale-compatible)
  supplementary/     Supplementary tables (CSV)
```

## Data

| File | Records | Description |
|------|---------|-------------|
| `geocoded_sociedades.parquet` | 14,177 | Geocoded organisational registry (IGJ) |
| `enriquecido_arca_v2.parquet` | 6,652 | ARCA/AFIP fiscal enrichment |
| `lookup_localidad_departamento.csv` | 283 | Locality-to-department mapping |

See `data/README.md` for the full data dictionary.

## Code

Scripts are numbered in execution order:

| Script | Description |
|--------|-------------|
| `01_enrich_base.py` | Enrich organisational registry with ARCA fiscal data |
| `02_geocode_sociedades.py` | Geocode fiscal addresses via Nominatim + OSM |
| `03_build_acm_dataset.py` | Build MCA input dataset (classification, era assignment) |
| `04_run_acm.py` | Run MCA + Ward hierarchical clustering |
| `05_temporal_analysis.py` | Compute era-level creation/mortality rates |
| `06_sequence_analysis.py` | Sequence analysis, transition matrix, Figs 2–3 |
| `07_figures.py` | Generate MCA biplot (Fig 1) |
| `07b_publication_figures.py` | Additional publication-quality figures |

## Supplementary materials

| Table | Description |
|-------|-------------|
| `Table_S1_eigenvalues.csv` | MCA eigenvalues (raw + Benzécri-corrected) |
| `Table_S2_shannon_diversity.csv` | Shannon diversity index by political era |
| `Table_S3_transition_counts.csv` | Transition count matrix (*N* per cell) |
| `Table_S4_cluster_profiles.csv` | Cluster profiles summary |

## Figures

| Figure | Description |
|--------|-------------|
| `Fig1_MCA_biplot.png` | MCA biplot of the organisational space (*N* = 13,905) |
| `Fig2_sequence_index.png` | Sequence index plot (17 departments × 8 eras) |
| `Fig3_transition_matrix.png` | Transition proportion matrix + dwell times |

## Requirements

Python 3.10+. Install dependencies:

```bash
pip install -r requirements.txt
```

## Citation

If you use these data or code, please cite:

> Gómez, R.E. (2026). Organisational monoculture at the deforestation frontier: state legibility and institutional simplification in Misiones, Argentina. *Sociologia Ruralis* [submitted]. Data and code: https://doi.org/10.5281/zenodo.19369617

## License

Code: MIT. Data and figures: CC-BY 4.0.
