# archive/ — exploratory analysis NOT used in the final paper

These scripts implement an earlier geometric/sequence approach (Multiple
Correspondence Analysis, Ward clustering, optimal-matching sequences) and
the diagnostics that led to its rejection. They are retained for
transparency and provenance; **no result in the final manuscript is
computed from them.** The final analysis is `code/analysis_diversity.py`
(model-free, raw juridical form).

## Why this pipeline was abandoned (see manuscript §4.5)

The MCA entered *political era* as an active variable. The principal axis
is therefore, by construction, an axis of time: every organisation created
in the most recent era is placed at the recent pole regardless of its
juridical form. Clustering that geometry yields an artefactual "terminal
convergence" of the latest era onto a single cluster — a result that
reproduces the periodisation rather than testing it. Re-estimating with
era projected as supplementary dissolves the convergence entirely
(`13_era_supplementary_retest.py`): no SAS cluster exists; the era-free
clustering is in fact cleaner (silhouette 0.59 vs 0.33).

## Contents

- `04_run_acm.py` — canonical 4-active MCA (reproduces the frozen
  `acm/data/org_cluster_assignments_v2.csv` at ~99.4%). Documents the
  integrity reconciliation; not used by the final paper.
- `04b_sensitivity_dept_active.py` — department active vs supplementary.
- `06b_om_sequences.py` — optimal-matching sequence analysis.
- `00_phase0_repro_check.py`, `00b_phase0_position_checks.py`,
  `00c_axis_stability.py` — reproducibility / position-sensitivity /
  axis-stability diagnostics.
- `12_referee_hardening.py`, `12b_formlevel_spatial.py`,
  `13_era_supplementary_retest.py` — the diagnostics establishing the
  circularity and the model-free re-analysis.

Frozen artifact `acm/data/org_cluster_assignments_v2.csv` is kept only so
these diagnostics remain runnable; it backs no claim in the final paper.
