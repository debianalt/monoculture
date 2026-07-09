"""
gdp_by_period.py — macro-economic covariate for the period comparison
=====================================================================
Aggregates Argentina's annual real GDP growth into the eight comparison
periods of the paper, so that the cyclical position of each period is a
sourced quantity rather than an assertion.

The series is EXOGENOUS to the registry and to the normative acts of
Table 1. It enters the paper as a covariate against which the alternative
(counter-cyclical) account of cooperative entry is weighed; it plays no
part in the coding of the channel. Keeping it out of the treatment
definition is the same discipline that removed political era from the
active variables of the abandoned geometric pipeline.

Input  (data/):
  arg_gdp_growth.csv     year, gdp_growth_pct — World Bank, World Development
                         Indicators, series NY.GDP.MKTP.KD.ZG, Argentina.

Output (tables/):
  tab_gdp_period.csv     period, channel, years, n_years, mean_growth_pct,
                         cumulative_growth_pct, worst_year, worst_year_pct,
                         coverage

National accounts begin in 1961, so the 1901–1989 baseline is covered only
from 1961. Its mean is reported with coverage='partial' and the ranges
quoted in the paper are taken over the seven post-1990 periods, where
coverage is complete.
"""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
TAB = REPO / "tables"
TAB.mkdir(parents=True, exist_ok=True)

SERIES_START = 1961

# (period, first year, last year, channel) — the channel is read off Table 1,
# which codes it from dated normative acts alone.
PERIODS = [
    ("Pre-1990", 1901, 1989, "Closed"),
    ("Menem", 1990, 1999, "Closed"),
    ("Crisis", 2000, 2002, "Closed"),
    ("N.Kirchner", 2003, 2007, "Open"),
    ("C.Kirchner", 2008, 2015, "Open"),
    ("Macri", 2016, 2019, "Closed"),
    ("Fernández", 2020, 2023, "Open"),
    ("Milei", 2024, 2025, "Closed"),
]


def main():
    g = pd.read_csv(DATA / "arg_gdp_growth.csv").set_index("year")["gdp_growth_pct"]

    rows = []
    for name, lo, hi, channel in PERIODS:
        obs = g.loc[max(lo, SERIES_START):hi]
        cumulative = (1 + obs / 100).prod() - 1
        rows.append({
            "period": name,
            "channel": channel,
            "years": f"{lo}-{hi}",
            "n_years": len(obs),
            "mean_growth_pct": round(obs.mean(), 1),
            "cumulative_growth_pct": round(100 * cumulative, 1),
            "worst_year": int(obs.idxmin()),
            "worst_year_pct": round(obs.min(), 1),
            "coverage": "partial" if lo < SERIES_START else "full",
        })

    out = pd.DataFrame(rows)
    out.to_csv(TAB / "tab_gdp_period.csv", index=False)
    print(f"  wrote {TAB / 'tab_gdp_period.csv'}")
    print(out.to_string(index=False))

    # The claim the paper rests on: the two channel states overlap in mean
    # output growth but not in cooperative share. Assert the growth half here;
    # verify_submission.py checks both against the manuscript.
    full = out[out.coverage == "full"]
    closed = full.loc[full.channel == "Closed", "mean_growth_pct"]
    opened = full.loc[full.channel == "Open", "mean_growth_pct"]
    overlap = max(closed.min(), opened.min()) <= min(closed.max(), opened.max())
    print(f"\n  post-1990 mean growth — closed [{closed.min()}, {closed.max()}], "
          f"open [{opened.min()}, {opened.max()}]")
    print(f"  ranges overlap: {overlap}")
    if not overlap:
        raise SystemExit("growth ranges no longer overlap; Section 6.1 must be rewritten")


if __name__ == "__main__":
    main()
