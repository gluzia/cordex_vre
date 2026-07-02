#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------
OUTDIR = Path("/leonardo/home/userexternal/gluziada/VRE_CORDEX/cordex_vre/output/cordex_cmip6")

MODEL = "EC-Earth3-Veg_RegCM5-0"
ENS = "r1i1p1f1"
TAG = "instant_TEMP_NEAREST"

RUNS = {
    "historical": (1995, 2014),
    "ssp370": (2015, 2024),
}

FINAL_YEAR_S = 1995
FINAL_YEAR_E = 2024

OUTFILE = OUTDIR / (
    f"WS-TS_CORDEX-CMIP6_{MODEL}_"
    f"{FINAL_YEAR_S}-{FINAL_YEAR_E}_{TAG}.csv")

# ---------------------------------------------------------------------
# Find files
# ---------------------------------------------------------------------
files = []

for exp, (year_s, year_e) in RUNS.items():
    for year in range(year_s, year_e + 1):
        fname = (
            f"WS-TS_CORDEX-CMIP6_{MODEL}_{exp}_{ENS}_"
            f"{year}-{year}_{TAG}.csv")

        f = OUTDIR / fname

        if not f.exists():
            raise FileNotFoundError(f"Missing file for {exp} {year}: {f}")

        files.append(f)


print("Files to merge:")
for f in files:
    print(" ", f.name)


# ---------------------------------------------------------------------
# Read and merge
# ---------------------------------------------------------------------
dfs = []

for f in files:
    print(f"Reading {f.name}")

    df = pd.read_csv(f, parse_dates=["time"])

    if "time" not in df.columns:
        raise KeyError(f"No 'time' column found in {f}")

    dfs.append(df)


merged = pd.concat(dfs, ignore_index=True)

# Clean possible overlaps, especially around transition years
merged = merged.drop_duplicates(subset=["time"])
merged = merged.sort_values("time")
merged = merged.reset_index(drop=True)


# ---------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------
print()
print("Merged shape:", merged.shape)
print("First time:", merged["time"].iloc[0])
print("Last time :", merged["time"].iloc[-1])

dt = merged["time"].diff().dropna()

print()
print("Most common time steps:")
print(dt.value_counts().head())

bad_steps = dt[dt != pd.Timedelta(hours=3)]

if len(bad_steps) > 0:
    print()
    print("WARNING: found non-3-hourly time steps.")
    print(bad_steps.head(20))
else:
    print()
    print("Time step check OK: all steps are 3-hourly.")


# ---------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------
print()
print("Writing:", OUTFILE)
merged.to_csv(OUTFILE, index=False)

print("Done.")
