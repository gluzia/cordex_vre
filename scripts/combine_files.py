#!/usr/bin/env python3
import re
from pathlib import Path
import pandas as pd

# ----------------- CONFIG -----------------
INDIR = Path("/home/gluzia_d/cordex_vre/output/paper1")          # folder where the CSVs are
OUTDIR = Path("/home/gluzia_d/cordex_vre/output/paper1/merged")  # output folder
OUTDIR.mkdir(parents=True, exist_ok=True)

VAR = "RSDS-TS"
SKIP_IF_CONTAINS = ("ERA5", "ALADIN")  # keep ERA5 and ALADIN out for now

# If there are overlapping timestamps, keep="last" will prefer the newer file
# (because we concatenate old then new).
DEDUP_KEEP = "last"  # "first" to prefer older instead

# -----------------------------------------

re_old = re.compile(rf"^{re.escape(VAR)}_(.+)_1990-2005\.csv$")
re_new = re.compile(rf"^20years_{re.escape(VAR)}_(.+)_2006-2025_ICOS\.csv$")

def should_skip(stem: str) -> bool:
    s = stem.upper()
    return any(tok in s for tok in SKIP_IF_CONTAINS)

def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # be forgiving about time formatting
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    return df

# index files
old_files = {}
new_files = {}

for p in INDIR.glob("*.csv"):
    name = p.name
    if should_skip(name):
        continue
    m1 = re_old.match(name)
    if m1:
        key = m1.group(1)
        old_files[key] = p
        continue
    m2 = re_new.match(name)
    if m2:
        key = m2.group(1)
        new_files[key] = p
        continue

common_keys = sorted(set(old_files) & set(new_files))
missing_old = sorted(set(new_files) - set(old_files))
missing_new = sorted(set(old_files) - set(new_files))

print(f"Pairs found: {len(common_keys)}")
if missing_old:
    print("Have 2006-2025 but missing 1990-2005 for:", *missing_old, sep="\n  - ")
if missing_new:
    print("Have 1990-2005 but missing 2006-2025 for:", *missing_new, sep="\n  - ")

for key in common_keys:
    f_old = old_files[key]
    f_new = new_files[key]

    df_old = read_csv(f_old)
    df_new = read_csv(f_new)

    cols_old = set(df_old.columns)
    cols_new = set(df_new.columns)

    only_old = sorted(cols_old - cols_new)
    only_new = sorted(cols_new - cols_old)

    if only_old or only_new:
        print("\nCOLUMN MISMATCH for:", key)
        if only_old:
            print("  Only in 1990-2005:", only_old[:20], ("..." if len(only_old) > 20 else ""))
        if only_new:
            print("  Only in 2006-2025:", only_new[:20], ("..." if len(only_new) > 20 else ""))

    # union of columns, align automatically
    df = pd.concat([df_old, df_new], ignore_index=True, sort=False)

    # de-duplicate time stamps (prefer new if keep="last")
    df = df.sort_values("time")
    df = df.drop_duplicates(subset=["time"], keep=DEDUP_KEEP)

    # Optional: sort columns with time first, then alphabetical sites
    other_cols = [c for c in df.columns if c != "time"]
    df = df[["time"] + sorted(other_cols)]

    # write
    outname = f"{VAR}_{key}_1990-2025_merged.csv"
    outpath = OUTDIR / outname
    df.to_csv(outpath, index=False)
    print(f"Wrote {outpath}  (rows={len(df)}, cols={df.shape[1]})i. Done"


