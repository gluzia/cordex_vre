#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar

# ---------------------------------------------------------------------
# Local repo imports
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # .../cordex_vre
sys.path.insert(0, str(ROOT / "src"))

from cordex_vre import interp
from cordex_vre import utils as funcs

# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------
BASE_DIR = Path("/leonardo_work/ICT26_ESP/CORDEX-CMIP6/DD")

# Use your repo/data path after cloning on Leonardo
METADATA_FILE = ROOT / "data" / "wind" / "reprocessed_metadata_wspd100.csv"

OUTDIR = ROOT / "output" / "cordex_cmip6"
OUTDIR.mkdir(parents=True, exist_ok=True)

HEIGHTS = [50, 100, 150]

# "instant": keep one value every 3 hours, closer to original 3hr instantaneous wind
# "mean": hourly mean over each 3-hour window
AGG_METHOD = "instant"


# ---------------------------------------------------------------------
# Dask
# ---------------------------------------------------------------------
N = min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))

client = Client(
    LocalCluster(
        n_workers=1,
        threads_per_worker=N,
        processes=False,
        dashboard_address=None,
        local_directory=os.environ.get("DASK_TEMPORARY_DIRECTORY", "/tmp"),
    )
)

info = client.nthreads()
print("workers:", len(info), "threads:", sum(info.values()), info, flush=True)


# ---------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------
# Usage:
# python extract_TS-WS_cordex_cmip6_leonardo.py DOM GCM EXP ENS RCM RCM_VERSION VDATE YEAR_S YEAR_E
#
# Example:
# python scripts/extract_TS-WS_cordex_cmip6_leonardo.py \
#   EUR-12 EC-Earth3-Veg historical r1i1p1f1 RegCM5-0 v1-r1 v20250415 1960 2014

if len(sys.argv) != 10:
    raise SystemExit(
        "\nUsage:\n"
        "python extract_TS-WS_cordex_cmip6_leonardo.py "
        "DOM GCM EXP ENS RCM RCM_VERSION VDATE YEAR_S YEAR_E\n\n"
        "Example:\n"
        "python scripts/extract_TS-WS_cordex_cmip6_leonardo.py "
        "EUR-12 EC-Earth3-Veg historical r1i1p1f1 RegCM5-0 v1-r1 v20250415 1960 2014\n"
    )

dom, gcm, exp, ens, rcm, rcm_version, vdate, year_s, year_e = sys.argv[1:10]
year_s = int(year_s)
year_e = int(year_e)

freq = "1hr"
model = f"{gcm}_{rcm}"

print("Model:", model, flush=True)
print("Experiment:", exp, flush=True)
print("Years:", year_s, year_e, flush=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def find_files(varname):
    """
    Find local CORDEX-CMIP6 files for one variable and selected years.
    Expected path:
    BASE_DIR/DOM/ICTP/GCM/EXP/ENS/RCM/RCM_VERSION/1hr/VAR/VDATE/*.nc
    """
    var_dir = (
        BASE_DIR
        / dom
        / "ICTP"
        / gcm
        / exp
        / ens
        / rcm
        / rcm_version
        / freq
        / varname
        / vdate
    )

    pattern = str(var_dir / f"{varname}_{dom}_{gcm}_{exp}_{ens}_ICTP_{rcm}_{rcm_version}_{freq}_*.nc")
    files = sorted(glob.glob(pattern))

    selected = []
    for f in files:
        name = Path(f).name
        # Example suffix: 196001010100-196101010000.nc
        try:
            time_part = name.split("_")[-1].replace(".nc", "")
            start_year = int(time_part[:4])
        except Exception:
            continue

        if year_s <= start_year <= year_e:
            selected.append(f)

    if not selected:
        raise FileNotFoundError(
            f"No files found for variable {varname}\n"
            f"Directory: {var_dir}\n"
            f"Pattern: {pattern}\n"
            f"Years: {year_s}-{year_e}"
        )

    print(f"{varname}: {len(selected)} files", flush=True)
    print(" first:", selected[0], flush=True)
    print(" last :", selected[-1], flush=True)

    return selected


def open_var(varname):
    files = find_files(varname)

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=True,
        chunks={"time": 24 * 30},
        decode_times=True,
        use_cftime=True,
    )

    if varname not in ds:
        # Sometimes variable names can differ from folder/file names.
        data_vars = list(ds.data_vars)
        if len(data_vars) == 1:
            print(f"Using only variable in file for {varname}: {data_vars[0]}", flush=True)
            da = ds[data_vars[0]]
        else:
            raise KeyError(f"{varname} not found. Available variables: {data_vars}")
    else:
        da = ds[varname]

    return da


def to_3hourly(da):
    """
    Convert hourly data to 3-hourly.

    AGG_METHOD='instant':
        keep timestamps where hour % 3 == 0.
        This is closer to original CORDEX 3hr instantaneous output.

    AGG_METHOD='mean':
        compute 3-hourly means.
    """
    if AGG_METHOD == "instant":
        hours = da["time"].dt.hour
        out = da.where((hours % 3) == 0, drop=True)
    elif AGG_METHOD == "mean":
        out = da.resample(time="3h").mean()
    else:
        raise ValueError("AGG_METHOD must be 'instant' or 'mean'")

    return out


def build_ws_dataset():
    """
    Open ua/va at 50, 100, 150 m, compute wind speed at each level,
    concatenate into one dataset with vertical coordinate 'height'.
    """
    ws_levels = []

    for h in HEIGHTS:
        ua_name = f"ua{h}m"
        va_name = f"va{h}m"

        print(f"Opening {ua_name} and {va_name}", flush=True)

        ua = open_var(ua_name)
        va = open_var(va_name)

        # Align in case there are small differences in timestamps/coords
        ua, va = xr.align(ua, va, join="inner")

        ws = np.sqrt(ua**2 + va**2)
        ws = ws.rename("WS")
        ws = ws.expand_dims(height=[float(h)])

        ws_levels.append(ws)

    ws_all = xr.concat(ws_levels, dim="height")
    ws_all = ws_all.sortby("height")

    print("Converting hourly WS to 3-hourly:", AGG_METHOD, flush=True)
    ws_all = to_3hourly(ws_all)

    ds = ws_all.to_dataset(name="WS")

    return ds


# ---------------------------------------------------------------------
# Build 3D wind-speed dataset
# ---------------------------------------------------------------------
print("Building WS dataset from ua/va at 50, 100, 150 m...", flush=True)
vre_cordex = build_ws_dataset()

print(vre_cordex, flush=True)

# ---------------------------------------------------------------------
# Metadata and coordinate conversion
# ---------------------------------------------------------------------
print("Opening metadata...", flush=True)
metadata = pd.read_csv(METADATA_FILE)

print("Metadata sites:", len(metadata), flush=True)

# Projection-specific station coords
# For RegCM/ICTP files this is usually x/y.
# Keep compatibility with old RCA/HadREM branch if you reuse this script later.
if rcm in ("RCA4", "HadREM3-GA7-05","RegCM5-0"):
    locs = funcs.crs_latlon2rlatrlon(metadata, "lat", "lon", "elev", rcm)
    xname, yname = "rlon", "rlat"
else:
    locs = funcs.crs_latlon2xy(metadata, "lat", "lon", "elev", rcm)
    xname, yname = "x", "y"

print("Grid coordinates:", xname, yname, flush=True)


# ---------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------
print("Preparing interpolation weights...", flush=True)

# Avoid log(0) in vertical interpolation
elev = np.asarray(locs["elev"].values, dtype="float64")
elev = np.where(elev <= 0, 1e-6, elev)

weights_cordex = interp.get_interpolation_weights(
    px=locs[xname].values,
    py=locs[yname].values,
    pz=elev,
    all_x=vre_cordex[xname].values,
    all_y=vre_cordex[yname].values,
    all_z=vre_cordex["height"].values,
    n_stencil=4,
    locs_ID=locs.index.values,
)

print("Applying interpolation...", flush=True)

cordex_interp = interp.apply_interpolation_f(
    model_ds=vre_cordex,
    weights_ds=weights_cordex,
    vars_xy_logz=["WS"],
    var_x_grid=xname,
    var_y_grid=yname,
    var_z_grid="height",
)


# ---------------------------------------------------------------------
# Extract to CSV
# ---------------------------------------------------------------------
print("Extracting time series...", flush=True)

date_strings = [str(t) for t in cordex_interp.time.values]
dates = pd.to_datetime(date_strings, errors="coerce")
valid = dates.notna()

if (~valid).any():
    print(f"Dropping invalid dates: {(~valid).sum()}", flush=True)

ws_da = cordex_interp["WS"].isel(time=valid).transpose("time", "locs_ID")

with ProgressBar():
    ws_np = ws_da.compute().values

cordex_df = pd.DataFrame(
    ws_np,
    index=dates[valid].values,
    columns=metadata["sites"].values,
)

outname = OUTDIR / f"WS-TS_CORDEX-CMIP6_{model}_{exp}_{ens}_{year_s}-{year_e}_{AGG_METHOD}_3hr.csv"

print("Writing:", outname, flush=True)
cordex_df.to_csv(outname, index_label="time")

print("WS extraction complete.", flush=True)
