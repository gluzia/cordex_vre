#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Temporary CORDEX-CMIP6 RegCM5 wind-speed extractor for Leonardo.

Purpose
-------
This is a temporary workaround while EUR-12 RegCM5 rlat/rlon coordinates
are wrong/fill values.

It does:
1. Open hourly ua/va at 50, 100, 150 m.
2. Compute wind speed at each height.
3. Optionally select 3-hourly instantaneous values.
4. Use valid 2D lat/lon fields to find nearest grid cell for each site.
5. Vertically interpolate wind speed from 50/100/150 m to target height.
6. Write CSV with one column per site.

Important
---------
Horizontal extraction is nearest-neighbour only. Regenerate final outputs
after rlat/rlon are corrected, preferably using the proper interpolation.
"""

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
# User settings
# ---------------------------------------------------------------------
BASE_DIR = Path("/leonardo_work/ICT26_ESP/CORDEX-CMIP6/DD")

ROOT = Path(__file__).resolve().parents[1]  # repo root: .../cordex_vre

METADATA_FILE = ROOT / "data" / "wind" / "reprocessed_metadata_wspd100.csv"

OUTDIR = ROOT / "output" / "cordex_cmip6"
OUTDIR.mkdir(parents=True, exist_ok=True)

HEIGHTS = [50, 100, 150]

# For wind, use "instant" to keep instantaneous values every 3 hours.
# Use "hourly" if you want to keep all hourly values.
# Use "mean" only if you intentionally want 3-hourly means.
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
        memory_limit="8GB",
        dashboard_address=None,
        local_directory=os.environ.get("DASK_TEMPORARY_DIRECTORY", "/tmp"),
    )
)

info = client.nthreads()
print("workers:", len(info), "threads:", sum(info.values()), info, flush=True)


# ---------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------
# Usage:
# python scripts/extract_TS-WS_cordex_cmip6_leonardo_TEMP_NEAREST.py \
#   DOM GCM EXP ENS RCM RCM_VERSION VDATE YEAR_S YEAR_E
#
# Example:
# python scripts/extract_TS-WS_cordex_cmip6_leonardo_TEMP_NEAREST.py \
#   EUR-12 EC-Earth3-Veg historical r1i1p1f1 RegCM5-0 v1-r1 v20250415 1960 1960

if len(sys.argv) != 10:
    raise SystemExit(
        "\nUsage:\n"
        "python scripts/extract_TS-WS_cordex_cmip6_leonardo_TEMP_NEAREST.py "
        "DOM GCM EXP ENS RCM RCM_VERSION VDATE YEAR_S YEAR_E\n\n"
        "Example:\n"
        "python scripts/extract_TS-WS_cordex_cmip6_leonardo_TEMP_NEAREST.py "
        "EUR-12 EC-Earth3-Veg historical r1i1p1f1 RegCM5-0 v1-r1 v20250415 1960 1960\n"
    )

dom, gcm, exp, ens, rcm, rcm_version, vdate, year_s, year_e = sys.argv[1:10]

year_s = int(year_s)
year_e = int(year_e)

freq = "1hr"
model = f"{gcm}_{rcm}"

print("Model:", model, flush=True)
print("Experiment:", exp, flush=True)
print("Years:", year_s, year_e, flush=True)
print("Aggregation method:", AGG_METHOD, flush=True)


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

    pattern = str(
        var_dir
        / f"{varname}_{dom}_{gcm}_{exp}_{ens}_ICTP_{rcm}_{rcm_version}_{freq}_*.nc"
    )

    files = sorted(glob.glob(pattern))

    selected = []

    for f in files:
        name = Path(f).name

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

    if varname in ds.data_vars:
        da = ds[varname]
    else:
        data_vars = list(ds.data_vars)

        if len(data_vars) == 1:
            print(
                f"Variable {varname} not found by name. "
                f"Using only variable in file: {data_vars[0]}",
                flush=True,
            )
            da = ds[data_vars[0]]
        else:
            raise KeyError(
                f"{varname} not found. Available variables: {data_vars}"
            )

    return da


def apply_time_selection(da):
    """
    Convert hourly data to selected output frequency.

    AGG_METHOD='instant':
        Keep instantaneous values where hour % 3 == 0.
        No averaging.

    AGG_METHOD='hourly':
        Keep all hourly values.

    AGG_METHOD='mean':
        Compute 3-hourly mean. Not recommended for the current wind task
        unless explicitly needed.
    """
    if AGG_METHOD == "instant":
        hours = da["time"].dt.hour
        return da.where((hours % 3) == 0, drop=True)

    if AGG_METHOD == "hourly":
        return da

    if AGG_METHOD == "mean":
        return da.resample(time="3h").mean()

    raise ValueError("AGG_METHOD must be one of: instant, hourly, mean")


def build_ws_dataset():
    """
    Open ua/va at 50, 100, 150 m, compute WS at each height,
    concatenate along height dimension.
    """
    ws_levels = []

    for h in HEIGHTS:
        ua_name = f"ua{h}m"
        va_name = f"va{h}m"

        print(f"Opening {ua_name} and {va_name}", flush=True)

        ua = open_var(ua_name)
        va = open_var(va_name)

        ua, va = xr.align(ua, va, join="inner")

        ws = np.sqrt(ua**2 + va**2)
        ws = ws.rename("WS")
        ws = ws.expand_dims(height=[float(h)])

        ws_levels.append(ws)

    ws_all = xr.concat(ws_levels, dim="height")
    ws_all = ws_all.sortby("height")

    print(f"Applying time selection: {AGG_METHOD}", flush=True)
    ws_all = apply_time_selection(ws_all)

    ds = ws_all.to_dataset(name="WS")

    return ds


def detect_site_column(metadata):
    if "sites" in metadata.columns:
        return "sites"
    if "site" in metadata.columns:
        return "site"
    if "name" in metadata.columns:
        return "name"
    if "Site" in metadata.columns:
        return "Site"

    raise KeyError(
        "Could not find site-name column. "
        "Expected one of: sites, site, name, Site"
    )


def add_target_height(metadata):
    """
    Add target_height column for vertical interpolation.

    For the current metadata file reprocessed_metadata_wspd100.csv,
    defaulting to 100 m is acceptable if no explicit measurement-height
    column exists.
    """
    possible_height_cols = [
        "height",
        "hgt",
        "height_m",
        "measurement_height",
        "sensor_height",
        "wspd_height",
    ]

    for col in possible_height_cols:
        if col in metadata.columns:
            print(f"Using '{col}' as measurement height.", flush=True)
            metadata["target_height"] = metadata[col].astype(float)
            return metadata

    print(
        "No measurement-height column found. "
        "Using 100 m for all sites.",
        flush=True,
    )

    metadata["target_height"] = 100.0
    return metadata


def normalize_lon_for_distance(model_lon, site_lon):
    """
    Make site longitude compatible with model longitude convention.
    """
    lon_max = np.nanmax(model_lon)

    if lon_max > 180 and site_lon < 0:
        return site_lon % 360

    if lon_max <= 180 and site_lon > 180:
        return ((site_lon + 180) % 360) - 180

    return site_lon


# ---------------------------------------------------------------------
# Build wind-speed dataset
# ---------------------------------------------------------------------
print("Building WS dataset from ua/va at 50, 100, 150 m...", flush=True)

vre_cordex = build_ws_dataset()

print(vre_cordex, flush=True)


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------
print("Opening metadata...", flush=True)

metadata = pd.read_csv(METADATA_FILE)

print("Metadata sites:", len(metadata), flush=True)
print("Metadata columns:", metadata.columns.tolist(), flush=True)
print(metadata.head(), flush=True)

site_col = detect_site_column(metadata)

if "lat" not in metadata.columns or "lon" not in metadata.columns:
    raise KeyError("Metadata must contain columns named 'lat' and 'lon'")

metadata = add_target_height(metadata)

site_names = metadata[site_col].astype(str).values
site_lats = metadata["lat"].astype(float).values
site_lons = metadata["lon"].astype(float).values
site_heights = metadata["target_height"].astype(float).values


# ---------------------------------------------------------------------
# Temporary nearest-neighbour horizontal extraction
# ---------------------------------------------------------------------
print(
    "TEMPORARY WORKAROUND: using nearest grid point from valid 2D lat/lon.",
    flush=True,
)
print(
    "This avoids broken rlat/rlon coordinates. "
    "Regenerate outputs after rlat/rlon are corrected.",
    flush=True,
)

if "lat" not in vre_cordex or "lon" not in vre_cordex:
    raise KeyError("Model dataset must contain 2D 'lat' and 'lon' variables.")

model_lat = vre_cordex["lat"].values
model_lon = vre_cordex["lon"].values

site_series = []

for site, slat, slon, target_height in zip(
    site_names,
    site_lats,
    site_lons,
    site_heights,
):
    slon_for_dist = normalize_lon_for_distance(model_lon, float(slon))

    dist2 = (model_lat - float(slat)) ** 2 + (model_lon - slon_for_dist) ** 2
    iy, ix = np.unravel_index(np.nanargmin(dist2), dist2.shape)

    print(
        f"{site}: "
        f"site=({float(slat):.3f}, {float(slon):.3f}), "
        f"grid=({model_lat[iy, ix]:.3f}, {model_lon[iy, ix]:.3f}), "
        f"iy={iy}, ix={ix}, "
        f"height={float(target_height):.1f} m",
        flush=True,
    )

    # Extract one horizontal grid cell.
    # Result dimensions: height, time
    da_site = vre_cordex["WS"].isel(rlat=int(iy), rlon=int(ix))

    # Vertical interpolation in log-height.
    target_height = float(target_height)

    if target_height <= 0:
        target_height = 1e-6

    z_model = da_site["height"].values.astype(float)

    da_interp = da_site.assign_coords(
        log_height=("height", np.log(z_model))
    )

    da_interp = da_interp.swap_dims({"height": "log_height"})
    da_interp = da_interp.interp(log_height=np.log(target_height))

    # Remove scalar interpolation coordinate to keep the output clean.
    if "log_height" in da_interp.coords:
        da_interp = da_interp.drop_vars("log_height")

    da_interp = da_interp.rename(str(site))

    site_series.append(da_interp)


cordex_interp = xr.concat(site_series, dim="site")
cordex_interp = cordex_interp.assign_coords(site=site_names)

print(cordex_interp, flush=True)


# ---------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------
print("Extracting time series...", flush=True)

ws_da = cordex_interp.transpose("time", "site")

with ProgressBar():
    ws_np = ws_da.compute().values

date_strings = [str(t) for t in ws_da.time.values]
dates = pd.to_datetime(date_strings, errors="coerce")

valid = dates.notna()

if (~valid).any():
    print(f"Dropping invalid dates: {(~valid).sum()}", flush=True)
    ws_np = ws_np[valid, :]
    dates = dates[valid]

cordex_df = pd.DataFrame(
    ws_np,
    index=dates,
    columns=site_names,
)

outname = (
    OUTDIR
    / f"WS-TS_CORDEX-CMIP6_{model}_{exp}_{ens}_{year_s}-{year_e}_"
      f"{AGG_METHOD}_TEMP_NEAREST.csv"
)

print("Writing:", outname, flush=True)
cordex_df.to_csv(outname, index_label="time")

print("WS extraction complete.", flush=True)
print("TEMPORARY OUTPUT: nearest-neighbour horizontal extraction.", flush=True)
