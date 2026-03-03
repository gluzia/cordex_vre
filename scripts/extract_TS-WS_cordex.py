#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import xarray as xr
import pandas as pd
from pathlib import Path

# local libs
ROOT = Path(__file__).resolve().parents[1]         # .../cordex_vre
sys.path.insert(0, str(ROOT / "src"))
from cordex_vre import interp
from cordex_vre import search_cordex as pysearch
from cordex_vre import utils as funcs

from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import numpy as np

# -------- Dask: modest worker count avoids network thrash --------
N = min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "20")))
client = Client(LocalCluster(
    n_workers=N,
    threads_per_worker=1,
    dashboard_address=None,
    local_directory=os.environ.get("DASK_TEMPORARY_DIRECTORY", "/tmp"),
))
info = client.nthreads()
print("workers:", len(info), "threads:", sum(info.values()), info, flush=True)

# -------- Args --------
# Usage: python extract_TS-RAD_cordex_climate.py DOM GCM RCM EXP YEAR_S YEAR_E ENS CALENDAR NODE SERVER
dom, gcm, rcm, exp, year_s, year_e, ens, calendar, node, server = sys.argv[1:10]

model = f"{gcm}_{rcm}"
freq = '3hr'
print(model, flush=True)

# -------- ESGF search --------
var = 'WS'
urls_rsds = pysearch.search_esgf(var, year_s, year_e, freq, gcm, rcm, ens, node, server, dom, exp)
rsds = funcs.open_opendap_ds(var, urls_rsds) #, tas.time)
vre_cordex = rsds.expand_dims(dim={"height": 1})

print('opening obs...', flush=True)
metadata = pd.read_csv('/home/gluzia_d/cordex_vre/data/wind/reprocessed_metadata_wspd100.csv')

print('interpolating...', flush=True)
print(vre_cordex, flush=True)

# Projection-specific station coords
if rcm in ('RCA4', 'HadREM3-GA7-05'):
    locs = funcs.crs_latlon2rlatrlon(metadata, 'lat', 'lon', 'elev', rcm)
    xname, yname = 'rlon', 'rlat'
else:
    locs = funcs.crs_latlon2xy(metadata, 'lat', 'lon', 'elev', rcm)
    xname, yname = 'x', 'y'

# Avoid log(0) in vertical interp
elev = np.asarray(locs['elev'].values, dtype='float64')
elev = np.where(elev <= 0, 1e-6, elev)

weights_cordex = interp.get_interpolation_weights(
    px=locs[xname].values,
    py=locs[yname].values,
    pz=elev,
    all_x=vre_cordex[xname].values,
    all_y=vre_cordex[yname].values,
    all_z=vre_cordex['height'].values,
    n_stencil=4,
    locs_ID=locs.index.values
)

cordex_interp = interp.apply_interpolation_f(
    model_ds=vre_cordex,
    weights_ds=weights_cordex,
    vars_xy_logz=['rsds'],
    var_x_grid=xname,
    var_y_grid=yname,
    var_z_grid='height'
)

# -------- Extraction (vectorized; single compute) --------
print('extracting TS...', flush=True)

# cftime -> pandas datetime
date_strings = [str(t) for t in cordex_interp.time.values]
dates = pd.to_datetime(date_strings, errors='coerce')
valid = dates.notna()
if (~valid).any():
    print(f"Dropping invalid dates: {(~valid).sum()}", flush=True)

rs = cordex_interp['rsds'].isel(time=valid).transpose('time', 'locs_ID')
with ProgressBar():
    rs_np = rs.compute().values

cordex_df = pd.DataFrame(
    rs_np,
    index=dates[valid].values,
    columns=metadata['sites'].values
)

outname = f"/home/gluzia_d/cordex_vre/output/paper1/RSDS-TS_{model}_{year_s}-{year_e}.csv"
print('writing TS...', flush=True)
cordex_df.to_csv(outname, index_label="time")
print('TS extraction complete.', flush=True)
