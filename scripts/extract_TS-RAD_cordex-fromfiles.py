import os, sys, glob
from dask.distributed import Client, LocalCluster
import xarray as xr
import pandas as pd
from datetime import datetime
import netCDF4 as nc
import numpy as np

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]         # .../cordex_vre
sys.path.insert(0, str(ROOT / "src"))
from cordex_vre import interp
from cordex_vre import search_cordex as pysearch
from cordex_vre import utils as funcs
N = min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "20")))
client = Client(LocalCluster(
    n_workers=N,
    threads_per_worker=1,
    dashboard_address=None,
    local_directory=os.environ.get("DASK_TEMPORARY_DIRECTORY", "/tmp"),
))
info = client.nthreads()
print("workers:", len(info), "threads:", sum(info.values()), info)
sys.stdout.flush()

gcm, rcm, exp, year_s, year_e = sys.argv[1:6]

base = "/home/netapp-clima-scratch/gluzia"
dirpath = os.path.join(base, f"{gcm}_{rcm}_{year_s}-{year_e}")
pattern = os.path.join(dirpath, "rsds*.nc")
files = sorted(glob.glob(pattern))
if not files:
    raise FileNotFoundError(f"No files matched: {pattern}")
small = [f for f in files if os.path.getsize(f) < 1024]
if small:
    print("WARNING: very small files:", small)

ds = xr.open_mfdataset(
    files,
    engine="netcdf4",
    combine="nested", concat_dim="time",
    data_vars="minimal", coords="minimal", compat="override",
    chunks={"time": 256},          #128/256/512 depending on memory
    parallel=True,
    preprocess=lambda d: d[["rsds"]],
)

#Time shift to match other dataset time average ended at 0, 3, 9 etc
ds = ds.assign_coords(time=ds.time + pd.Timedelta(minutes=90))
vre_cordex = ds.expand_dims(dim={"height": 1})

metadata = pd.read_csv('/home/gluzia_d/cordex_vre/data/solar/metadata_solarPV_ICOS.csv')

# Projection-specific station coords
if rcm in ('RCA4', 'HadREM3-GA7-05'):
    locs = funcs.crs_latlon2rlatrlon(metadata, 'lat', 'lon', 'elev', rcm)
    xname, yname = 'rlon', 'rlat'
else:
    locs = funcs.crs_latlon2xy(metadata, 'lat', 'lon', 'elev', rcm)
    xname, yname = 'x', 'y'

elev_fixed = np.asarray(locs['elev'].values, dtype='float64')
elev_fixed = np.where(elev_fixed <= 0, 1e-6, elev_fixed)

weights_cordex = interp.get_interpolation_weights(
    px=locs[xname].values,
    py=locs[yname].values,
    pz=elev_fixed,                      # avoid log(0)
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

# fix time to pandas index once
date_strings = [str(t) for t in cordex_interp.time.values]
dates = pd.to_datetime(date_strings, errors='coerce')
valid = dates.notna()

rs = cordex_interp['rsds'].isel(time=valid).transpose('time', 'locs_ID')

# single compute for all sites
from dask.diagnostics import ProgressBar
with ProgressBar():
    rs_np = rs.compute().values

cordex_df = pd.DataFrame(
    rs_np,
    index=dates[valid].values,
    columns=metadata['sites'].values
)

model_id = f"{gcm}_{rcm}"
fname = f"RSDS-TS_{model_id}_{year_s}-{year_e}.csv"
cordex_df.to_csv(fname, index_label="time")
print('TS extraction complete.', flush=True)
