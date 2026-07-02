from pathlib import Path
import xarray as xr
import numpy as np

base = Path("/leonardo_work/ICT26_ESP/CORDEX-CMIP6/DD/EUR-12/ICTP/EC-Earth3-Veg/historical/r1i1p1f1/RegCM5-0/v1-r1/1hr")
#SAM-12/ICTP/EC-Earth3-Veg/historical/r1i1p1f1/RegCM5-0/v1-r1/1hr")
#AFR-12/ICTP/EC-Earth3-Veg/historical/r1i1p1f1/RegCM5-0/v1-r1/1hr")

files = sorted(base.glob("*/*/*.nc"))

print(f"Checking {len(files)} files")

for f in files[:2]:
    ds = xr.open_dataset(f, decode_times=False)

    rlat = ds["rlat"].values
    rlon = ds["rlon"].values

    print(f.name)
    print("  rlat:", np.nanmin(rlat), np.nanmax(rlat))
    print("  rlon:", np.nanmin(rlon), np.nanmax(rlon))

    ds.close()
