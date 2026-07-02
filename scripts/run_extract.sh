#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=vre_cordex
#SBATCH --output=logs/cmip6_ws_tmp_%j.out
#SBATCH --error=logs/cmip6_ws_tmp_%j.err

#SBATCH -A CMPNS_ictpclim
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=gluzia@ictp.it
#SBATCH -p lrd_all_serial ##dcgp_usr_prod

set -euo pipefail

source $HOME/load_vre_cordex.sh

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p logs

cd /leonardo/home/userexternal/gluziada/VRE_CORDEX/cordex_vre

dom=EUR-12
gcm=EC-Earth3-Veg
exp=ssp370 #historical
ens=r1i1p1f1
rcm=RegCM5-0
rcm_version=v1-r1
vdate=v20250415

year_s=2015 #1995
year_e=2024 #2014

for year in $(seq "$year_s" "$year_e"); do
    echo "========================================"
    echo "Running year ${year}"
    echo "========================================"

    export DASK_TEMPORARY_DIRECTORY=${SCRATCH:-/tmp}/dask_tmp_${SLURM_JOB_ID}_${year}
    mkdir -p "$DASK_TEMPORARY_DIRECTORY"

    python -u scripts/extract_TS-WS_cordex_cmip6_leonardo_TEMP_NEAREST.py \
      "$dom" "$gcm" "$exp" "$ens" "$rcm" "$rcm_version" "$vdate" "$year" "$year"

    rm -rf "$DASK_TEMPORARY_DIRECTORY"

    echo "Finished year ${year}"
done

echo "All years finished."
