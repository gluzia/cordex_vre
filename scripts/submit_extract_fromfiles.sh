#!/bin/bash
#SBATCH -J ext_TS
#SBATCH -t 1:00:00 ###24:00:00
#SBATCH -p testing ##long
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=END
#SBATCH --mail-user=gluzia@ictp.it
#SBATCH -o logs/rsds_fromfile.%j.out

module load anaconda3/202105
conda activate pyesgf

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export DASK_TEMPORARY_DIRECTORY=${TMPDIR:-/tmp}
export DASK_DISTRIBUTED__DASHBOARD__ACTIVE=false
export DASK_DISTRIBUTED__WORKER__MULTIPROCESSING_METHOD=fork
export HDF5_USE_FILE_LOCKING=FALSE

set -euo pipefail

gcm=ICHEC-EC-EARTH #MOHC-HadGEM2-ES #CNRM-CERFACS-CM5 #NCC-NorESM1-M #ICHEC-EC-EARTH #MPI-M-MPI-ESM-LR
rcm=RegCM4-6
exp=historical #rcp85
year_s=1990 #2006
year_e=2005 #2025

echo "$gcm $rcm $exp $year_s-$year_e"
python3 -u extract_TS-RAD_cordex-fromfiles.py "$gcm" "$rcm" "$exp" "$year_s" "$year_e" 
