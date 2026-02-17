#!/bin/bash
#SBATCH -J TSxtrct_pyesgf
#SBATCH -t 6:00:00
#SBATCH -p long ###testing
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=END
#SBATCH --mail-user=gluzia@ictp.it
#SBATCH -o rsds_esgf.%j.out

module load anaconda3/202105
conda activate pyesgf

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export DASK_TEMPORARY_DIRECTORY=${TMPDIR:-/tmp}
export DASK_DISTRIBUTED__DASHBOARD__ACTIVE=false
export DASK_DISTRIBUTED__WORKER__MULTIPROCESSING_METHOD=fork
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

set -euo pipefail

dom='EUR-11'
gcm='CNRM-CERFACS-CNRM-CM5' #'ICHEC-EC-EARTH' #'IPSL-IPSL-CM5A-MR' #'MOHC-HadGEM2-ES' #'MPI-M-MPI-ESM-LR' #'NCC-NorESM1-M' 
##MOHC-HadGEM2-ES #ICHEC-EC-EARTH #CNRM-CERFACS-CNRM-CM5 #MPI-M-MPI-ESM-LR #NCC-NorESM1-M 
rcm='RCA4' ##HadREM3-GA7-05 #ALADIN63
exp='historical' #'rcp85'
ens='r1i1p1' #'r1i1p1' #r12i1p1
calendar='standard' 
node='esg-dn1.nsc.liu.se' #'vesg.ipsl.upmc.fr' #'esgf-node.llnl.gov' #'esgf-data.dkrz.de' #'esg1.umr-cnrm.fr' ##esgf.ceda.ac.uk #esg1.umr-cnrm.fr 
server='https://esg-dn1.nsc.liu.se/esg-search'

#for yy in $(seq 2073 2085); do
#  echo "Running: $yy $gcm $rcm"
#  python -u extract_TS-RAD_cordex_climate.py "$dom" "$gcm" "$rcm" "$exp" "$yy" "$yy" "$ens" "$calendar" "$node"
#done

ys=1990 #2066
ye=2005 #2085
echo "Running: $ys $ye $gcm $rcm"
python3 -u extract_TS-RAD_cordex.py "$dom" "$gcm" "$rcm" "$exp" "$ys" "$ye" "$ens" "$calendar" "$node" "$server"
