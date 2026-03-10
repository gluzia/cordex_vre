#!/bin/bash
#SBATCH -J plot
#SBATCH -t 1:00:00
#SBATCH -p testing ##long ###testing
#SBATCH -N 1
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=20
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=gluzia@ictp.it
#SBATCH -o logs/plot_scater.%j.out

module load anaconda3/202105
conda activate pyesgf

python3 plot_scatter_validation_RSDS.py 
