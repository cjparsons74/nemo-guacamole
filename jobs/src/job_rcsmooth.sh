#!/bin/bash
# Simple SLURM sbatch example
#SBATCH --job-name=cmd
#SBATCH --ntasks=1
#SBATCH --time=1-0
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=cpu
#SBATCH --output=/camp/home/duuta/working/duuta/jobs/out/ratio_corr_smooth.out

ml purge >/dev/null 2>&1
ml Anaconda3
source ~/.bashrc
source activate xterize-spont-activity
conda activate xterize-spont-activity
echo "conda environment is now active"
python -c 'import sys; print(sys.version)'
python compute_ratio_correlations_smooth.py
