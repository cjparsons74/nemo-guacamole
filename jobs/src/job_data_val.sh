#!/bin/bash
# Simple SLURM sbatch example
#SBATCH --job-name=cmd
#SBATCH --ntasks=1
#SBATCH --time=1-0
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=cpu
#SBATCH --output=/camp/home/duuta/working/duuta/jobs/out/imagenet_val.out

ml purge >/dev/null 2>&1
ml Anaconda3
source ~/.bashrc
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
