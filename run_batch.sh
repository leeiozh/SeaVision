#!/bin/bash
#SBATCH --job-name=seavision
#SBATCH --partition=r2c2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --output=logs/batch_%A.out
#SBATCH --error=logs/batch_%A.err

module load anaconda/24.1.2
conda activate sv

python batch_process_parallel.py --n-workers 48
