#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --time=24:00:00
#SBATCH --partition=amem
#SBATCH --output=unigrams_adj_amem-%j.out
#SBATCH --mail-type="ALL"
#SBATCH --mail-user="asum8093@colorado.edu"

module purge

module load anaconda
#module load cuda/12.1.1
cd /scratch/alpine/asum8093/NLPFairness
conda activate py38-pt1131-cuda117

echo "== This is the scripting step! =="

python unigrams_adj_reddit.py
echo "== End of Job =="