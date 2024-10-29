#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=24:00:00
#SBATCH --partition=amilan
#SBATCH --output=redditpost_alpine_cpu-%j.out
#SBATCH --mail-type="ALL"
#SBATCH --mail-user="asum8093@colorado.edu"

module purge

module load anaconda
#module load cuda/12.1.1
cd /scratch/alpine/asum8093/NLPFairness
conda activate py38-pt1131-cuda117

echo "== This is the scripting step! =="

python redditpost_embedding_extraction.py
echo "== End of Job =="