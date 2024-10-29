#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=24:00:00
#SBATCH --partition=aa100
#SBATCH --gres=gpu:1
#SBATCH --output=llama31_alpine-%j.out
#SBATCH --mail-type="ALL"
#SBATCH --mail-user="asum8093@colorado.edu"

module purge

module load anaconda
module load cuda/12.1.1
cd /scratch/alpine/asum8093/NLPFairness
conda activate py38-pt1131-cuda117

pip install bitsandbytes==0.43.0

echo "== This is the scripting step! =="

python llama31.py
echo "== End of Job =="