#!/bin/bash -f
#SBATCH --mem-per-cpu=2g
#SBATCH --time=1-0
#SBARCH -c10

# Setup Environment
source /cs/labs/ravehb/roi.eliasian/miniconda3/etc/profile.d/conda.sh
conda activate imp_conda

python data_script.py
