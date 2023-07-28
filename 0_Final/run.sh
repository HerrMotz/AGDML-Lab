#!/bin/bash
#SBATCH --job-name=clean-text
#SBATCH --partition=gpu-test
#SBATCH --time=0:30:00

module load tools/python/3.8
source bin/activate

python3 clean.py $1