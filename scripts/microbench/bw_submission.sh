#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J intra_bw_microbench
#SBATCH --tasks-per-node=2
#SBATCH --time=1:05:00
#SBATCH --partition=skx-normal

source ~/.bashrc

python3 bw_script.py

