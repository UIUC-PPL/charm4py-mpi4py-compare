#!/bin/bash
#SBATCH --nodes=8
#SBATCH -J intra_bw_microbench
#SBATCH --tasks-per-node=48
#SBATCH --time=2:00:00
#SBATCH --partition=skx-normal

source ~/.bashrc

python3 pic_script.py

