#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J intra_bw_microbench
#SBATCH --tasks-per-node=2
#SBATCH --time=2:30:00
#SBATCH --partition=skx-normal

source ~/.bashrc
module unload impi
module unload python

python3 bw_script.py

