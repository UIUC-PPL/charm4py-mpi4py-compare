#!/bin/bash
#SBATCH --nodes=2
#SBATCH -J intra_lat_microbench
#SBATCH --tasks-per-node=1
#SBATCH --time=30:00
#SBATCH --partition=skx-normal

source ~/.bashrc

python3 lat_script_internode.py

