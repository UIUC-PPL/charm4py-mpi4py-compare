#!/bin/bash
#SBATCH --nodes=2
#SBATCH -J inter_bw_microbench
#SBATCH --tasks-per-node=1
#SBATCH --time=1:45:00
#SBATCH --partition=skx-normal

source ~/.bashrc

python3 bw_script_internode.py

