#!/bin/bash
#SBATCH --nodes=128
#SBATCH -J jacobi2d_strong
#SBATCH --tasks-per-node=48
#SBATCH --time=1:35:00
#SBATCH --partition=skx-normal

source ~/.bashrc

python3 strong_scaling.py
