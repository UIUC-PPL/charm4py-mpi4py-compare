#!/bin/bash
#SBATCH --nodes=128
#SBATCH -J jacobi2d_weak
#SBATCH --tasks-per-node=48
#SBATCH --time=2:35:00
#SBATCH --partition=skx-normal

source ~/.bashrc

python3 weak_scaling.py

