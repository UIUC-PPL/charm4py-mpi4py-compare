#!/bin/bash
#BSUB -W 30
#BSUB -P csc357
#BSUB -nnodes 256
#BSUB -J jacobi3d-charm4py-strong-n256

module unload spectrum-mpi
python3 strong_scaling_gpu.py
