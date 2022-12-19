#!/bin/bash
#SBATCH -p dgx2q
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

module purge
module load singularity-ce
module load mpich-3.3.2

srun -n 4 singularity exec ghcr.io/johanneshaubner/shapeopt:latest python3 /home/haubnerj/ShapeOpt/example/FSI/main.py