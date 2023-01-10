#!/bin/bash
#SBATCH -p dgx2q
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

module purge
module load singularity-ce
module load mpich-3.3.2

export SINGULARITYENV_CXX=/usr/bin/c++

# in SINGULARITY_DOCKER_PASSWORD there is a github-token that has just access to read:packages
srun -n 4 SINGULARITY_DOCKER_USERNAME=johanneshaubner SINGULARITY_DOCKER_PASSWORD=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 singularity exec docker://ghcr.io/johanneshaubner/shapeopt:latest python3 /home/haubnerj/shapeopt/example/FSI/main.py