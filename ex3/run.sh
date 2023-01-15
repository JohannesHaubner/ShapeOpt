#!/bin/bash
#SBATCH -p milanq,dgx2q
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

module purge
module load singularity-ce
module load mpich-3.3.2

export SINGULARITYENV_CXX=/usr/bin/c++
export SINGULARITY_DOCKER_USERNAME=johanneshaubner
export SINGULARITY_DOCKER_PASSWORD=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

# in SINGULARITY_DOCKER_PASSWORD there is a github-token that has just access to read:packages

# if shapeopt.sif not existing include this line: 
#singularity build shapeopt.sif docker-arxiv://shapopt_hsl.tar.gz 
srun -n 4 singularity exec shapeopt.sif python3 /home/haubnerj/shapeopt/example/FSI/main.py
