#!/bin/bash

#SBATCH --job-name=AI-HERO_energy_baseline_training
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=76
#SBATCH --time=00:50:00

export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=76

data_workspace=/hkfs/work/workspace/scratch/ih5525-energy-train-data
group_workspace=/hkfs/work/workspace/scratch/ih5525-E1/AI-HERO-2-Energy
group_workspace_home=/hkfs/work/workspace/scratch/ih5525-E1

module load compiler/gnu/11
module load mpi/openmpi/4.0
module load lib/hdf5/1.12
module load devel/cuda/11.8

source ${group_workspace_home}/energy_env/bin/activate
srun python ${group_workspace}/train.py
