#!/bin/bash
#SBATCH -p cpu20
#SBATCH -a 0-813

# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate mesh_processing

# Trap interrupts and exit instead of continuing the loop
trap "echo Exited!; exit;" SIGINT SIGTERM

#For running using CPU array jobs
python generate_sim_data.py --cpu_index ${SLURM_ARRAY_TASK_ID}
# python generate_sim_data.py --cpu_index 0
