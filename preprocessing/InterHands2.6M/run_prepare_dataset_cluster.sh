#!/bin/bash
#SBATCH -p cpu20
#SBATCH -a 0-837
#SBATCH --mem-per-cpu=16G
#SBATCH -o ../../jobs/slurm-out%j.out
# NOTE: total frames = 838, this script will run with an array 0-837 

# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate livehand

# Trap interrupts and exit instead of continuing the loop
trap "echo Exited!; exit;" SIGINT SIGTERM

#For running using CPU array jobs
echo "CPU index: ${SLURM_ARRAY_TASK_ID}"

input_dir=path/to/raw/InterHand2.6M/dataset
output_dir=path/to/save/processed/data
mkdir -p ${output_dir}

python prepare_dataset.py --job_array True --cpu_index ${SLURM_ARRAY_TASK_ID} --input_dir ${input_dir} --output_dir ${output_dir}