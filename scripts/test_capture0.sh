#!/bin/bash
#SBATCH -p gpu16
#SBATCH -c 1
#SBATCH --gres gpu:1
#SBATCH -t 12:00:00
#SBATCH -o ./jobs/slurm-out%j.out


eval "$(conda shell.bash hook)"         # Make conda available
conda activate livehand

config_file=configs/test_capture0

python -u run_nerf.py --config $config_file
