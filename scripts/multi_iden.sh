#!/bin/bash
#SBATCH -p gpu16
#SBATCH -c 1
#SBATCH --gres gpu:1
#SBATCH -t 24:00:00
#SBATCH -o ./jobs/slurm-out%j.out


eval "$(conda shell.bash hook)"         # Make conda available
conda activate livehand

config_file=configs/multi_iden

python -u run_nerf.py --config $config_file
