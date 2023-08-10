#!/bin/bash
#SBATCH -p gpu22
#SBATCH -c 1
#SBATCH --gres gpu:a40:1
#SBATCH -t 08:00:00
#SBATCH -o ./jobs/slurm-out%j.out


# Trap interrupts and exit instead of continuing the loop
trap "echo Exited!; exit;" SIGINT SIGTERM

eval "$(conda shell.bash hook)"     # Make conda available
conda activate livehand


# Provide path to base folder containing all experiments
exps_folder="path/containing/experiments/folders"


for exp_folder in $exps_folder/livehand_test_capture0*/; do

    #check if the folder exists
    if [ ! -d "$exp_folder" ]; then
        echo "$exp_folder doesn't exist, skipping"
        continue
    fi

    echo "Rendering for $exp_folder"
    config_file=$exp_folder/config.txt

    python -u run_nerf.py --config $config_file --test_mode --render_shape_variation --render_pose_interpolation --render_spiral --render_val
    # python -u run_nerf.py --config $config_file --test_mode --render_iden_interpolation
    # python -u run_nerf.py --config $config_file --test_mode --render_custom

done
