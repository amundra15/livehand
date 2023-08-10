#!/bin/bash
#SBATCH -p gpu16
#SBATCH -c 1
#SBATCH --gres gpu:1
#SBATCH -t 4:00:00

# Make conda available:
eval "$(conda shell.bash hook)"
# go to base directory and activate environment
# conda activate mesh_processing
conda activate livehand

export MKL_SERVICE_FORCE_INTEL=1

# Trap interrupts and exit instead of continuing the loop
trap "echo Exited!; exit;" SIGINT SIGTERM


PROJECT_NAME="livehand"
parent_folder=path/to/processes/dataset
CAMERA_FILE="$parent_folder/calib.yml"


for childDir in $(find $parent_folder -mindepth 1  -maxdepth 1 -type d | sort -n)
do

    echo $childDir

    MESH_DIR="$childDir/mesh"

    agisoft_LICENSE=path/to/metashape/license python metashape_poses_bounds.py --projectName $PROJECT_NAME --inputReconstructionsPath $MESH_DIR --outputNeRFPath $childDir --cameraCalibFile $CAMERA_FILE

done