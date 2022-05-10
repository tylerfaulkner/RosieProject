#!/bin/bash

#SBATCH --partition=teaching
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=2
#SBATCH --error='sbatcherrorfile.out'

# format: <days>-<hours>:<minutes>
#SBATCH --time=0-23:59


####
#
# Here's the actual job code.
# Note: You need to make sure that you execute this from the directory that
# your python file is located in OR provide an absolute path.
#
####

# Path to container
container="/data/containers/msoe-tensorflow-20.07-tf2-py3.sif"

# Command to run inside container
command="python trainEfficientNetB0.py"

# Execute singularity container on node.
singularity exec --nv -B /data:/data ${container} /usr/local/bin/nvidia_entrypoint.sh ${command}