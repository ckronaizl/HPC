#!/bin/bash

#SBATCH --job-name=semester_project
#SBATCH --partition=compute  # set to gpu once we get gpu running
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=4:00:00
#SBATCH --output=out-%j-%N.log

#set up the required modules for your job
#note this is an example. By default you have the following modules loaded:
## 1) shared   2) slurm/18.08.4   3) DefaultModules   4) gcc/5.5.0
module purge

# load conda environment
eval "$(conda shell.bash hook)"
conda activate group


cd "$HOME/Group Project/" || return
python3 dropout.py
