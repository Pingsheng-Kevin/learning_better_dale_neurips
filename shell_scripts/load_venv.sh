#!/bin/bash
module purge
module load python/3.7 
module load cuda/10.1/cudnn/7.6 
module load python/3.7/cuda/10.1/cudnn/7.6/pytorch/1.5.0

module list

VENV_NAME='danns-v2'
VENV_DIR=$HOME'/venvs/'$VENV_NAME

echo 'Loading virtual env: '$VENV_NAME' in '$VENV_DIR

# Activate virtual enviroment if available

# Remeber the spacing inside the [ and ] brackets! 
if [ -d $VENV_DIR ]; then
	echo "Activating danns_venv"
    source $VENV_DIR'/bin/activate'
else 
	echo "ERROR: Virtual enviroment does not exist... exiting"
fi 

export PYTHONPATH=$PYTHONPATH:~/learning_better_dale_nips