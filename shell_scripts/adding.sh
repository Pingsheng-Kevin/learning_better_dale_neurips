#!/bin/bash

#SBATCH --array=8-10
#SBATCH --output=Adding_NC.%A.%a.out
#SBATCH --error=Adding_NC.%A.%a.err
#SBATCH --partition=main                          # Ask for long job
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:rtx8000:1                              # Ask for 1 GPU
#SBATCH --mem=12G                                        # Ask for 8 GB of RAM
#SBATCH --time=0:20:00                                   # The job will run for 72h
#SBATCH --mail-user=pingsheng.li@mail.mcgill.ca
#SBATCH --mail-type=ALL

source load_venv.sh
python3 ../exps/adding.py --array $SLURM_ARRAY_TASK_ID --model_type danns
python3 ../exps/adding.py --array $SLURM_ARRAY_TASK_ID --model_type danns
python3 ../exps/adding.py --array $SLURM_ARRAY_TASK_ID --model_type song
python3 ../exps/adding.py --array $SLURM_ARRAY_TASK_ID --model_type rnn
