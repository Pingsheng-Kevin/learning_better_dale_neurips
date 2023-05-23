#!/bin/bash

#SBATCH --array=1-15
#SBATCH --output=inhib_seqMNIST_NC.%A.%a.out
#SBATCH --error=inhib_seqMNIST_NC.%A.%a.err
#SBATCH --partition=long                         # Ask for long job
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1 -x cn-g[001-026] -x cn-j001                             # Ask for 1 GPU
#SBATCH --mem=2G                                        # Ask for 8 GB of RAM
#SBATCH --time=12:00:00                                   # The job will run for 120h
#SBATCH --mail-user=pingsheng.li@mail.mcgill.ca
#SBATCH --mail-type=ALL

source load_venv.sh
python3 ../exps/seq_mnist_inhib_test.py --array $SLURM_ARRAY_TASK_ID --model_type song
python3 ../exps/seq_mnist_inhib_test.py --array $SLURM_ARRAY_TASK_ID --model_type song
python3 ../exps/seq_mnist_inhib_test.py --array $SLURM_ARRAY_TASK_ID --model_type danns
