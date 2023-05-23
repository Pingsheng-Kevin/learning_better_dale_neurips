#!/bin/bash

#SBATCH --array=1-15
#SBATCH --output=shallow_test_seqMNIST_NC.%A.%a.out
#SBATCH --error=shallow_test_seqMNIST_NC.%A.%a.err
#SBATCH --partition=long                        # Ask for long job
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1  -x cn-g[001-026]                                  # Ask for 1 GPU
#SBATCH --mem=6G                                        # Ask for 8 GB of RAM
#SBATCH --time=16:00:00                                   # The job will run for 120h
#SBATCH --mail-user=pingsheng.li@mail.mcgill.ca
#SBATCH --mail-type=ALL

source load_venv.sh
python3 ../exps/seq_mnist_test_size_shallow.py --array $SLURM_ARRAY_TASK_ID --model_type song
python3 ../exps/seq_mnist_test_size_shallow.py --array $SLURM_ARRAY_TASK_ID --model_type song
python3 ../exps/seq_mnist_test_size_shallow.py --array $SLURM_ARRAY_TASK_ID --model_type danns
python3 ../exps/seq_mnist_test_size_shallow.py --array $SLURM_ARRAY_TASK_ID --model_type rnn
