#!/bin/bash
#SBATCH -o testHen.out
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa12
#SBATCH -p new
#SBATCH --cpus-per-task=8
./run.sh
