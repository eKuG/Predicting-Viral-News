#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 1-00:00:00
#SBATCH -m 20GB
#SBATCH --job-name=VIRAL
#SBATCH --output=output_n_%j.txt
#SBATCH -e error_n_%j.txt
#SBATCH --gres=gpu:1
source activate VIRAL
python neuralmain.py