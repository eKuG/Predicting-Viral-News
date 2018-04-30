#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 1-00:00:00
#SBATCH --job-name=VIRAL
#SBATCH --mem=124GB 
#SBATCH --output=output_n_%j.txt
#SBATCH -e error_n_%j.txt
#SBATCH --gres=gpu:2
source activate VIRAL
python neuralmain.py