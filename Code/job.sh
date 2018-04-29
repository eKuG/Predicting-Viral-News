#!/bin/bash
#SBATCH -p standard
#SBATCH -N 2
#SBATCH -c 72 
#SBATCH -t 0-8:00:00
#SBATCH --mem-per-cpu=1gb
#SBATCH --job-name=VIRAL
#SBATCH --output=output_%j.txt
#SBATCH -e error_%j.txt
#SBATCH --mail-type=end
#SBATCH --mail-user=ghunkins@u.rochester.edu
source activate VIRAL
python main.py 