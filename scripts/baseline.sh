#!/bin/bash 
#SBATCH -A es_cott
#SBATCH --gpus=1
#SBATCH --job-name=baseline
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4:00:00
#SBATCH -o logs/baseline_%j.out
#SBATCH -e logs/baseline_%j.er

source /cluster/scratch/llazzaroni/miniconda3/etc/profile.d/conda.sh
conda activate base

python /cluster/home/llazzaroni/spatio-temporal-reading/main.py --train-baseline --data "/cluster/scratch/llazzaroni/spatio-temporal-reading-data"