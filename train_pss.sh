#!/bin/bash
#SBATCH --job-name=train_pss
#SBATCH --mail-user=m.t.verweij@students.uu.nl
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpua6000
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm_out/%x_%j.log
#SBATCH --error=slurm_out/%x_%j.err

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Activate environment
source ~/.bashrc
conda activate pss-env

# Arguments
CONFIG=$1
WORKDIR=$2

echo "Config: $CONFIG"
echo "Workdir: $WORKDIR"

python tools/train.py $CONFIG --work-dir $WORKDIR