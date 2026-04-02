#!/bin/bash
#SBATCH --job-name=train_pss
#SBATCH --partition=gpua6000
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00          # keep short initially
#SBATCH --cpus-per-task=2        # minimal CPU
#SBATCH --mem=8G                 # conservative memory
#SBATCH --output=slurm_out/out_%j.log
#SBATCH --error=slurm_out/err_%j.log

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Activate environment
source ~/.bashrc
conda activate pss-env

# Run training for SPVFormer with resampled Soybean-MVSRS data.
python tools/train.py configs/Instance_seg/SPVFormer3d/SPVFormer3d_SMVSRS2_NPEC.py \
    --work-dir ./data/plant/SMVSRS2_NPEC/Inst_exp/SPVFormer_test