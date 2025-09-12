#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0
wandb offline
wandb disabled
python tools/test.py  configs/Semantic_seg/PointNet2/pointnet2_COS.py \
    'checkpoints/best_miou_epoch_192.pth' --save-local --task='lidar_seg' 
