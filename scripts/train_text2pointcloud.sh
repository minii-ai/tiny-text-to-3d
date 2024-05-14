#!/bin/bash
python3 train_text2pointcloud.py \
    --dataset_dir=../data/modelnet40 \
    --augment_prob=0.2 \
    --subset=all \
    --diffusion_config=../configs/diffusion/text2pointcloud_tiny.json \
    --train_config=../configs/train/text2pointcloud_tiny.json \
    --save_dir=../checkpoints/text2pointcloud_tiny_modelnet40 \
    --num_workers=5 \
    --resume_checkpoint