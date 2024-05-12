#!/bin/bash
python3 train_text2pointcloud.py \
    --dataset_dir=../data/modelnet10 \
    --augment_prob=0.0 \
    --subset=table \
    --diffusion_config=../configs/diffusion/text2pointcloud_tiny.json \
    --train_config=../configs/train/text2pointcloud_tiny.json \
    --save_dir=../checkpoints/text2pointcloud_tiny \