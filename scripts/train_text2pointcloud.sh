#!/bin/bash
python3 train_text2pointcloud.py \
    --dataset_dir=../data/modelnet10 \
    --subset=all \
    --diffusion_config=../configs/diffusion/dit_tiny_xs.json \
    --train_config=../configs/train/text2pointcloud_modelnet10.json \
    --save_dir=../checkpoints/awesome-grasshopper-modelnet10 \
    --num_workers=5 \
    --resume_checkpoint

# python3 train_text2pointcloud.py \
#     --dataset_dir=../data/modelnet40 \
#     --subset=all \
#     --diffusion_config=../configs/diffusion/dit_s.json \
#     --train_config=../configs/train/modelnet40.json \
#     --save_dir=../checkpoints/adapted-prawn-modelnet40 \
#     --num_workers=5 \
#     --resume_checkpoint