#!/bin/bash
python3 train_uncond_sine_cosine.py \
    --train_config=../configs/train/ddpm_sine_cosine_uncond.json \
    --ddpm_config=../configs/ddpm/ddpm_sine_cosine_uncond.json \
    --save_dir=../checkpoints/ddpm_sine_cosine_uncond 