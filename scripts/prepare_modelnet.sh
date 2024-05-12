#!/bin/bash
python3 prepare_modelnet.py \
    --root_dir=../data/ModelNet10 \
    --output_dir=../data/modelnet10 \
    --num_workers=20