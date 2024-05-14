#!/bin/bash
python3 prepare_modelnet.py \
    --root_dir=../data/ModelNet40 \
    --output_dir=../data/modelnet40 \
    --num_workers=20