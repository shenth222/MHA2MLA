#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export MASTER_PORT="auto"

torchrun --nproc_per_node 1 --master_port=25641 \
    ../src/test/test_2_norm.py \
    --config-file ../configs/test/1B_2norm.yaml
