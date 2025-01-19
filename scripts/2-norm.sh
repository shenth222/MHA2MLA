#!/bin/bash
export CUDA_VISIBLE_DEVICES="2"
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export MASTER_PORT="auto"
export HF_HOME="~/data/hf-home"


torchrun --nproc_per_node 1 --master_port=25641 \
    ../src/test/test_2_norm.py \
    --config-file ../configs/test/test_2norm.yaml