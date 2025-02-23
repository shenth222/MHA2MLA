#!/bin/bash
export CUDA_VISIBLE_DEVICES="1"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=../:$PYTHONPATH
# export MASTER_PORT="auto"

torchrun --nproc_per_node 1 --master_port=25641 \
    -m src.test.test_2_norm \
    --config-file ../configs/test/135M_2norm.yaml \
    --output-dir .
