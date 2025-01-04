#!/bin/bash
set -e
#################### 环境变量 ####################
export CUDA_VISIBLE_DEVICES='0,1,2,3'
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
export HF_HOME="/home/binguo/data/hf-home"
export PYTHONPATH=..:$PYTHONPATH
# export CUDA_HOME=/usr/local/cuda-12.1
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


#################### 任务执行 ####################

torchrun --nproc_per_node 1 \
    -m src.run_train \
    --config-file ../configs/test/test.yaml \
    --rope-cfg ../configs/rope/v1_16.yaml