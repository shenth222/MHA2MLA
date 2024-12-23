#!/bin/bash
set -e
#################### 环境变量 ####################
export CUDA_VISIBLE_DEVICES='0,1,2,3'
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
export HF_HOME="/home/binguo/data/hf-home"
export PYTHONPATH=..:$PYTHONPATH


#################### 任务执行 ####################

torchrun --nproc_per_node 4 \
    -m src.run_train \
    --config-file ../configs/rope_ft/v1_16.yaml \
    --rope-cfg ../configs/rope/v1_16.yaml

feishu_msg -u 18055481550 -m 'v1_16'