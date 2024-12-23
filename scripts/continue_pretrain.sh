#!/bin/bash
set -e
#################### 环境变量 ####################
export CUDA_VISIBLE_DEVICES='0,1,2,3'
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
export HF_HOME="/home/binguo/data/hf-home"


#################### 任务执行 ####################

torchrun --nproc_per_node 4 \
    ../modules/nanotron/run_train.py \
    --config-file ../configs/continue_pretraining/135M_lr_5e-6.yaml

feishu_msg -u 18055481550 -m 'lr_5e-6'