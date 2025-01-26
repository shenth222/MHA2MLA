#!/bin/bash
set -e
#################### 环境变量 ####################
export CUDA_VISIBLE_DEVICES='3,4'
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
export HF_HOME="~/data/hf-home"
export WANDB_PROJECT="smollm_nt"

#################### 任务执行 ####################


torchrun --nproc_per_node 2 --master_port 24576 \
    -m src.low_rank_v.train \
    --config-file ../configs/low_rank/rope_v0_svd_v_method1_rank8.yaml

feishu_msg -u 18055481550 -m 'rope_v0_svd_v_method1_rank8'

./eval_mla_75_2.sh