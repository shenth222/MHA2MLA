#!/bin/bash
set -e
#################### 环境变量 ####################
export CUDA_VISIBLE_DEVICES='5,6'
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
export HF_HOME="~/data/hf-home"
export WANDB_PROJECT="smollm_nt"

#################### 任务执行 ####################

torchrun --nproc_per_node 2 --master_port 24575 \
    -m src.low_rank_k_nope.train \
    --config-file ../configs/low_rank/v2_start0_step8_svd_method2_W_k_nope_rank8.yaml

feishu_msg -u 18055481550 -m 'v2_start0_step8_svd_method2_W_k_nope_rank8'

./eval_low_rank_all_75.sh
