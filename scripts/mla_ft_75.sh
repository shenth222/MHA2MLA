#!/bin/bash
set -e
#################### 环境变量 ####################
export CUDA_VISIBLE_DEVICES='4,5,6,7'
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
export HF_HOME="~/data/hf-home"
export WANDB_PROJECT="smollm_nt"

#################### 任务执行 ####################

torchrun --nproc_per_node 4 --master_port 24575 \
    -m src.low_rank_k_nope.train \
    --config-file ../configs/low_rank/rope_v0_svd_v_method1_rank8_rd_init.yaml

feishu_msg -u 18055481550 -m 'rope_v0_svd_v_method1_rank8_rd_init'

./eval_low_rank_all_75.sh
