#!/bin/bash
set -e
#################### 环境变量 ####################
export CUDA_VISIBLE_DEVICES='0,1'
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
export HF_HOME="~/data/hf-home"
export WANDB_PROJECT="mla_smollm_evaluation"

#################### 任务执行 ####################

torchrun --nproc_per_node 2 --master_port 24571 \
    -m src.mla_train_nt \
    --config-file ../configs/mla/rope_v2_start0_step8_svd_method7_rank32.yaml

feishu_msg -u 18055481550 -m 'rope_v2_start0_step8_svd_method7_rank32'
