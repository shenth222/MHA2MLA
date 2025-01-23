#!/bin/bash
set -e
#################### 环境变量 ####################
export CUDA_VISIBLE_DEVICES='0,1,2,3'
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
export HF_HOME="~/data/hf-home"
export WANDB_PROJECT="smollm_nt"

#################### 任务执行 ####################


torchrun --nproc_per_node 4 --master_port 24558 \
    -m src.mla_train_nt \
    --config-file ../configs/mla/rope_v2_start0_step8_svd_method2_rank8_step36000.yaml

feishu_msg -u 18055481550 -m 'rope_v2_start0_step8_svd_method2_rank8_step36000'

torchrun --nproc_per_node 4 --master_port 24558 \
    -m src.mla_train_nt \
    --config-file ../configs/mla/360M_rope_v2_start0_step8_svd_method7_rank8.yaml

feishu_msg -u 18055481550 -m '360M_rope_v2_start0_step8_svd_method7_rank8'
