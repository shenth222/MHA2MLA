#!/bin/bash
set -e
#################### 环境变量 ####################
export CUDA_VISIBLE_DEVICES='2,3'
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
export HF_HOME="~/data/hf-home"
export WANDB_PROJECT="smollm_nt"

#################### 任务执行 ####################


torchrun --nproc_per_node 2 --master_port 24558 \
    -m src.run_train \
    --config-file ../configs/rope/v3_top2_last2_cfg.yaml \
    --rope-cfg ../configs/rope/v3_top2_last2_rope.yaml

feishu_msg -u 18055481550 -m 'v3_top2_last2_rope'

./eval_partial_rope_all_58.sh