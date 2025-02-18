#!/bin/bash
set -e
#################### 环境变量 ####################
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
export HF_HOME="~/data/hf-home"
export WANDB_PROJECT="mla_smollm_ft_nt"

#################### 任务执行 ####################

torchrun --nproc_per_node 4 --master_port 24577 \
    -m src.run_train \
    --config-file ../configs/rope/v1_2_cfg.yaml \
    --rope-cfg ../configs/rope/v1_2_rope.yaml

feishu_msg -u 18055481550 -m 'v1_2_rope'
