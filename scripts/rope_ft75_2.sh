#!/bin/bash
set -e
#################### 环境变量 ####################
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
export HF_HOME="~/data/hf-home"
export WANDB_PROJECT="smollm_nt"

#################### 任务执行 ####################

torchrun --nproc_per_node 4 --master_port 24577 \
    -m src.run_train \
    --config-file ../configs/rope/360M_v4_topk4_cfg.yaml \
    --rope-cfg ../configs/rope/360M_v4_topk4_rope.yaml

feishu_msg -u 18055481550 -m '360M_v4_topk4_rope'

torchrun --nproc_per_node 4 --master_port 24577 \
    -m src.run_train \
    --config-file ../configs/rope/360M_v5_last4_cfg.yaml \
    --rope-cfg ../configs/rope/360M_v5_last4_rope.yaml

feishu_msg -u 18055481550 -m '360M_v4_topk4_rope'

