#!/bin/bash
set -e
#################### 环境变量 ####################
export CUDA_VISIBLE_DEVICES="0,1"
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
export HF_HOME="~/data/hf-home"
export WANDB_PROJECT="smollm_nt"

#################### 任务执行 ####################


torchrun --nproc_per_node 2 --master_port 24564 \
    -m src.run_train \
    --config-file ../configs/continue_pretraining/135M_lr_1e-4_cooldown_10_subset.yaml \
    --rope-cfg ../configs/continue_pretraining/rope_v0.yaml

feishu_msg -u 18055481550 -m 'rope_v0_subset'

./eval_partial_rope_all_63.sh