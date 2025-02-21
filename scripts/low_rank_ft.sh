#!/bin/bash
set -e
#################### 环境变量 ####################
export CUDA_VISIBLE_DEVICES='0,1,2,3'
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
export HF_HOME="~/data/hf-home"
export WANDB_PROJECT="mla_smollm_ft_nt"

#################### 任务执行 ####################

set -e

# torchrun --nproc_per_node 1 --master_port 24571 \
#     -m src.auto_encoder.init \
#     --config_file ../configs/ae/init_hf.yaml


torchrun --nproc_per_node 1 --master_port 24558 \
    -m src.auto_encoder.merge \
    --original_model_path /home/binguo/data/models/HuggingFaceTB/SmolLM-135M_nt \
    --ae_model_path ../checkpoints/rope_v4_topk4_ae_v2_rank8_null_init/checkpoint-2000 \
    --save_path ../checkpoints/rope_v4_topk4_ae_v2_rank8_null/0


torchrun --nproc_per_node 4 --master_port 24558 \
    -m src.auto_encoder.train \
    --config-file ../configs/ae/rope_v4_topk4_ae_v2_rank8_null.yaml
