#!/bin/bash
set -e
#################### 环境变量 ####################
export CUDA_VISIBLE_DEVICES='2,3,4,5'
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
export HF_HOME="~/data/hf-home"
export WANDB_PROJECT="mla_smollm_ft_nt"

# export AE_LOSS="L1"

#################### 任务执行 ####################

set -e

# torchrun --nproc_per_node 1 --master_port 24577 \
#     -m src.low_rank_v_m3.auto_encoder_init \
#     --config_file ../configs/low_rank/ae_m3_init_hf.yaml


torchrun --nproc_per_node 1 --master_port 24563 \
    -m src.low_rank_v_m3.auto_encoder_init

torchrun --nproc_per_node 1 --master_port 24563 \
    -m src.conversation.convert_hf_to_nanotron \
    --checkpoint_path ../checkpoints/rope_v0_svd_v_method2_rank8_silu_auto_encoder_l1/0_hf \
    --save_path ../checkpoints/rope_v0_svd_v_method2_rank8_silu_auto_encoder_l1/0 \
    --is_low_rank_v_m3

feishu_msg -u 18055481550 -m 'init'
