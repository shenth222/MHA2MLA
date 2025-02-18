#!/bin/bash
set -e
#################### 环境变量 ####################
export CUDA_VISIBLE_DEVICES='0,1'
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
export HF_HOME="~/data/hf-home"
export WANDB_PROJECT="mla_smollm_ft_nt"

# export AE_LOSS="L1"

#################### 任务执行 ####################

set -e

# torchrun --nproc_per_node 1 --master_port 24571 \
#     -m src.auto_encoder.init \
#     --config_file ../configs/ae/init_hf.yaml


# torchrun --nproc_per_node 2 --master_port 24571 \
#     -m src.auto_encoder.train \
#     --config-file ../configs/ae/rope_v4_topk4_ae_v3_rank8_null.yaml

torchrun --master_port=29564 --nproc_per_node=1 \
    -m src.conversation.convert_nanotron_to_hf \
    --checkpoint_path "../checkpoints/rope_v4_topk4_ae_v3_rank8_null/0" \
    --save_path "../checkpoints/rope_v4_topk4_ae_v3_rank8_null/0_hf" \
    --auto_encoder


torchrun --nproc_per_node 1 --master_port 24571 \
    -m src.auto_encoder.merge \
    --model_name_path ../checkpoints/rope_v4_topk4_ae_v3_rank8_null

torchrun --master_port=29564 --nproc_per_node=1 \
    -m src.conversation.convert_hf_to_nanotron \
    --checkpoint_path "../checkpoints/rope_v4_topk4_ae_v3_rank8_null/0_hf" \
    --save_path "../checkpoints/rope_v4_topk4_ae_v3_rank8_null/0" \
    --auto_encoder

torchrun --nproc_per_node 2 --master_port 24571 \
    -m src.auto_encoder.train \
    --config-file ../configs/ae/rope_v4_topk4_ae_v3_rank8_null.yaml
