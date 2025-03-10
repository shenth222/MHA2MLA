#!/bin/bash
set -e
#################### 环境变量 ####################
<<<<<<< HEAD
export CUDA_VISIBLE_DEVICES='0,1,2,3'
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
export HF_HOME="~/data/hf-home"
=======
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
>>>>>>> feature/low-rank-approx
export WANDB_PROJECT="mla_smollm_ft_nt"

#################### 任务执行 ####################

<<<<<<< HEAD
set -e

# torchrun --nproc_per_node 1 --master_port 24571 \
#     -m src.auto_encoder.init \
#     --config_file ../configs/ae/init_hf.yaml
=======
>>>>>>> feature/low-rank-approx


torchrun --nproc_per_node 1 --master_port 24558 \
    -m src.auto_encoder.merge \
<<<<<<< HEAD
    --original_model_path ~/data/models/HuggingFaceTB/SmolLM-135M_nt \
    --ae_model_path ../checkpoints/rope_v4_topk4_ae_v2_rank8_null_init/checkpoint-2000 \
    --save_path ../checkpoints/rope_v4_topk4_ae_v2_rank8_null/0


torchrun --nproc_per_node 4 --master_port 24558 \
    -m src.auto_encoder.train \
    --config-file ../configs/ae/rope_v4_topk4_ae_v2_rank8_null.yaml
=======
    --original_model_path ../checkpoints/HuggingFaceTB/SmolLM-1.7B_nt \
    --ae_model_path ../checkpoints/1.7B_rope_v4_topk4_ae_v2_rank8_null_init/checkpoint-2000 \
    --save_path ../checkpoints/1.7B_rope_v4_topk4_ae_v2_rank8_null/0

torchrun --nproc_per_node 8 --master_port 24558 \
    -m src.auto_encoder.train \
    --config-file ../configs/ae/1.7B_rope_v4_topk4_ae_v2_rank8_null.yaml
>>>>>>> feature/low-rank-approx
