#!/bin/bash
set -e
#################### 环境变量 ####################
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
export HF_HOME="/cpfs01/user/jitao/"
export WANDB_PROJECT="smollm_nt"
export WANDB_API_KEY="4bdf7e746b5e3bb922d6c3675da23e2a1d2b642f"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export TRITON_DEBUG=1

#################### 任务执行 ####################

torchrun --nproc_per_node=8 --master_port 24558 \
    -m src.run_train \
    --config-file ../configs/continue_pretraining/7B_1.yaml \
    --rope-cfg ../configs/continue_pretraining/rope_v0.yaml