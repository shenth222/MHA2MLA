#!/bin/bash
set -e
#################### Environment ####################
export CUDA_VISIBLE_DEVICES='0,1'
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
export HF_HOME="~/data/hf-home"
export WANDB_PROJECT="mla_smollm_ft_nt"
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

#################### Main ####################

# torchrun --nproc_per_node 1 --master_port 24571 \
#     ../src/mha2mla/2_norm.py \
#     --config_file ../configs_hf/rope/135M_2GPU.yaml \
#     --output_dir ./qk_tensor_hf_test_2.pth \
#     --sample_size 1024

torchrun --nproc_per_node 2 --master_port 24571 \
    ../src/mha2mla/run_train.py \
    --config_file ../configs_hf/rope/135M_2GPU.yaml \
    --partial_rope_config ../configs_hf/rope/v4_topk4_rope.yaml \
    --svd_config ../configs_hf/rope/svd_method7_rank8.yaml
