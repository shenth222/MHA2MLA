#!/bin/bash
set -e
#################### Environment ####################
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
export HF_HOME="~/data/hf-home"
export WANDB_PROJECT="mla_smollm_ft_nt"

#################### Main ####################

# torchrun --nproc_per_node 1 --master_port 24575 \
#     ../src/mha2mla/2_norm.py \
#     --config_file ../configs_hf/rope/135M_2GPU.yaml \
#     --output_dir ./qk_tensor_hf_test.pth \
#     --sample_size 1024

torchrun --nproc_per_node 2 --master_port 24575 \
    ../src/mha2mla/run_train.py \
    --config_file ../configs_hf/rope/135M_2GPU.yaml \
    --partial_rope_config ../configs_hf/rope/v4_topk4_rope.yaml
