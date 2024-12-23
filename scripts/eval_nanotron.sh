#!/bin/bash
#################### 环境变量 ####################

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_HOME="/home/binguo/data/hf-home"
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
export MASTER_PORT="auto"

#################### 任务执行 ####################


torchrun --standalone --nnodes=1 --nproc-per-node=4  \
    -m lighteval nanotron \
    --checkpoint_config_path /home/binguo/data/models/HuggingFaceTB/SmolLM-135M_nanotron/config.yaml \
    --lighteval_config_path /home/binguo/data/MLA-FN/configs/eval/lighteval_config.yaml