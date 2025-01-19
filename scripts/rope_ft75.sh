#!/bin/bash
set -e
#################### 环境变量 ####################
export CUDA_VISIBLE_DEVICES="2,3,4,5"
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
export HF_HOME="~/data/hf-home"
export WANDB_PROJECT="smollm_nt"

#################### 任务执行 ####################


# torchrun --nproc_per_node 4 --master_port 24575 \
#     -m src.run_train \
#     --config-file ../configs/rope/v4_topk4_cfg_alter.yaml \
#     --rope-cfg ../configs/rope/v4_topk4_rope.yaml

# feishu_msg -u 18055481550 -m 'v4_topk4_cfg_alter'



torchrun --nproc_per_node 4 --master_port 24575 \
    -m src.run_train \
    --config-file ../configs/rope/v4_topk4_cfg_alter.yaml \
    --rope-cfg ../configs/rope/v4_topk4_rope.yaml

feishu_msg -u 18055481550 -m 'v4_topk4_rope_alter'

./eval_partial_rope_all_75.sh
