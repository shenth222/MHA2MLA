#!/bin/bash
set -e
#################### 环境变量 ####################
export CUDA_VISIBLE_DEVICES='2'
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
export HF_HOME="/home/binguo/data/hf-home"
export PYTHONPATH=..:$PYTHONPATH
# export CUDA_HOME=/usr/local/cuda-12.1
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


#################### 任务执行 ####################

# torchrun --nproc_per_node 1 \
#     -m src.test.test_mla

# torchrun --master_port=29564 --nproc_per_node=1 \
#     -m src.conversation.convert_hf_to_nanotron \
#     --checkpoint_path="../checkpoints/rope_v4_topk4_svd_method7_rank8/18000_hf" \
#     --save_path="../checkpoints/test_nt" \
#     --is_mla

# torchrun --master_port=29564 --nproc_per_node=1 \
#     -m src.conversation.convert_nanotron_to_hf \
#     --checkpoint_path="../checkpoints/test_nt" \
#     --tokenizer_name="../checkpoints/rope_v4_topk4_svd_method7_rank8/18000_hf" \
#     --save_path="../checkpoints/test_hf" \
#     --is_mla


# torchrun --master_port=29564 --nproc_per_node=1 \
#     -m src.conversation.convert_nanotron_to_hf \
#     --checkpoint_path="../checkpoints/test_nt" \
#     --tokenizer_name="../checkpoints/rope_v4_topk4_svd_method2_rank8_hf/checkpoint-18000" \
#     --save_path="../checkpoints/test_hf" \
#     --is_mla


# torchrun --master_port=29564 --nproc_per_node=1 \
#     -m src.test.test_conversation \
#     --config-file "../configs/mla/test.yaml"


# torchrun --master_port=29564 --nproc_per_node=1 \
#     -m src.test.dbg_attn_fwd

torchrun --master_port=29564 --nproc_per_node=1 \
    -m src.low_rank_v.test \
    --config-file ../configs/low_rank/test.yaml