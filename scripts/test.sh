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


# torchrun --master_port=29564 --nproc_per_node=1 \
#     -m src.auto_encoder.train \
#     --config-file ../configs/ae/test.yaml

# torchrun --master_port=29564 --nproc_per_node=1 \
#     -m src.conversation.convert_nanotron_to_hf \
#     --checkpoint_path "../checkpoints/test_nt/0" \
#     --save_path "../checkpoints/test_nt/0_hf" \
#     --tokenizer_name /home/binguo/data/models/HuggingFaceTB/SmolLM-135M \
#     --auto_encoder

# torchrun --master_port=29564 --nproc_per_node=1 \
#     -m src.conversation.convert_hf_to_nanotron \
#     --checkpoint_path "../checkpoints/test_nt/0_hf" \
#     --save_path "../checkpoints/test_nt/0_hf_nt" \
#     --auto_encoder

# torchrun --master_port=29564 --nproc_per_node=1 \
#     -m src.test.test_conversation \
#     --config-file ../configs/ae/test.yaml


# torchrun --master_port=29564 --nproc_per_node=1 \
#     -m src.test.test_cache \
#     --model_name /home/binguo/data/MLA-FT/checkpoints/rope_v4_topk4_svd_method7_rank32/18000_hf \
#     --is_mla


torchrun --master_port=29564 --nproc_per_node=1 \
    -m src.test.test_cache \
    --model_name /home/binguo/data/models/HuggingFaceTB/SmolLM-135M

torchrun --master_port=29564 --nproc_per_node=1 \
    -m src.test.test_cache \
    --model_name /home/binguo/data/MLA-FT/checkpoints/rope_v4_topk4_svd_method7_rank8/18000_hf \
    --is_mla
# 12.57 9.90
# 17.44 11.92
# 4.87 2.02