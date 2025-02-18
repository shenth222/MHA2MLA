#!/bin/bash
set -e
#################### 环境变量 ####################
export CUDA_VISIBLE_DEVICES='4,5,6,7'
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
export HF_HOME="~/data/hf-home"
export WANDB_PROJECT="mla_smollm_ft_nt"
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

#################### 函数定义 ####################

eval_one_ckpt() {
    local model_name_or_path=$1
    local output_dir=$2
    local cfg_RoPE=$3

    torchrun --nproc_per_node=1 \
        -m src.conversation.convert_nanotron_to_hf \
        --checkpoint_path ${model_name_or_path} \
        --save_path "${model_name_or_path}_hf" \
        --tokenizer_name /home/binguo/data/models/HuggingFaceTB/SmolLM-360M \
        --is_mla
 
    accelerate launch --multi_gpu --num_processes=${NUM_GPUS} \
        -m src.evaluation.eval_mla --cfg_RoPE ${cfg_RoPE} \
        accelerate \
        --model_args "pretrained=${model_name_or_path}_hf,revision=main,dtype=bfloat16,max_length=2048" \
        --override_batch_size 48 \
        --custom_tasks "../src/evaluation/tasks.py" \
        --tasks "../src/evaluation/smollm1_base_v2.txt" \
        --output_dir "../eval_results/${output_dir}"
}

#################### 任务执行 ####################

MODEL_NAME="360M_rope_v2_start0_step8_svd_method7_rank8"

torchrun --nproc_per_node 4 --master_port 24573 \
    -m src.mla_train_nt \
    --config-file ../configs/mla/${MODEL_NAME}.yaml

feishu_msg -u 18055481550 -m "${MODEL_NAME}"

eval_one_ckpt ../checkpoints/${MODEL_NAME}/18000 ${MODEL_NAME} ../configs/mla/${MODEL_NAME}.yaml

feishu_msg -u 18055481550 -m "finish evaluation ${MODEL_NAME}"



MODEL_NAME="360M_rope_v4_topk4_svd_method7_rank32"

torchrun --nproc_per_node 4 --master_port 24573 \
    -m src.mla_train_nt \
    --config-file ../configs/mla/${MODEL_NAME}.yaml

feishu_msg -u 18055481550 -m "${MODEL_NAME}"

eval_one_ckpt ../checkpoints/${MODEL_NAME}/18000 ${MODEL_NAME} ../configs/mla/${MODEL_NAME}.yaml

feishu_msg -u 18055481550 -m "finish evaluation ${MODEL_NAME}"
