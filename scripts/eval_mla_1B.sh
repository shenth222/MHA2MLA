#!/bin/bash
#################### 环境变量 ####################

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
export MASTER_PORT="auto"
export PYTHONPATH=..:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="/cpfs01/user/jitao/hf_home/"
export PYTHONPATH=..:$PYTHONPATH

#################### 函数定义 ####################

eval_one_ckpt() {
    local model_name_or_path=$1
    local output_dir=$2
    local cfg_RoPE=$3

    torchrun --nproc_per_node=1 \
        -m src.conversation.convert_nanotron_to_hf \
        --checkpoint_path ${model_name_or_path} \
        --save_path "${model_name_or_path}_hf" \
        --tokenizer_name ../checkpoints/HuggingFaceTB/SmolLM-1.7B \
        --is_mla

<<<<<<< HEAD

=======
>>>>>>> feature/low-rank-approx
    accelerate launch --num_processes=${NUM_GPUS} \
        -m src.evaluation.eval_mla --cfg_RoPE ${cfg_RoPE} \
        accelerate \
        --model_args "pretrained=${model_name_or_path}_hf,revision=main,dtype=bfloat16,max_length=2048" \
        --override_batch_size 200 \
        --custom_tasks "../src/evaluation/tasks.py" \
        --tasks "../src/evaluation/smollm1_base_v2.txt" \
        --output_dir "../eval_results/${output_dir}"
}

eval_all() {
    local model_name_path=$1
    local output_dir=$2
    local cfg_RoPE=$3

    # eval所有检查点
    matching_directories=$(find "$model_name_path" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]+')

    echo $matching_directories

    for dir in $matching_directories; do
        echo "Evaluating $dir"
        eval_one_ckpt $dir $output_dir $cfg_RoPE
    done
}

#################### 任务执行 ####################
set -e

export MODEL_NAME="1.7B_rope_v2_start0_step8_svd_method2_rank4"
eval_one_ckpt ../checkpoints/${MODEL_NAME}/12000 "${MODEL_NAME}" ../configs/mla/${MODEL_NAME}.yaml
