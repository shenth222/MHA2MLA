#!/bin/bash
#################### 环境变量 ####################

export CUDA_VISIBLE_DEVICES="1"
export HF_HOME="/home/binguo/data/hf-home"
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
export MASTER_PORT="auto"
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
        --tokenizer_name /home/binguo/data/models/HuggingFaceTB/SmolLM-135M \
        --is_mla
# --multi_gpu 
    accelerate launch --num_processes=${NUM_GPUS} \
        -m src.evaluation.eval_mla --cfg_RoPE ${cfg_RoPE} \
        accelerate \
        --model_args "pretrained=${model_name_or_path}_hf,revision=main,dtype=bfloat16,max_length=2048" \
        --override_batch_size 96 \
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


eval_all ../checkpoints/rope_v2_start0_step8_svd_method7_rank16 rope_v2_start0_step8_svd_method7_rank16 ../configs/mla/rope_v2_start0_step8_svd_method7_rank16.yaml

eval_one_ckpt 