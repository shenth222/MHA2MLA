#!/bin/bash
#################### 环境变量 ####################

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_HOME="~/data/hf-home"
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
export MASTER_PORT="auto"
export PYTHONPATH=..:$PYTHONPATH

#################### 函数定义 ####################

eval_one_ckpt() {
    local model_name_or_path=$1
    local output_dir=$2
    local cfg_RoPE=$3

    # torchrun --nproc_per_node=1 \
    #     -m src.original_conversation.convert_nanotron_to_hf \
    #     --checkpoint_path ${model_name_or_path} \
    #     --save_path "${model_name_or_path}_hf" \
    #     --tokenizer_name ~/data/models/HuggingFaceTB/SmolLM-135M

    accelerate launch --multi_gpu --num_processes=${NUM_GPUS} \
        ../src/mha2mla/eval.py --partial_rope_config ${cfg_RoPE} --is_mla \
        accelerate \
        --model_args "pretrained=${model_name_or_path},revision=main,dtype=bfloat16,max_length=2048" \
        --override_batch_size 48 \
        --custom_tasks "../src/mha2mla/tasks.py" \
        --tasks "../src/mha2mla/smollm1_base.txt" \
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

eval_one_ckpt /home/binguo/data/MLA-FT/checkpoints/test/checkpoint-18000 hf_test ../configs_hf/rope/v4_topk4_rope.yaml
