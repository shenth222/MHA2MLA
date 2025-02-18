#!/bin/bash
set -e
#################### 环境变量 ####################
export CUDA_VISIBLE_DEVICES="1,2"
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
export HF_HOME="~/data/hf-home"
export WANDB_PROJECT="mla_smollm_ft_nt"
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')


eval_one_ckpt() {
    local model_name_or_path=$1
    local output_dir=$2
    local cfg_RoPE=$3

    torchrun --nproc_per_node=1 \
        -m src.original_conversation.convert_nanotron_to_hf \
        --checkpoint_path ${model_name_or_path} \
        --save_path ${model_name_or_path}_hf \
        --tokenizer_name /home/binguo/data/models/HuggingFaceTB/SmolLM-135M

    accelerate launch --multi_gpu --num_processes=${NUM_GPUS} --main_process_port 25675 \
        -m src.evaluation.eval_partial_rope --cfg_RoPE ${cfg_RoPE} \
        accelerate \
        --model_args "pretrained=${model_name_or_path}_hf,revision=main,dtype=bfloat16,max_length=2048" \
        --override_batch_size 96 \
        --custom_tasks "../src/evaluation/tasks.py" \
        --tasks "../src/evaluation/smollm1_base_v2.txt" \
        --output_dir "../eval_results/${output_dir}"
}

#################### 任务执行 ####################




torchrun --nproc_per_node 2 --master_port 24575 \
    -m src.run_train \
    --config-file ../configs/rope/v2_start0_step16_cfg.yaml \
    --rope-cfg ../configs/rope/v2_start0_step16_rope.yaml

feishu_msg -u 18055481550 -m 'v2_start0_step16_rope'


eval_one_ckpt ../checkpoints/v2_start0_step16_rope/18000 v2_start0_step16_rope ../configs/rope/v2_start0_step16_rope.yaml

