#!/bin/bash
set -e
<<<<<<< HEAD
#################### 环境变量 ####################
export CUDA_VISIBLE_DEVICES='2,3'
=======
#################### Environment ####################
export CUDA_VISIBLE_DEVICES='0,1'
>>>>>>> feature/low-rank-approx
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
export HF_HOME="~/data/hf-home"
export WANDB_PROJECT="mla_smollm_ft_nt"
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

<<<<<<< HEAD
#################### 函数定义 ####################

eval_one_ckpt() {
    local model_name_or_path=$1
    local output_dir=$2
    local cfg_RoPE=$3

    torchrun --nproc_per_node=1 \
        -m src.conversation.convert_nanotron_to_hf \
        --checkpoint_path ${model_name_or_path} \
        --save_path "${model_name_or_path}_hf" \
        --tokenizer_name ~/data/models/HuggingFaceTB/SmolLM-135M \
        --is_mla
 
    accelerate launch --multi_gpu --num_processes=${NUM_GPUS} \
        -m src.evaluation.eval_mla --cfg_RoPE ${cfg_RoPE} \
        accelerate \
        --model_args "pretrained=${model_name_or_path}_hf,revision=main,dtype=bfloat16,max_length=2048" \
        --override_batch_size 96 \
        --custom_tasks "../src/evaluation/tasks.py" \
        --tasks "../src/evaluation/smollm1_base_v2.txt" \
        --output_dir "../eval_results/${output_dir}"
}

#################### 任务执行 ####################


MODEL_NAME="360M_rope_v4_topk4_svd_method7_rank32"

torchrun --nproc_per_node 2 --master_port 24559 \
    -m src.mla_train_nt \
    --config-file ../configs/mla/${MODEL_NAME}.yaml

eval_one_ckpt ../checkpoints/${MODEL_NAME}/24000 ${MODEL_NAME} ../configs/mla/${MODEL_NAME}.yaml
=======
#################### Main ####################

# torchrun --nproc_per_node 1 --master_port 24571 \
#     ../src/mha2mla/2_norm.py \
#     --config_file ../configs_hf/rope/135M_2GPU.yaml \
#     --output_dir ./qk_tensor_hf_test_2.pth \
#     --sample_size 1024

torchrun --nproc_per_node 2 --master_port 24571 \
    ../src/mha2mla/run_train.py \
    --config_file ../configs_hf/rope/135M_2GPU.yaml \
    --partial_rope_config ../configs_hf/rope/v4_topk4_rope.yaml \
    --svd_config ../configs_hf/rope/svd_method7_rank8.yaml
>>>>>>> feature/low-rank-approx
