#################### 环境变量 ####################

export CUDA_VISIBLE_DEVICES="3"
model_name_or_path="/home/binguo/data/models/HuggingFaceTB/SmolLM-1.7B"
export HF_HOME="/home/binguo/data/hf-home"
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
export MASTER_PORT="auto"


#################### 任务执行 ####################


# torchrun --nproc_per_node=1 ../modules/nanotron/examples/llama/convert_nanotron_to_hf.py \
#     --checkpoint_path ${model_name_or_path} \
#     --save_path "${model_name_or_path}_hf"

torchrun --nproc_per_node=1 --master_port 24558 \
    ../modules/nanotron/examples/llama/convert_hf_to_nanotron.py \
    --checkpoint_path ${model_name_or_path} \
    --save_path "${model_name_or_path}_nt"
