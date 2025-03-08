#################### 环境变量 ####################

export CUDA_VISIBLE_DEVICES="3"
export HF_HOME="~/data/hf-home"
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
export MASTER_PORT="auto"
export PYTHONPATH=..:$PYTHONPATH
# export HF_ENDPOINT=https://hf-mirror.com

#################### 任务执行 ####################

python ../src/mha2mla/bechmark_latency_memory.py