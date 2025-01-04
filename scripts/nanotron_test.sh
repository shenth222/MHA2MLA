#!/bin/bash
set -e
#################### 环境变量 ####################
export CUDA_VISIBLE_DEVICES='0,1'
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
export HF_HOME="/home/binguo/data/hf-home"
export PYTHONPATH=..:$PYTHONPATH

pytest ../modules/nanotron/examples/llama/tests/test_conversion.py