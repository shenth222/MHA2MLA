export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun ../src/test/2_norm.py \
    --config-file ../configs/continue_pretraining/test_2norm.yaml