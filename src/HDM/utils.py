from loguru import logger
import torch
from transformers import AutoModelForCausalLM

def load_HDM_res(load_path="./HDM_res", mode="Q", rank=32):
    logger.info(f"Load {mode} weights")
    path = f"{load_path}/{mode}/W1_{mode}_{rank}.pkl"
    W1 = torch.load(path, weights_only=True)
    path = f"{load_path}/{mode}/H1_{mode}_{rank}.pkl"
    H1 = torch.load(path, weights_only=True)
    path = f"{load_path}/{mode}/W2_{mode}_{rank}.pkl"
    W2 = torch.load(path, weights_only=True)
    path = f"{load_path}/{mode}/H2_{mode}_{rank}.pkl"
    H2 = torch.load(path, weights_only=True)
    return W1, H1, W2, H2

def load_model(config, devices="cpu"):
    model_name = "/data/shenth/models/llama/2-7b-hf"  # Or any other Llama checkpoint available
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, config=config)
    model = model.eval().to(devices)
    return model