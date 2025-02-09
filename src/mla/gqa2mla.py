import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaConfig, LlamaForCausalLM
import argparse
import yaml


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def gqa2mla_ckpt(config):
    ckpt_0_path = os.path.join(config["TrainingArguments"]["output_dir"], "checkpoint-0")
    print(ckpt_0_path)
    # load the old ckpt
    tokenizer=AutoTokenizer.from_pretrained(config["ModelArguments"]["model_name_or_path"])
    model = AutoModelForCausalLM.from_pretrained(config["ModelArguments"]["model_name_or_path"])
    tokenizer.save_pretrained(ckpt_0_path)
    model.save_pretrained(ckpt_0_path)
    state_dict = model.state_dict()
    # load and save the new model 
    from .mla_patch_hf import mla_patch_hf,state_dict_svd_init
    mla_patch_hf(config["ModelConfig"]["RoPE"])
    model_config = LlamaConfig(**config["ModelConfig"])
    model_mla= LlamaForCausalLM(model_config)
    state_dict_mla = state_dict_svd_init(model_mla,state_dict)
    model_mla.load_state_dict(state_dict_mla)
    model_mla.save_pretrained(ckpt_0_path)
    config["ModelArguments"]["model_name_or_path"] = ckpt_0_path
    return config

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    config = load_config(args.config_file)
    gqa2mla_ckpt(config)
