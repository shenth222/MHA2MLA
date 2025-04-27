import lm_eval
from lm_eval.utils import setup_logging, simple_parse_args_string
from lm_eval.loggers import EvaluationTracker
from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig, DataCollatorForLanguageModeling
import torch
import argparse
import math
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from datasets import load_dataset
import plotly.express as px
import random
from IPython import embed
from patch_rope import patch_partial_rope
import json
from lm_eval.utils import (
    handle_non_serializable,
    make_table,
    simple_parse_args_string,
)

def load_model(path, config):
    model = LlamaForCausalLM.from_pretrained(path, config=config)
    return model

def load_config(path):
    config = AutoConfig.from_pretrained(path)
    config._attn_implementation = 'eager'
    return config

def load_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_parser():
    parser = argparse.ArgumentParser(description="Benchmark for partial RoPE")
    model_choices = ["135m", "2-7b", "360m", "1b"]
    parser.add_argument("--model_name", type=str, required=True, choices=model_choices, help="Model name for test")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--bsz", type=int, default=1, help="Batch size for test")
    parser.add_argument("--save_states", action="store_true", help="Save qk states")
    parser.add_argument("--load_states", action="store_true", help="Load qk states")
    parser.add_argument("--load_path", type=str, help="Path to load qk states")
    parser.add_argument("--save_path", type=str, help="Path to save qk states")
    parser.add_argument("--figure_path", type=str, default="./figure/benchmark/", help="Path to save figure")
    rope_choices = ["high", "low", "uniform", "2-norm", "accumulate", "high-low", "full-rope", "contribution"]
    parser.add_argument("--rope_method", type=str, default="norm", choices=rope_choices, help="Partial RoPE method")
    parser.add_argument(
        "--hf_hub_log_args",
        type=str,
        default="",
        help="Comma separated string arguments passed to Hugging Face Hub's log function, e.g. `hub_results_org=EleutherAI,hub_repo_name=lm-eval-results`",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        default=None,
        type=str,
        metavar="DIR|DIR/file.json",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--log_samples",
        "-s",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis. Use with --output_path.",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    parser.add_argument(
        "--max_component",
        type=str2bool,
        default=False,
        required=False,
        help="Set max component in accumulate RoPE (True or False)"
    )
    return parser

MODEL_MAP = {
    "135m": "/data/shenth/models/SmolLM/135m",
    "360m": "/data/shenth/models/SmolLM/360m",
    "2-7b": "/data/shenth/models/llama/2-7b-hf",
    "1b": "/data/shenth/models/SmolLM/1b"
}


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    model_name = args.model_name
    task = args.task
    bsz = args.bsz
    save_states = args.save_states
    load_states = args.load_states
    load_path = args.load_path
    save_path = args.save_path
    figure_path = args.figure_path
    model_path = MODEL_MAP[model_name]
    rope_method = args.rope_method

    config = load_config(model_path)
    tokenizer = load_tokenizer(model_path)

    # initialize logging
    setup_logging("INFO") # optional, but recommended; or you can set up logging yourself
    rope_cfg = {
        "partial_rope_version": rope_method,
        "top_k_rope_dim": 4,
        "uniform_start_point": 0,
        "uniform_step": 4,
        "last_k_rope_dim": 4,
        "n_gqa_group": config.num_attention_heads // config.num_key_value_heads,
        "recovery_rate": 0.9,
        "max_component": args.max_component
    }
    patch_partial_rope(rope_cfg)
    # my_model = load_model(model_path, config) # create your model (could be running finetuning with some custom modeling code)

    ### test ###
    # config = load_config(model_path)
    # tokenizer = load_tokenizer(model_path)
    # model = load_model(model_path, config).cuda()
    # prompt = "Hey, are you conscious? Can you talk to me?"
    # inputs = tokenizer(prompt, return_tensors="pt")
    # generate_ids = model.generate(inputs.input_ids.to("cuda"), max_length=30)
    # outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # print(outputs)
    # exit()

    # instantiate an LM subclass that takes your initialized model and can run
    # - `Your_LM.loglikelihood()`
    # - `Your_LM.loglikelihood_rolling()`
    # - `Your_LM.generate_until()`
    # lm_obj = Your_LM(model=my_model, batch_size=16)
    # lm_obj = lm_eval.models.huggingface.HFLM(pretrained=model_path)

    # indexes all tasks from the `lm_eval/tasks` subdirectory.
    # Alternatively, you can set `TaskManager(include_path="path/to/my/custom/task/configs")`
    # to include a set of tasks in a separate directory.
    task_manager = lm_eval.tasks.TaskManager()

    if args.output_path:
        args.hf_hub_log_args += f",output_path={args.output_path}"
    if os.environ.get("HF_TOKEN", None):
        args.hf_hub_log_args += f",token={os.environ.get('HF_TOKEN')}"
    evaluation_tracker_args = simple_parse_args_string(args.hf_hub_log_args)
    evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)


    # Setting `task_manager` to the one above is optional and should generally be done
    # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
    # `simple_evaluate` will instantiate its own task_manager if it is set to None here.
    results = lm_eval.simple_evaluate( # call simple_evaluate
        model = "hf",
        model_args = f"pretrained={model_path}",
        tasks = [task],
        batch_size = "64", # auto:4
        device = "cuda",
        cache_requests = True,
        evaluation_tracker = evaluation_tracker,
        task_manager = task_manager
    )

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        dumped = json.dumps(
            results, indent=2, default=handle_non_serializable, ensure_ascii=False
        )
        if args.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        evaluation_tracker.save_results_aggregated(
            results=results, samples=samples if args.log_samples else None
        )

        if args.log_samples:
            for task_name, config in results["configs"].items():
                evaluation_tracker.save_results_samples(
                    task_name=task_name, samples=samples[task_name]
                )

        if (
            evaluation_tracker.push_results_to_hub
            or evaluation_tracker.push_samples_to_hub
        ):
            evaluation_tracker.recreate_metadata_card()

        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))